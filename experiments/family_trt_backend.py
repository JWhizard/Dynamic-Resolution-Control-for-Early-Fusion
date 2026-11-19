"""Complete Torchvision detection graphs on TensorRT; exact CPU input transform.

Runs the CNN, heads, proposal/ROI processing and NMS on GPU. CPU operations
are original normalization/resize/padding and final native-coordinate mapping.
"""
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from experiments.evaluate_families import sha256
from experiments.tensorrt_backend import Cuda


class Session:
    def __init__(self,manifest):
        import tensorrt as trt
        self.trt=trt;self.cuda=Cuda();self.buffers={};self.events=[];self.stream=C.c_void_p()
        self.config=json.loads(Path(manifest).read_text());self.path=Path(manifest).parent/self.config['engine']
        if sha256(self.path)!=self.config['engine_sha256']:raise ValueError('engine checksum mismatch')
        self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,'')
        self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:raise RuntimeError('engine deserialization failed')
        self.context=self.engine.create_execution_context()
        self.inputs=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                     if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.INPUT]
        self.outputs=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                      if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
        if set(self.inputs)!={'images','image_size'} or self.outputs!=['detections']:
            raise ValueError('unexpected graph bindings')
        self.cuda.call('cudaStreamCreate',C.byref(self.stream))
        for _ in range(2):
            e=C.c_void_p();self.cuda.call('cudaEventCreate',C.byref(e));self.events.append(e)

    def buffer(self,name,size):
        if name not in self.buffers or self.buffers[name][1]<size:
            if name in self.buffers:self.cuda.call('cudaFree',self.buffers[name][0])
            p=C.c_void_p();self.cuda.call('cudaMalloc',C.byref(p),size);self.buffers[name]=(p,size)
        return self.buffers[name][0]

    def run(self,images,image_size):
        arrays={name:np.ascontiguousarray(value,dtype=self.trt.nptype(self.engine.get_tensor_dtype(name)))
                for name,value in [('images',images),('image_size',image_size)]}
        for name,value in arrays.items():
            if not self.context.set_input_shape(name,value.shape):raise ValueError('unsupported shape '+str(value.shape))
        shape=tuple(self.context.get_tensor_shape('detections'))
        if shape!=(300,6):raise ValueError('expected fixed padded detections, got '+str(shape))
        output=np.empty(shape,dtype=self.trt.nptype(self.engine.get_tensor_dtype('detections')))
        for name,value in arrays.items():
            ptr=self.buffer(name,value.nbytes);self.context.set_tensor_address(name,ptr.value)
            self.cuda.call('cudaMemcpyAsync',ptr,C.c_void_p(value.ctypes.data),value.nbytes,1,self.stream)
        ptr=self.buffer('detections',output.nbytes);self.context.set_tensor_address('detections',ptr.value)
        self.cuda.call('cudaEventRecord',self.events[0],self.stream)
        if not self.context.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT execute failed')
        self.cuda.call('cudaEventRecord',self.events[1],self.stream)
        self.cuda.call('cudaMemcpyAsync',C.c_void_p(output.ctypes.data),ptr,output.nbytes,2,self.stream)
        self.cuda.call('cudaStreamSynchronize',self.stream)
        duration=C.c_float();self.cuda.call('cudaEventElapsedTime',C.byref(duration),*self.events)
        self.gpu_ms=duration.value
        return output

    def close(self):
        if self.stream.value:
            self.cuda.call('cudaStreamSynchronize',self.stream)
            for p,_ in self.buffers.values():self.cuda.call('cudaFree',p)
            for e in self.events:self.cuda.call('cudaEventDestroy',e)
            self.cuda.call('cudaStreamDestroy',self.stream)
            self.stream=C.c_void_p();self.buffers={};self.events=[]

    def __del__(self):
        try:self.close()
        except Exception:pass


class Detector:
    def __init__(self,weights,device='tensorrt'):
        self.session=Session(weights)
        self.transform=GeneralizedRCNNTransform(576,576,[.406,.456,.485,.5],[.225,.224,.229,.25]).eval()
        self.metadata=dict(backend='TensorRT full Torchvision detection graph',
                           adapter_source_sha256=sha256(__file__),tensorrt=self.session.trt.__version__,
                           family=self.session.config['family'],execution_device='Jetson GPU',
                           precision=self.session.config['precision'],input_order='B,G,R,IR',
                           checkpoint_sha256=self.session.config['checkpoint_sha256'],
                           engine_sha256=self.session.config['engine_sha256'],synchronous_return=True,
                           cpu_operations='Torchvision input transform and native coordinate restoration')
        self.last_stage_ms={}

    @torch.inference_mode()
    def infer(self,frame,resolution):
        start=time.monotonic()
        self.transform.min_size=(resolution,);self.transform.max_size=resolution
        x=torch.from_numpy(np.ascontiguousarray(frame)).permute(2,0,1).float()/255
        batch,_=self.transform([x],None)
        before_gpu=time.monotonic()
        output=self.session.run(batch.tensors.numpy(),np.array(batch.image_sizes[0],np.int64))
        after_gpu=time.monotonic()
        output=output[output[:,4]>.001].copy()
        mapped=self.transform.postprocess([{'boxes':torch.from_numpy(output[:,:4])}],
                                         batch.image_sizes,[tuple(frame.shape[:2])])
        output[:,:4]=mapped[0]['boxes'].numpy()
        self.last_stage_ms=dict(preprocess_ms=(before_gpu-start)*1000,
            gpu_execute_ms=self.session.gpu_ms,shape_transfer_execute_ms=(after_gpu-before_gpu)*1000,
            postprocess_ms=(time.monotonic()-after_gpu)*1000)
        return output

    def close(self):self.session.close()


def create(weights,device):return Detector(weights,device)
