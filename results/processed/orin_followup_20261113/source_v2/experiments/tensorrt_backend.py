"""TensorRT 10 GPU detector with the project's exact CPU preprocessing.

The system CUDA runtime is accessed through typed ctypes calls, allowing this
backend to coexist with CPU-only PyTorch. TensorRT bindings must match JetPack.
All GPU work is synchronized before infer() returns. No hidden CUDA PyTorch use.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from fusion.preprocess import to_input,postprocess


class Cuda:
    def __init__(self):
        self.lib=C.CDLL('/usr/local/cuda/lib64/libcudart.so.12')
        signatures={
            'cudaMalloc':([C.POINTER(C.c_void_p),C.c_size_t],C.c_int),
            'cudaFree':([C.c_void_p],C.c_int),
            'cudaStreamCreate':([C.POINTER(C.c_void_p)],C.c_int),
            'cudaStreamSynchronize':([C.c_void_p],C.c_int),
            'cudaStreamDestroy':([C.c_void_p],C.c_int),
            'cudaMemcpyAsync':([C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p],C.c_int),
            'cudaEventCreate':([C.POINTER(C.c_void_p)],C.c_int),
            'cudaEventRecord':([C.c_void_p,C.c_void_p],C.c_int),
            'cudaEventElapsedTime':([C.POINTER(C.c_float),C.c_void_p,C.c_void_p],C.c_int),
            'cudaEventDestroy':([C.c_void_p],C.c_int),
            'cudaGetErrorString':([C.c_int],C.c_char_p),
        }
        for name,(args,restype) in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=restype

    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code:raise RuntimeError(f'{name}: {self.lib.cudaGetErrorString(code).decode()}')


class TensorRTDetector:
    def __init__(self,weights,device='tensorrt'):
        import tensorrt as trt
        self.trt=trt;self.cuda=Cuda();self.stream=C.c_void_p();self.events=[];self.buffers={}
        self.cuda.call('cudaStreamCreate',C.byref(self.stream))
        for _ in range(2):
            event=C.c_void_p();self.cuda.call('cudaEventCreate',C.byref(event));self.events.append(event)
        config_path=Path(weights);self.config=json.loads(config_path.read_text())
        engine_path=config_path.parent/self.config['engine']
        data=engine_path.read_bytes();actual=hashlib.sha256(data).hexdigest()
        if actual!=self.config['engine_sha256']:raise ValueError('engine checksum differs from manifest')
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:raise RuntimeError('TensorRT engine deserialization failed')
        self.context=self.engine.create_execution_context()
        self.input_names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.INPUT]
        self.output_names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
        if len(self.input_names)!=1 or len(self.output_names)!=1:raise ValueError('expected one input and one end-to-end detection output')
        self.metadata={'backend':'TensorRT','version':trt.__version__,'execution_device':'Jetson GPU',
                       'precision':self.config['precision'],'engine_sha256':actual,
                       'checkpoint_sha256':self.config['checkpoint_sha256'],
                       'preprocessing':'same CPU torch antialiased bilinear and letterbox as project',
                       'input_order':'B,G,R,IR','output':'native xyxy/conf/class','synchronous_return':True}
        self.last_stage_ms={}

    def _buffer(self,name,size):
        existing=self.buffers.get(name)
        if existing is None or existing[1]<size:
            if existing:self.cuda.call('cudaFree',existing[0])
            pointer=C.c_void_p();self.cuda.call('cudaMalloc',C.byref(pointer),size)
            self.buffers[name]=(pointer,size)
        return self.buffers[name][0]

    def run_tensor(self,x):
        trt=self.trt
        x=np.ascontiguousarray(x,dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_names[0])))
        if not self.context.set_input_shape(self.input_names[0],x.shape):raise ValueError(f'unsupported input shape {x.shape}')
        output_shape=tuple(self.context.get_tensor_shape(self.output_names[0]))
        if any(v<0 for v in output_shape):raise ValueError(f'unresolved output shape {output_shape}')
        output=np.empty(output_shape,dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_names[0])))
        inp=self._buffer(self.input_names[0],x.nbytes);out=self._buffer(self.output_names[0],output.nbytes)
        self.context.set_tensor_address(self.input_names[0],inp.value)
        self.context.set_tensor_address(self.output_names[0],out.value)
        transfer_start=time.perf_counter()
        self.cuda.call('cudaMemcpyAsync',inp,C.c_void_p(x.ctypes.data),x.nbytes,1,self.stream)
        self.cuda.call('cudaEventRecord',self.events[0],self.stream)
        if not self.context.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT execution failed')
        self.cuda.call('cudaEventRecord',self.events[1],self.stream)
        self.cuda.call('cudaMemcpyAsync',C.c_void_p(output.ctypes.data),out,output.nbytes,2,self.stream)
        self.cuda.call('cudaStreamSynchronize',self.stream)
        gpu=C.c_float();self.cuda.call('cudaEventElapsedTime',C.byref(gpu),*self.events)
        self.last_stage_ms={'gpu_execute_ms':gpu.value,'transfer_execute_host_ms':(time.perf_counter()-transfer_start)*1000}
        if output.ndim!=3 or output.shape[-1]!=6:raise ValueError(f'unexpected model output {output.shape}')
        return output

    def infer(self,frame,resolution):
        started=time.perf_counter()
        x,lb=to_input(torch.from_numpy(frame),resolution)
        pre_done=time.perf_counter()
        output=self.run_tensor(x.numpy())
        infer_done=time.perf_counter()
        detections=postprocess(torch.from_numpy(output),lb,*frame.shape[:2],conf=.001).numpy()
        self.last_stage_ms.update({'preprocess_ms':(pre_done-started)*1000,
                                   'inference_with_shape_and_transfer_ms':(infer_done-pre_done)*1000,
                                   'postprocess_ms':(time.perf_counter()-infer_done)*1000})
        return detections

    def close(self):
        if getattr(self,'stream',None) and self.stream.value:
            self.cuda.call('cudaStreamSynchronize',self.stream)
            for pointer,_ in self.buffers.values():self.cuda.call('cudaFree',pointer)
            self.buffers.clear()
            for event in self.events:self.cuda.call('cudaEventDestroy',event)
            self.events=[]
            self.cuda.call('cudaStreamDestroy',self.stream);self.stream=C.c_void_p()

    def __del__(self):
        try:self.close()
        except Exception:pass


def create(weights,device):return TensorRTDetector(weights,device)
