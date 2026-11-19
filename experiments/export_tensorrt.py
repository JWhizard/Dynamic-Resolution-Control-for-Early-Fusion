"""Export the exact project detector and build a dynamic TensorRT GPU engine.

Uses installed JetPack TensorRT; never installs/replaces system GPU libraries.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import onnx
import torch
from fusion.preprocess import load_model
from live_dashboard import DEFAULT_WEIGHTS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--fp16',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model=load_model(str(a.weights),'cpu')
    head=model.model[-1];head.export=True;head.dynamic=True;head.format='onnx'
    x=torch.zeros(1,4,352,576)
    path=a.output/'model.onnx'
    with torch.inference_mode():
        torch.onnx.export(model,x,str(path),input_names=['images'],output_names=['detections'],
                          dynamic_axes={'images':{2:'height',3:'width'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    engine=a.output/'model.engine'
    cmd=['/usr/src/tensorrt/bin/trtexec','--onnx='+str(path),'--saveEngine='+str(engine),
         '--minShapes=images:1x4x256x320','--optShapes=images:1x4x352x576',
         '--maxShapes=images:1x4x640x640','--memPoolSize=workspace:512',
         '--builderOptimizationLevel=2','--noTF32','--skipInference']
    if a.fp16:cmd.append('--fp16')
    started=time.monotonic()
    with (a.output/'build.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    record={'engine':'model.engine','onnx':'model.onnx','checkpoint':str(a.weights.resolve()),
            'checkpoint_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'precision':'fp16' if a.fp16 else 'fp32_no_tf32','input_channel_order':'B,G,R,IR',
            'class_names':['Bus','Car','Lamp','Motorcycle','People','Truck'],
            'build_seconds':time.monotonic()-started,'build_command':cmd,
            'created':datetime.now().isoformat()}
    (a.output/'engine.json').write_text(json.dumps(record,indent=2));print(a.output/'engine.json')


if __name__=='__main__':main()
