"""Fixed576 deployment cost on identical predecoded pairs; fresh process per run.

TensorRT/Orin and PyTorch/workstation are different deployment stacks. Report
latencies as deployments, and never ratio whole-host rails against GPU energy.
"""
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from experiments.evaluate_families import sha256
from experiments.power import RailSampler


class GPUCounter:
    def __init__(self,index):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');self.handle=ctypes.c_void_p()
        if self.lib.nvmlInit_v2()!=0 or self.lib.nvmlDeviceGetHandleByIndex_v2(index,ctypes.byref(self.handle))!=0:raise RuntimeError('NVML unavailable')
    def energy(self):
        value=ctypes.c_ulonglong()
        if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle,ctypes.byref(value))!=0:raise RuntimeError('NVML energy counter unsupported')
        return value.value/1000


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['plan','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--backend',choices=['tensorrt','pytorch'],required=True)
    p.add_argument('--nvml-index',type=int,default=3)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous output')
    plan=json.loads(a.plan.read_text());entry=next(m for m in plan['models'] if m['name']==a.model)
    manifest=json.loads(a.manifest.read_text())
    frames=manifest['frames'][:120]
    cores=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,cores)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    decoded=[];hashes=[];before=time.monotonic()
    for frame in frames:
        paths=[a.manifest.parent/frame[k] for k in ['rgb','thermal']]
        rgb=cv2.imread(str(paths[0]));ir=cv2.imread(str(paths[1]),0)
        if rgb is None or ir is None or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid pair')
        actual=[sha256(path) for path in paths]
        if actual!=[frame['rgb_sha256'],frame['thermal_sha256']]:raise ValueError('Input hash mismatch')
        hashes.append(dict(id=frame['id'],rgb=actual[0],thermal=actual[1]))
        decoded.append(np.ascontiguousarray(np.dstack([rgb,ir])))
    prefetch_seconds=time.monotonic()-before
    if a.backend=='tensorrt':
        config=json.loads(Path(entry['engine']).read_text());gate=json.loads(Path(entry['validation']).read_text())
        if not gate['passed'] or config['engine_sha256']!=gate.get('engine_sha256',gate.get('engine',{}).get('engine_sha256')):raise ValueError('Engine gate mismatch')
        module,factory=entry['adapter'].split(':');detector=getattr(importlib.import_module(module),factory)(entry['engine'],'tensorrt')
        infer=lambda frame:detector.infer(frame,576)
        counter=None
    else:
        if sha256(entry['checkpoint'])!=entry['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        from experiments.detector_families import Detector
        from fusion.preprocess import load_model,detect
        if entry['family']=='yolo':
            detector=load_model(entry['checkpoint'],'cuda:0')
            infer=lambda frame:detect(detector,torch.from_numpy(frame).to('cuda:0'),576,conf=.001).cpu().numpy()
        else:
            detector=Detector(entry['checkpoint'],'cuda:0');infer=lambda frame:detector.infer(frame,576)
        counter=GPUCounter(a.nvml_index)
    for _ in range(12):infer(decoded[0])
    if a.backend=='pytorch':torch.cuda.synchronize()
    sampler=RailSampler();sampler.start();time.sleep(1)
    rows=[]
    try:
        e0=counter.energy() if counter else None;start=time.monotonic()
        for frame,pixels in zip(frames,decoded):
            before=time.monotonic();detections=infer(pixels)
            if a.backend=='pytorch':torch.cuda.synchronize()
            end=time.monotonic()
            if not np.isfinite(detections).all():raise ValueError('Invalid predictions')
            rows.append(dict(id=frame['id'],service_ms=(end-before)*1000,detections=np.asarray(detections).tolist()))
        end=time.monotonic();e1=counter.energy() if counter else None
    finally:sampler.stop()
    rails=sampler.energy(start,end)
    joules=e1-e0 if counter else sum(rails.values()) if rails and all(x is not None for x in rails.values()) else None
    if joules is None or sampler.errors:raise ValueError('Missing/invalid energy trace')
    values=[r['service_ms'] for r in rows]
    record=dict(model=a.model,backend=a.backend,platform=platform.platform(),host=platform.node(),
        batch_size=1,resolution=576,cpu_threads=4,cpu_affinity=cores,frames=len(rows),
        mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),
        joules=joules,J_per_frame=joules/len(rows),energy_scope='GPU device NVML total-energy delta' if counter else 'Orin named-rail host sum; shared services included',
        input_mode='predecoded host arrays; H2D/transforms/inference/postprocessing included; decoding/prefetch excluded',
        prefetch_seconds=prefetch_seconds,prefetch_bytes=sum(x.nbytes for x in decoded),
        source_sha256=sha256(__file__),plan_sha256=sha256(a.plan),manifest_sha256=sha256(a.manifest),input_hashes=hashes,
        torch=torch.__version__,cuda=torch.version.cuda,rows=rows,power_samples=sampler.samples,rail_joules=rails,power_errors=sampler.errors)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record))
    print(json.dumps({k:v for k,v in record.items() if k not in ['rows','power_samples','input_hashes']}),flush=True)


if __name__=='__main__':main()
