"""Measure shared dynamic-engine build and runtime warmup energy separately.

Fresh builder process; OS/driver caches retained. Canonical engine is not replaced.
INA rail sums are not wall-plug energy, and short warmup integration is coarse.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import cv2
import torch
from experiments.power import RailSampler
from experiments.run import energy_summary
from experiments.tensorrt_backend import TensorRTDetector
from live_dashboard import fuse_frames


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);config=json.loads(a.engine.read_text());torch.set_num_threads(4)
    frame=json.loads(a.manifest.read_text())['frames'][0]
    fused,_=fuse_frames(*[cv2.imread(str(a.manifest.parent/frame[s])) for s in ['rgb','thermal']])
    sampler=RailSampler();sampler.start()
    try:
        idle_start=time.monotonic();time.sleep(2);idle_end=time.monotonic();sampler.sample()
        idle=sampler.energy(idle_start,idle_end);idle_w=sum(idle.values())/(idle_end-idle_start) if idle and all(v is not None for v in idle.values()) else None
        command=list(config['build_command'])
        command=[('--saveEngine='+str(a.output/'measurement_only.engine')) if arg.startswith('--saveEngine=') else arg for arg in command]
        build_start=time.monotonic()
        with (a.output/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        build_end=time.monotonic();sampler.sample()
        startup_start=time.monotonic();model=TensorRTDetector(a.engine);loaded=time.monotonic();warm=[]
        try:
            for res in [480,544,576,640]:
                for _ in range(3):
                    t=time.monotonic();model.infer(fused,res);warm.append({'resolution':res,'ms':(time.monotonic()-t)*1000})
            startup_end=time.monotonic();sampler.sample()
        finally:model.close()
    finally:sampler.stop()
    record={'engine_build':energy_summary(sampler,build_start,build_end,idle_w),
            'runtime_initialization_and_12_warmup_calls':energy_summary(sampler,startup_start,startup_end,idle_w),
            'runtime_initialization_ms':(loaded-startup_start)*1000,'warmup_calls':warm,'build_command':command,
            'power_samples':sampler.samples,'power_errors':sampler.errors,'idle_rail_sum_watts':idle_w,
            'scope':'Common shared dynamic engine; repeat builder process with OS/driver caches retained. Not training energy, wall-plug energy, or exact original-build consumption.',
            'limitations':['Warmup energy covers a short interval at 10Hz and is coarse.','Steady per-frame results exclude these setup costs; amortize explicitly.']}
    (a.output/'summary.json').write_text(json.dumps(record,indent=2));print(a.output/'summary.json')


if __name__=='__main__':main()
