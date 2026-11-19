"""Measure fresh-process TensorRT startup and first/warm inference separately.

Driver and filesystem caches are not purged. No claim of hardware-cold boot.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

CHILD='''
import json,time,sys
from pathlib import Path
import cv2,torch
from experiments.tensorrt_backend import TensorRTDetector
from live_dashboard import fuse_frames
engine,manifest,output=map(Path,sys.argv[1:])
torch.set_num_threads(4)
frame=json.loads(manifest.read_text())['frames'][0]
fused,_=fuse_frames(*[cv2.imread(str(manifest.parent/frame[s])) for s in ['rgb','thermal']])
t=time.perf_counter();model=TensorRTDetector(engine);initialized=time.perf_counter()
model.infer(fused,576);first=time.perf_counter();warm=[]
for _ in range(20):
 started=time.perf_counter();model.infer(fused,576);warm.append((time.perf_counter()-started)*1000)
output.write_text(json.dumps({'initialization_ms':(initialized-t)*1000,'first_infer_ms':(first-initialized)*1000,'warm_ms':warm,'metadata':model.metadata},indent=2))
model.close()
'''


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--engine',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reps',type=int,default=5);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in range(a.reps):
        result=a.output/f'process_{i}.json';start=time.perf_counter()
        with (a.output/f'process_{i}.log').open('w') as log:
            subprocess.run([sys.executable,'-c',CHILD,str(a.engine),str(a.manifest),str(result)],stdout=log,stderr=subprocess.STDOUT,check=True)
        row=json.loads(result.read_text());row['whole_subprocess_ms']=(time.perf_counter()-start)*1000;rows.append(row)
    output={'definition':'Fresh process/context/engine; OS and driver caches retained. Includes no engine building.',
            'processes':rows}
    (a.output/'summary.json').write_text(json.dumps(output,indent=2));print(a.output/'summary.json')


if __name__=='__main__':main()
