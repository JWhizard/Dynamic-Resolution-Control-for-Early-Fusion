"""Owned, bounded contention workers; never changes host power/clock settings."""
import argparse
import os
from pathlib import Path
import select
import subprocess
import sys


def stop_workers(workers):
    for child in workers:
        if child.poll() is None:child.terminate()
    for child in workers:
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


def start_workers(kind,count,cores,engine=None,memory_mb=256):
    workers=[]
    try:
        for i in range(count):
            cmd=[sys.executable,'-m','experiments.resources','--kind',kind,'--core',str(cores[i%len(cores)]),'--memory-mb',str(memory_mb)]
            if engine:cmd+=['--engine',str(engine)]
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                                   env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            workers.append(child)
            ready,_,_=select.select([child.stdout],[],[],30)
            if not ready or child.stdout.readline().strip()!='ready':
                raise RuntimeError(f'{kind} contention worker failed to become ready')
        return workers
    except Exception:
        stop_workers(workers);raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--kind',choices=['cpu','gpu','memory'],required=True)
    p.add_argument('--core',type=int,required=True);p.add_argument('--engine',type=Path)
    p.add_argument('--memory-mb',type=int,default=256);a=p.parse_args();os.sched_setaffinity(0,{a.core})
    if a.kind=='cpu':
        print('ready',flush=True)
        while True:pass
    elif a.kind=='memory':
        if not 1<=a.memory_mb<=1024:raise ValueError('memory worker size must be 1–1024 MiB')
        import numpy as np
        memory=np.ones(a.memory_mb*1024*1024//8,dtype=np.float64)
        print('ready',flush=True)
        while True:np.add(memory,1,out=memory)
    else:
        if a.engine is None:raise ValueError('GPU contention needs a compatible engine manifest')
        import numpy as np
        from experiments.tensorrt_backend import TensorRTDetector
        model=TensorRTDetector(a.engine)
        x=np.zeros((1,4,640,640),np.float32);model.run_tensor(x)
        print('ready',flush=True)
        while True:model.run_tensor(x)


if __name__=='__main__':main()
