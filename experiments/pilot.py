"""Run a bounded pilot; pause UI inference temporarily and always restore it."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import urllib.request


def control(payload=None):
    url='http://127.0.0.1:5000/api/' + ('control' if payload is not None else 'status')
    req=urllib.request.Request(url)
    if payload is not None:
        req.data=json.dumps(payload).encode();req.add_header('Content-Type','application/json')
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reps',type=int,default=2)
    p.add_argument('--frames',type=int,default=120)
    a=p.parse_args()
    if a.reps<1 or a.frames<6:p.error('need at least one rep and six frames')
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for rep in range(a.reps):
        for policy in ['fixed:480','fixed:544','fixed:576','fixed:640','sequence','deadline','aimd','random','switch']:
            jobs.append((f'{policy.replace(":","-")}_rep{rep}_threads',policy,rep,[]))
    for corruption in ['rgb_dark','rgb_blur','ir_dropout','ir_shift']:
        for policy in ['sequence','fixed:576','deadline']:
            jobs.append((f'{policy.replace(":","-")}_{corruption}',policy,42,['--corruption',corruption,'--corruption-middle-only','--thread-trace','4,4,4']))
    for policy in ['sequence','fixed:576','deadline']:
        jobs.append((f'{policy.replace(":","-")}_clean_constant',policy,42,['--thread-trace','4,4,4']))
    for policy in ['fixed:576','sequence','deadline','aimd']:
        jobs.append((f'{policy.replace(":","-")}_contention',policy,42,['--thread-trace','4,4,4','--contenders','2']))
    for policy in ['fixed:576','sequence','deadline']:
        jobs.append((f'{policy.replace(":","-")}_realtime',policy,42,['--mode','realtime']))
    random.Random(42).shuffle(jobs)
    (a.output/'order.json').write_text(json.dumps(jobs,indent=2))
    original=control()
    control({'running':False})
    try:
        for i,(name,policy,seed,extras) in enumerate(jobs):
            dest=a.output/(name+'.json')
            if dest.exists():
                print('Existing result:',dest,flush=True);continue
            cmd=[sys.executable,'-m','experiments.run','--manifest',str(a.manifest),'--output',str(dest),
                 '--policy',policy,'--seed',str(seed),'--max-frames',str(a.frames),*extras]
            print(f'[{i+1}/{len(jobs)}] {name}',flush=True)
            with (a.output/(name+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'})
    finally:
        control({'running':original['running'],'view':original['view'],
                 'confidence':original['confidence'],'policy':original['policy']})


if __name__=='__main__':main()
