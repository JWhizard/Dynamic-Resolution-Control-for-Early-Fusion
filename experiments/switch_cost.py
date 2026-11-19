"""Randomized paired same-frame stay/switch detector-cost experiment.

For each directed resolution pair, prime the source then time the destination;
compare with prime destination then time destination on the identical frame.
Randomize which member of the pair runs first. Measures warm inference, not
cold engine construction or controller time. Shared-host DVFS remains a caveat.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import time
import cv2
import torch
from experiments.policies import ACTIONS
from experiments.run import ProjectDetector, hardware, sha, stats
from live_dashboard import fuse_frames, DEFAULT_WEIGHTS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--adapter',default='project')
    a=p.parse_args()
    if a.output.exists() or a.reps<1:p.error('need a fresh output path and positive repetitions')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manifest=json.loads(a.manifest.read_text());frames=manifest['frames'];rng=random.Random(a.seed)
    if a.adapter=='project':model=ProjectDetector(a.weights,a.device)
    else:
        module,factory=a.adapter.split(':');model=getattr(importlib.import_module(module),factory)(a.weights,a.device)
    sync=(lambda:torch.cuda.synchronize(a.device)) if a.device.startswith('cuda') else lambda:None
    def frame(row):
        return fuse_frames(*[cv2.imread(str(a.manifest.parent/row[s]),cv2.IMREAD_COLOR) for s in ['rgb','thermal']])[0]
    def timed(f,res):
        sync();t=time.perf_counter();model.infer(f,res);sync();return (time.perf_counter()-t)*1000
    trials=[(s,t,k) for s in ACTIONS for t in ACTIONS if s!=t for k in range(a.reps)]
    rng.shuffle(trials);rows=[]
    with torch.inference_mode():
        for res in ACTIONS:
            for _ in range(3):timed(frame(frames[0]),res)
        for source,target,rep in trials:
            index=rng.randrange(len(frames));f=frame(frames[index]);order=['switch','stay'];rng.shuffle(order);values={}
            for condition in order:
                timed(f,source if condition=='switch' else target)
                values[condition]=timed(f,target)
            rows.append({'source':source,'target':target,'rep':rep,'frame_index':index,'order':order,
                         'switch_ms':values['switch'],'stay_ms':values['stay'],
                         'paired_delta_ms':values['switch']-values['stay']})
    summary={f'{s}->{t}':stats([r['paired_delta_ms'] for r in rows if r['source']==s and r['target']==t]) for s in ACTIONS for t in ACTIONS if s!=t}
    result={'hardware':hardware(),'detector_metadata':getattr(model,'metadata',{}),'weights_sha256':sha(a.weights),'manifest_sha256':sha(a.manifest),
            'repetitions_per_directed_pair':a.reps,'summary_paired_delta_ms':summary,'rows':rows,
            'limitations':['Warm detector-only costs; no cold engine build or tracking state.',
                           'Short pilot repetitions do not establish stable tail effects.',
                           'Negative paired deltas are preserved; host noise and thermal drift remain possible.']}
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2));print(a.output)


if __name__=='__main__':main()
