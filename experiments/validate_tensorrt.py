"""Gate TensorRT deployment using same-input, class-aware matched detections.

TopK rows can reorder; raw row-wise box subtraction is not an equivalence test.
This checks numerical agreement, not dataset detection accuracy.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from experiments.tensorrt_backend import TensorRTDetector
from fusion.preprocess import load_model,detect
from live_dashboard import fuse_frames


def match(reference,candidate,threshold=.1):
    a=reference[reference[:,4]>=threshold];b=candidate[candidate[:,4]>=threshold]
    used=set();matches=[]
    for row in sorted(a,key=lambda r:-r[4]):
        choices=[]
        for j,other in enumerate(b):
            if j in used or row[5]!=other[5]:continue
            inter=np.maximum(0,np.minimum(row[2:4],other[2:4])-np.maximum(row[:2],other[:2])).prod()
            area=np.maximum(0,row[2:4]-row[:2]).prod()+np.maximum(0,other[2:4]-other[:2]).prod()-inter
            choices.append((float(inter/max(area,1e-9)),j))
        if choices:
            iou,j=max(choices)
            if iou>=.9:
                used.add(j);matches.append({'iou':iou,'score_error':float(abs(row[4]-b[j,4]))})
    return {'reference_count':len(a),'candidate_count':len(b),'matched':len(matches),'matches':matches,
            'passed':len(matches)==len(a)==len(b) and all(m['score_error']<=.02 for m in matches)}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--confidence',type=float,default=.1);a=p.parse_args()
    if not .001<=a.confidence<=1:p.error('confidence must be between .001 and 1')
    config=json.loads(a.engine.read_text());torch.set_num_threads(4)
    reference=load_model(config['checkpoint'],'cpu');candidate=TensorRTDetector(a.engine)
    manifest=json.loads(a.manifest.read_text());frames=manifest['frames']
    indices=np.linspace(0,len(frames)-1,min(a.frames,len(frames)),dtype=int);rows=[]
    with torch.inference_mode():
        for i in indices:
            frame=frames[i]
            fused,_=fuse_frames(*[cv2.imread(str(a.manifest.parent/frame[s])) for s in ['rgb','thermal']])
            for res in [480,544,576,640]:
                cpu=detect(reference,torch.from_numpy(fused),res,conf=.001).numpy()
                gpu=candidate.infer(fused,res)
                rows.append({'frame_index':int(i),'resolution':res,**match(cpu,gpu,a.confidence)})
    result={'passed':all(r['passed'] for r in rows),'gate':{'confidence':a.confidence,'minimum_iou':.9,'maximum_score_error':.02,'equal_detection_counts':True},
            'engine':candidate.metadata,'comparisons':rows,'limitations':['Numerical agreement on sampled inputs, not AP equivalence on labeled test data.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    candidate.close();print(json.dumps({'passed':result['passed'],'comparisons':len(rows),'output':str(a.output)}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
