"""Exact development-only fixed/adaptive evaluation of the target-trained YOLO.

The sealed holdout and previously viewed official test are never evaluated here.
These are offline serial costs, not camera streaming/dropped-frame measurements.
"""
import argparse
import contextlib
from collections import Counter
import json
import io
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from experiments.evaluate_families import score,sha256
from experiments.matched_platform_cost import GPUCounter
from experiments.policies import Policy,ACTIONS
from fusion.preprocess import load_model,detect


def feedback(detections):
    """Map target bus/car/pedestrian to the frozen M3FD controller's labels."""
    mapping={4:0,1:1,3:4};rows=[]
    for row in detections:
        if int(row[5]) in mapping:
            r=list(row);r[5]=mapping[int(row[5])];rows.append(r)
    return np.asarray(rows,dtype=float).reshape(-1,6)


def person_recall(gt_dict,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    selected=[p for p in predictions if p['category_id']==3 and p['score']>=.25]
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=gt_dict;gt.createIndex()
        if selected:dt=gt.loadRes(selected)
        else:
            dt=COCO();dt.dataset=dict(images=gt_dict['images'],categories=gt_dict['categories'],annotations=[]);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[3];ev.params.iouThrs=np.array([.5])
        ev.evaluate();ev.accumulate()
    return dict(person_recall50_conf25=float(ev.eval['recall'][0,0,0,-1]),
        small_person_recall50_conf25=float(ev.eval['recall'][0,0,1,-1]))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--controller',type=Path,required=True);p.add_argument('--nvml-index',type=int,default=3);a=p.parse_args()
    plan=json.loads((a.study/'plan.json').read_text());best=json.loads((a.study/'BEST_MODEL.json').read_text())
    if plan['holdout_unlocked'] or sha256(best['checkpoint'])!=best['sha256']:raise ValueError('Frozen development checkpoint/protocol mismatch')
    output=a.study/'development';output.mkdir(exist_ok=False)
    manifest=json.loads((a.study/'val_manifest.json').read_text());frames=manifest['frames']
    gt=json.loads((a.study/'val_annotations.json').read_text());ids={f['image_id'] for f in frames}
    if ids!={x['id'] for x in gt['images']}:raise ValueError('Development GT/frame mismatch')
    holdout_ids={f['image_id'] for f in json.loads((a.study/'holdout_manifest.json').read_text())['frames']}
    if ids&holdout_ids:raise ValueError('Holdout leakage')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=load_model(best['checkpoint'],'cuda:0');counter=GPUCounter(a.nvml_index)
    def pixels(frame):
        rgb=cv2.imread(frame['rgb']);ir=cv2.imread(frame['thermal'],0)
        if rgb is None or ir is None or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid development pair')
        return np.ascontiguousarray(np.dstack([rgb,ir]))
    profile_frame=json.loads((a.study/'train_manifest.json').read_text())['frames'][0]
    first=pixels(profile_frame);profile={}
    for resolution in ACTIONS:
        durations=[]
        for _ in range(5):
            before=time.monotonic();detect(model,torch.from_numpy(first).to('cuda:0'),resolution,conf=.001);torch.cuda.synchronize()
            durations.append((time.monotonic()-before)*1000)
        profile[resolution]=float(np.median(durations[-3:]))
    policies=['fixed:'+str(r) for r in plan['development_fixed_resolutions']]+plan['development_policies']
    random.Random(plan['seed']).shuffle(policies)
    evaluation_plan=dict(checkpoint=best,policies=policies,profile_ms=profile,
        controller=str(a.controller),controller_sha256=sha256(a.controller),development_pairs=len(frames),
        val_manifest_sha256=sha256(a.study/'val_manifest.json'),annotation_sha256=sha256(a.study/'val_annotations.json'),
        feedback_mapping='Tiny bus4->M3FD bus0; car1->car1; pedestrian3->People4; other target classes excluded from source-controller feedback only',
        scoring='All seven target classes, native unmodified COCO geometry, confidence.001, maxDets1/10/100',
        holdout_evaluated=False,mode='offline serial; no dropped-frame/causal streaming score',source_sha256=sha256(__file__))
    evaluation_plan['cpu_affinity']=sorted(os.sched_getaffinity(0))
    (output/'plan.json').write_text(json.dumps(evaluation_plan,indent=2)+'\n')
    results=[]
    for name in policies:
        policy=Policy(name,budget_ms=plan['development_budget_ms'],seed=plan['seed'],controller=a.controller)
        if policy.sequence:policy.sequence.groups={str(f['image_id']):f['sequence'] for f in frames}
        if policy.guarded:policy.cost.update(profile)
        rows=[];predictions=[];e0=counter.energy();start=time.monotonic()
        for index,frame in enumerate(frames,1):
            before=time.monotonic();image=pixels(frame);identifier=str(frame['image_id']);resolution=policy.choose(identifier)
            det=detect(model,torch.from_numpy(image).to('cuda:0'),resolution,conf=.001).cpu().numpy();torch.cuda.synchronize()
            inferred=time.monotonic();policy.update(identifier,feedback(det),resolution,(inferred-before)*1000);end=time.monotonic()
            rows.append(dict(image_id=frame['image_id'],sequence=frame['sequence'],resolution=resolution,
                service_ms=(end-before)*1000,controller_update_ms=(end-inferred)*1000))
            for x1,y1,x2,y2,confidence,cls in det:
                predictions.append(dict(image_id=frame['image_id'],category_id=int(cls),bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(confidence)))
            if index%500==0:print(name,index,'/',len(frames),flush=True)
        joules=counter.energy()-e0;seconds=time.monotonic()-start
        metrics=score(gt,predictions);metrics.update(person_recall(gt,predictions));costs=[r['service_ms'] for r in rows]
        result=dict(policy=name,metrics=metrics,mean_service_ms=float(np.mean(costs)),p95_service_ms=float(np.percentile(costs,95)),
            p99_service_ms=float(np.percentile(costs,99)),J_per_frame=joules/len(frames),energy_scope='NVML GPU-device energy',
            seconds=seconds,frames=len(frames),resolution_mix=dict(Counter(r['resolution'] for r in rows)))
        tag=name.replace(':','-');(output/(tag+'_predictions.json')).write_text(json.dumps(predictions))
        (output/(tag+'_costs.json')).write_text(json.dumps(rows))
        results.append(result);(output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        (output/'progress.json').write_text(json.dumps(dict(completed=len(results),total=len(policies),latest=name)))
        print(json.dumps(result),flush=True)
    fixed=[r for r in results if r['policy'].startswith('fixed:')]
    feasible=[r for r in fixed if r['p95_service_ms']<=plan['development_budget_ms']]
    chosen=max(feasible,key=lambda r:(r['metrics']['AP'],-r['mean_service_ms'])) if feasible else min(fixed,key=lambda r:r['mean_service_ms'])
    (output/'SELECTED_FIXED.json').write_text(json.dumps(dict(policy=chosen['policy'],
        rule='Highest development COCO AP among p95<=25ms; ties lower mean cost; if infeasible choose fastest mean',
        development_result=chosen,holdout_evaluated=False),indent=2)+'\n')
    (output/'COMPLETED.json').write_text(json.dumps(dict(cases=len(results),development_pairs=len(frames),holdout_evaluated=False),indent=2))
    lines=['# RGBT-Tiny development baselines','',
        'One target-fine-tuned seven-class YOLO26n, selected on development validation, is compared at five fixed input sizes and with four existing local adaptive controls. All results below use the development split. The fresh holdout remains unscored.','',
        '| Policy | Seven-class COCO AP % | AP-small % | Pedestrian AP % | Person recall % | Small-person recall % | Mean service ms | p95 service ms | GPU J/frame |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(results,key=lambda r:r['policy']):
        lines.append(f"| {r['policy']} | {100*r['metrics']['AP']:.3f} | {100*r['metrics']['AP_small']:.3f} | {100*r['metrics']['AP_per_class']['pedestrian']:.3f} | {100*r['metrics']['person_recall50_conf25']:.2f} | {100*r['metrics']['small_person_recall50_conf25']:.2f} | {r['mean_service_ms']:.2f} | {r['p95_service_ms']:.2f} | {r['J_per_frame']:.3f} |")
    lines+=['',f"Validation-selected fixed reference: **{chosen['policy']}**. Rule: maximize validation COCO AP among fixed settings with p95 service ≤25 ms; ties use lower mean service. If none meet the budget, select the fastest mean service and report infeasibility.",'',
        'Costs include serial input reading/decoding, transfer, inference and policy update. GPU energy is the NVML device counter delta; loading/profiling/warmup are outside the loop. Offline processing does not establish real-time dropped-frame or output-age performance. One run per policy provides no repeat-run uncertainty.','',
        'Person recall uses IoU.5, confidence.25 and maxDets100; small-person recall uses native area below32².','',
        'The source-trained sequence policy receives an explicit bus/car/person class mapping; other target classes are excluded from its feedback, while accuracy scores all seven target classes. The existing local controls are not faithful published-baseline reproductions. A new reliability-aware controller has not been fitted here.','',
        'Training selection uses the pinned Ultralytics validation mAP50–95 fitness; this table recomputes exact native-coordinate pycocotools metrics. Original COCO geometry is retained for this scoring; training/Ultralytics validation labels clip boxes to the image. Training augmentation/validation transforms and native inference can differ slightly.','',
        'Next: diagnose per-sequence and small-object failures, fit the proposed controller using train/development data only, freeze that protocol, and then unlock the fresh holdout. Do not tune on the holdout or treat the previously viewed official test as untouched.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
