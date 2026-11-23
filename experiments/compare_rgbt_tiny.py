"""Frozen M3FD checkpoint transfer to the full official RGBT-Tiny test split.

Use shared bus/car/person classes and native visible-coordinate COCO labels.
Existing M3FD test predictions are rescored with the same class subset.
No target fitting or model selection; other target classes are out of scope.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from experiments.detector_families import Detector
from experiments.evaluate_families import score, sha256
from fusion.preprocess import detect, load_model

COMMON={1:'bus',2:'car',5:'person'}
TINY_NAMES=['ship','car','cyclist','pedestrian','bus','drone','plane']
TINY_MAP={4:1,1:2,3:5}


def common_gt(gt,mapping):
    annotations=[]
    for annotation in gt['annotations']:
        if annotation['category_id'] not in mapping:continue
        a=dict(annotation,category_id=mapping[annotation['category_id']])
        if a.get('ignore',0):a['iscrowd']=1
        annotations.append(a)
    return dict(info=gt.get('info',{}),images=gt['images'],annotations=annotations,
        categories=[dict(id=i,name=name) for i,name in COMMON.items()])


def common_predictions(predictions):
    return [p for p in predictions if p['category_id'] in COMMON]


def annotation_stats(gt):
    boxes=[a for a in gt['annotations'] if not a.get('iscrowd',0) and not a.get('ignore',0)]
    areas=np.array([a['bbox'][2]*a['bbox'][3] for a in boxes])
    density=Counter(a['image_id'] for a in boxes)
    return dict(images=len(gt['images']),boxes=len(boxes),
        class_boxes=dict(Counter(next(c['name'] for c in gt['categories'] if c['id']==a['category_id']) for a in boxes)),
        below_16_squared=int((areas<256).sum()),below_32_squared=int((areas<1024).sum()),
        fraction_below_16_squared=float((areas<256).mean()) if len(areas) else None,
        fraction_below_32_squared=float((areas<1024).mean()) if len(areas) else None,
        mean_boxes_per_image=len(boxes)/len(gt['images']),images_over_100_boxes=sum(n>100 for n in density.values()))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint-plan',type=Path,required=True)
    p.add_argument('--m3fd-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true',help='Rescore saved predictions after verifying the frozen plan')
    a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():p.error('CUDA required')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    previous_plan=json.loads((a.output/'plan.json').read_text()) if a.resume else None
    def progress(stage,**kwargs):
        value=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        temp=a.output/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(a.output/'status.json')
        print(json.dumps(value),flush=True)
    progress('preparing_official_test')
    raw=a.dataset/'raw'
    test_path=raw/'annotations_coco/instances_00_test2017.json'
    gt=json.loads(test_path.read_text())
    if [c['name'] for c in sorted(gt['categories'],key=lambda c:c['id'])]!=TINY_NAMES:
        raise ValueError('Unexpected target class mapping')
    train=(raw/'data_split/00_train.txt').read_text().splitlines()
    test=(raw/'data_split/00_test.txt').read_text().splitlines()
    train_sequences={s.split('/')[0] for s in train};test_sequences={s.split('/')[0] for s in test}
    if train_sequences&test_sequences or len(test)!=len(gt['images']):raise ValueError('Official split leakage/count mismatch')
    if {str(Path(i['file_name']).with_suffix('')) for i in gt['images']}!=set(test):raise ValueError('Split/image list mismatch')
    target=common_gt(gt,TINY_MAP)
    source_gt_path=a.m3fd_results/'ground_truth.json'
    source=common_gt(json.loads(source_gt_path.read_text()),{i:i for i in COMMON})
    checkpoints=json.loads(a.checkpoint_plan.read_text())
    for model in checkpoints['models']:
        if sha256(model['checkpoint'])!=model['sha256']:raise ValueError('Frozen checkpoint hash mismatch')
    images=sorted(gt['images'],key=lambda i:i['file_name'])
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_dataset='M3FD grouped test (419 pairs)',
        target_dataset='RGBT-Tiny complete official visible-coordinate test',test_pairs=len(images),
        test_sequences=sorted(test_sequences),train_sequences=sorted(train_sequences),
        target_annotation_sha256=sha256(test_path),source_annotation_sha256=sha256(source_gt_path),
        source_prediction_sha256={m['name']:sha256(a.m3fd_results/(m['name']+'_predictions.json')) for m in checkpoints['models']},
        models=checkpoints['models'],resolution=576,confidence=.001,device=a.device,seed=20261123,
        source_class_ids=COMMON,target_class_mapping=TINY_MAP,
        annotation_convention='Native unmodified 00 visible boxes; ignore regions become crowd. All images retained, including negatives. Other classes excluded.',
        scope='Frozen M3FD-to-RGBT-Tiny transfer; not training-budget-matched within-dataset training. COCO IoU AP, not official SAFit benchmark scores.',
        source_sha256=sha256(__file__),reproduction='python -m experiments.compare_rgbt_tiny with original CLI arguments',
        arguments={k:str(v) for k,v in vars(a).items()})
    if previous_plan:
        for key in ['target_annotation_sha256','source_annotation_sha256','source_prediction_sha256','models','resolution','confidence','test_sequences','target_class_mapping']:
            if previous_plan[key]!=json.loads(json.dumps(plan[key])):raise ValueError('Resume plan changed: '+key)
        (a.output/'RESUME.json').write_text(json.dumps(dict(utc=plan['created_utc'],
            original_source_sha256=previous_plan['source_sha256'],current_source_sha256=plan['source_sha256'],scorer_sha256=sha256(score.__code__.co_filename),
            reason='Shared-class evaluator compatibility; reuse native saved predictions, no target fitting.'),indent=2))
    else:(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'target_common_ground_truth.json').write_text(json.dumps(target))
    stats=dict(M3FD=annotation_stats(source),RGBT_Tiny=annotation_stats(target),
        target_all_classes=annotation_stats(gt),source_train_test_disjoint=True,
        target_paired_images=len(images),target_individual_modality_frames=2*len(images))
    (a.output/'dataset_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(plan['seed']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    (a.output/'environment.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(a.device)),indent=2))
    order=np.random.default_rng(plan['seed']).permutation(len(plan['models']))
    results=[];input_hashes=[]
    for position in order:
        entry=plan['models'][int(position)]
        cached=a.output/(entry['name']+'_predictions.json')
        if a.resume and cached.exists():
            progress('rescoring_saved_predictions',model=entry['name'])
            predictions=json.loads(cached.read_text())
            timing=json.loads((a.output/(entry['name']+'_timing.json')).read_text())
            if {t['image_id'] for t in timing}!={i['id'] for i in images} or len(timing)!=len(images):
                raise ValueError('Cached inference did not cover the full test split')
            source_predictions=common_predictions(json.loads((a.m3fd_results/(entry['name']+'_predictions.json')).read_text()))
            results.append(dict(model=entry['name'],M3FD=score(source,source_predictions),RGBT_Tiny=score(target,predictions),
                mean_service_ms=float(np.mean([r['service_ms'] for r in timing])),p95_service_ms=float(np.percentile([r['service_ms'] for r in timing],95))))
            (a.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
            continue
        progress('evaluating',model=entry['name'],processed=0,total=len(images))
        if entry['family']=='yolo':
            model=load_model(entry['checkpoint'],a.device)
            infer=lambda frame:detect(model,torch.from_numpy(frame).to(a.device),576,conf=.001).cpu().numpy()
        else:
            model=Detector(entry['checkpoint'],a.device)
            infer=lambda frame:model.infer(frame,576)
        predictions=[];timing=[]
        with torch.inference_mode():
            for index,image in enumerate(images):
                visible=raw/'images'/image['file_name']
                parts=Path(image['file_name']).parts
                if parts[1]!='00':raise ValueError('Unexpected modality directory')
                thermal=raw/'images'/parts[0]/'01'/parts[2]
                before=time.monotonic();rgb=cv2.imread(str(visible));ir=cv2.imread(str(thermal),0)
                if rgb is None or ir is None or rgb.shape[:2]!=ir.shape or ir.shape!=(image['height'],image['width']):raise ValueError('Pair/native-coordinate mismatch')
                fused=np.ascontiguousarray(np.dstack([rgb,ir]))
                if index==0:
                    for _ in range(3):infer(fused)
                    torch.cuda.synchronize(a.device);before=time.monotonic()
                decoded=time.monotonic();detections=infer(fused);torch.cuda.synchronize(a.device);end=time.monotonic()
                if not np.isfinite(detections).all():raise ValueError('Nonfinite model output')
                timing.append(dict(image_id=image['id'],decode_ms=(decoded-before)*1000,inference_ms=(end-decoded)*1000,service_ms=(end-before)*1000))
                for x1,y1,x2,y2,confidence,cls in detections:
                    category=int(cls)+1
                    if category not in COMMON:continue
                    predictions.append(dict(image_id=image['id'],category_id=category,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(confidence)))
                if not input_hashes or len(input_hashes)<len(images):
                    input_hashes.append(dict(image_id=image['id'],rgb=str(visible),thermal=str(thermal),rgb_sha256=sha256(visible),thermal_sha256=sha256(thermal)))
                if (index+1)%500==0:progress('evaluating',model=entry['name'],processed=index+1,total=len(images))
        (a.output/(entry['name']+'_predictions.json')).write_text(json.dumps(predictions))
        (a.output/(entry['name']+'_timing.json')).write_text(json.dumps(timing))
        progress('scoring',model=entry['name'])
        source_predictions=common_predictions(json.loads((a.m3fd_results/(entry['name']+'_predictions.json')).read_text()))
        result=dict(model=entry['name'],M3FD=score(source,source_predictions),RGBT_Tiny=score(target,predictions),
            mean_service_ms=float(np.mean([r['service_ms'] for r in timing])),p95_service_ms=float(np.percentile([r['service_ms'] for r in timing],95)))
        results.append(result);(a.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
        (a.output/'input_hashes.json').write_text(json.dumps(input_hashes))
        print(json.dumps(result),flush=True)
        del infer,model;torch.cuda.empty_cache()
    (a.output/'COMPLETED.json').write_text(json.dumps(dict(models=len(results),test_pairs=len(images),evaluated_pairs=len(images)*len(results)),indent=2))
    progress('completed',models=len(results),test_pairs=len(images))


if __name__=='__main__':main()
