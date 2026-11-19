"""Same-input TensorRT gate with original PyTorch predictions and labeled AP."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.evaluate_families import sha256,score
from experiments.validate_tensorrt import match
from experiments.family_trt_backend import Detector


def match_degenerate(reference,candidate,threshold):
    """Positive boxes use original IoU gate; zero-area boxes use coordinates.

    Identical clipped zero-area boxes have undefined IoU, not disagreement.
    The separate zero-area check requires class/count parity, .01-pixel
    coordinate agreement and the same .02 score tolerance.
    """
    def partition(x):
        x=x[x[:,4]>=threshold]
        area=np.maximum(0,x[:,2]-x[:,0])*np.maximum(0,x[:,3]-x[:,1])
        return x[area>0],x[area==0]
    positive_a,zero_a=partition(reference);positive_b,zero_b=partition(candidate)
    result=match(positive_a,positive_b,threshold)
    used=set();zero_matches=[]
    for a in sorted(zero_a,key=lambda x:-x[4]):
        choices=[(float(np.max(np.abs(a[:4]-b[:4]))),j)
                 for j,b in enumerate(zero_b) if j not in used and a[5]==b[5]]
        if choices:
            error,j=min(choices)
            if error<=.01 and abs(a[4]-zero_b[j,4])<=.02:
                used.add(j);zero_matches.append(dict(coordinate_error=error,score_error=float(abs(a[4]-zero_b[j,4]))))
    result.update(zero_area_reference_count=len(zero_a),zero_area_candidate_count=len(zero_b),
                  zero_area_matched=len(zero_matches),zero_area_matches=zero_matches,
                  passed=result['passed'] and len(zero_matches)==len(zero_a)==len(zero_b))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['engine','reference','inputs','labels','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--degenerate-aware',action='store_true',help='Supplemental low-confidence geometry diagnostic')
    a=p.parse_args()
    if a.output.exists():p.error('preserve previous validation')
    ref=json.loads(a.reference.read_text());config=json.loads(a.engine.read_text())
    if ref['inputs_sha256']!=sha256(a.inputs) or ref['checkpoint_sha256']!=config['checkpoint_sha256']:
        raise ValueError('reference/input/checkpoint mismatch')
    torch.set_num_threads(4);d=Detector(a.engine);inputs=np.load(a.inputs,allow_pickle=False)
    labels=json.loads(a.labels.read_text());gt=labels['ground_truth'];ids=labels['key_to_id']
    mapping={int(k):int(v) for k,v in labels['class_to_category'].items()} if 'class_to_category' in labels else None
    rows=[];pred_ref={r:[] for r in [480,544,576,640]};pred_gpu={r:[] for r in pred_ref}
    for row in ref['rows']:
        actual=d.infer(inputs[row['key']],row['resolution'])
        original=np.asarray(row['detections']).reshape(-1,6)
        matcher=match_degenerate if a.degenerate_aware else match
        comparisons={str(c):matcher(original,actual,c) for c in [.1,.01,.001]}
        rows.append(dict(key=row['key'],resolution=row['resolution'],comparisons=comparisons))
        if row['key'] in ids:
            for predictions,dets in [(pred_ref,original),(pred_gpu,actual)]:
                for x1,y1,x2,y2,confidence,cls in dets.tolist():
                    if mapping is not None and int(cls) not in mapping:continue
                    category=mapping[int(cls)] if mapping is not None else int(cls)+1
                    predictions[row['resolution']].append(dict(image_id=ids[row['key']],category_id=category,
                        bbox=[x1,y1,x2-x1,y2-y1],score=confidence))
        print(row['key'],row['resolution'],comparisons['0.1']['passed'],flush=True)
    ap=[]
    for r in pred_ref:
        before=score(gt,pred_ref[r]);after=score(gt,pred_gpu[r]);diff=abs(before['AP']-after['AP'])
        ap.append(dict(resolution=r,reference=before,candidate=after,AP_abs_difference=diff,passed=diff<=.005))
    passed=all(r['comparisons']['0.1']['passed'] for r in rows) and all(r['passed'] for r in ap)
    result=dict(passed=passed,gate=dict(confidence=.1,minimum_iou=.9,maximum_score_error=.02,
        equal_detection_counts=True,maximum_validation_AP_difference=.005),family=config['family'],
        engine_sha256=config['engine_sha256'],checkpoint_sha256=config['checkpoint_sha256'],
        inputs_sha256=sha256(a.inputs),reference_sha256=sha256(a.reference),
        validator_source_sha256=sha256(__file__),degenerate_aware=a.degenerate_aware,
        rows=rows,validation_AP=ap,class_to_category=mapping,
        limitations=['Sampled numerical agreement and12 labeled validation pairs, not full test AP equivalence or robustness.',
                     'Low-confidence .01/.001 counts are diagnostic; main exact-count gate is fixed at .10.'])
    a.output.write_text(json.dumps(result,indent=2));d.close()
    print('PASSED',passed,flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
