"""Score native-coordinate predictions against externally supplied COCO labels.

Requires manifest.coco_annotations and manifest.class_to_category; no guessed
class mapping, generated pseudo-labels, or subset-only scoring for dropped input.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());result=json.loads(a.result.read_text())
    if result.get('arguments',{}).get('cycles',1)>1:
        raise ValueError('Repeated latency endurance cycles cannot be used for accuracy scoring')
    if not manifest.get('coco_annotations') or not manifest.get('class_to_category'):
        raise ValueError('Labeled COCO annotations and an explicit detector-class mapping are required')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import hashlib
    if hashlib.sha256(a.manifest.read_bytes()).hexdigest()!=result['manifest_sha256']:
        raise ValueError('manifest differs from the measured input')
    mapping={int(k):int(v) for k,v in manifest['class_to_category'].items()}
    gt=COCO(str(a.manifest.parent/manifest['coco_annotations']))
    ids=[f['id'] for f in manifest['frames'][:result['input_frames']]]
    if len(set(ids))!=len(ids) or any(i not in gt.imgs for i in ids):
        raise ValueError('input frame IDs must be unique and present in ground truth')
    import cv2
    for f in manifest['frames'][:result['input_frames']]:
        frame=cv2.imread(str(a.manifest.parent/f['rgb']))
        target=gt.imgs[f['id']]
        if frame is None or frame.shape[:2]!=(target['height'],target['width']):
            raise ValueError('annotation coordinates do not match calibrated input dimensions')
    evaluated_classes=manifest.get('scored_model_classes')
    def decode_predictions(rows,confidence=0):
        detections=[]
        for row in rows:
            for x1,y1,x2,y2,score_value,cls in row['detections']:
                if score_value<confidence:continue
                if evaluated_classes is not None and int(cls) not in evaluated_classes:continue
                detections.append({'image_id':row['image_id'],'category_id':mapping[int(cls)],
                                   'bbox':[x1,y1,x2-x1,y2-y1],'score':score_value})
        return detections
    def detection_coco(detections):
        if detections:return gt.loadRes(detections)
        dt=COCO();dt.dataset={'images':gt.dataset['images'],'categories':gt.dataset['categories'],'annotations':[]};dt.createIndex();return dt
    score_cache={}
    def score(image_ids,rows):
        key=(tuple(image_ids),id(rows))
        if key in score_cache:return score_cache[key]
        ev=COCOeval(gt,detection_coco(decode_predictions(rows)),'bbox');ev.params.imgIds=image_ids
        with contextlib.redirect_stdout(io.StringIO()):ev.evaluate();ev.accumulate();ev.summarize()
        metrics=dict(zip(['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large'],map(float,ev.stats)))
        import numpy as np
        recall=COCOeval(gt,detection_coco(decode_predictions(rows,.25)),'bbox')
        recall.params.imgIds=image_ids;recall.params.iouThrs=np.array([.5])
        with contextlib.redirect_stdout(io.StringIO()):recall.evaluate();recall.accumulate()
        person_ids=[category for category in recall.params.catIds if gt.cats[category]['name'].lower() in ['person','people','pedestrian']]
        if len(person_ids)==1:
            person=recall.params.catIds.index(person_ids[0])
            metrics['person_recall_iou50_conf25']=float(recall.eval['recall'][0,person,0,-1])
            metrics['small_person_recall_iou50_conf25']=float(recall.eval['recall'][0,person,1,-1])
        score_cache[key]=metrics
        return metrics
    environments={f.get('environment','unknown') for f in manifest['frames'][:result['input_frames']]}
    output={'overall':score(ids,result['rows']),'environments':{env:score([f['id'] for f in manifest['frames'][:result['input_frames']] if f.get('environment','unknown')==env],result['rows']) for env in environments},
            'dropped_frame_treatment':'All input images scored; skipped inputs receive no detections.',
            'limitations':'Environment subsets are descriptive; independent held-out sequences and repeated runs are required.'}
    if result.get('arguments',{}).get('mode')=='realtime':
        from experiments.streaming import align_predictions
        aligned=align_predictions(manifest['frames'][:result['input_frames']],result['rows'])
        output['causal_time_aligned']=score(ids,aligned)
        output['alignment_protocol']='Latest completed boxes at each recorded arrival; no tracking, interpolation, or future results. Explicit convention, not an official KAIST score.'
    output['provenance']={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                          'annotations_sha256':hashlib.sha256((a.manifest.parent/manifest['coco_annotations']).read_bytes()).hexdigest(),
                          'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),
                          'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False));print(a.output)


if __name__=='__main__':main()
