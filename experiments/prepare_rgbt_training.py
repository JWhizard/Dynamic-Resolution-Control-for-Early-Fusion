"""Freeze conservative capture groups and a fresh, unscored RGBT-Tiny holdout.

Only the official training partition can supply train/validation/new holdout.
Any prefix/content overlap with the already inspected official test is excluded.
Training uses paired JPEG symlinks and clipped YOLO boxes; native COCO geometry
is retained separately for exact development evaluation. No holdout is loaded by
the training YAML or the automatic evaluation stage.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from experiments.compare_rgbt_tiny import TINY_NAMES
from experiments.evaluate_families import sha256


def prefix(sequence):return '_'.join(sequence.split('_')[:2])


def choose_split(groups,seed=42,trials=5000):
    """Predeclared frame/class-balance objective, with whole connected groups."""
    names=sorted(groups);n=len(names)
    if n<9:raise ValueError('Too few independent candidate groups')
    vectors=np.array([[groups[k]['frames'],*groups[k]['classes']] for k in names],dtype=float)
    target=np.array([.70,.15,.15]);nv=max(1,round(n*.15));nh=nv
    rng=np.random.default_rng(seed);best=None
    for _ in range(trials):
        order=rng.permutation(n);indices=[order[2*nv:],order[:nv],order[nv:2*nv]]
        sums=np.stack([vectors[i].sum(axis=0) for i in indices])
        if np.any(sums[:,1:]<=0):continue
        proportions=sums/vectors.sum(axis=0)
        objective=float(((proportions[:,0]-target)**2).sum()+.2*((proportions[:,1:]-target[:,None])**2).mean())
        if best is None or objective<best[0]:best=(objective,indices)
    if best is None:raise ValueError('Cannot allocate every class to all three splits without group leakage')
    return {split:sorted(names[i] for i in indices) for split,indices in zip(['train','val','holdout'],best[1])}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    raw=a.dataset/'raw';train_path=raw/'annotations_coco/instances_00_train2017.json'
    legacy_path=raw/'annotations_coco/instances_00_test2017.json'
    train=json.loads(train_path.read_text());legacy=json.loads(legacy_path.read_text())
    if [c['name'] for c in sorted(train['categories'],key=lambda c:c['id'])]!=TINY_NAMES:raise ValueError('Class contract changed')
    def status(stage,**fields):
        value=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**fields)
        temp=a.output/'prepare.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(a.output/'prepare_status.json')
        print(json.dumps(value),flush=True)
    def sequence(image):return Path(image['file_name']).parts[0]
    legacy_prefixes={prefix(sequence(i)) for i in legacy['images']}
    # Pixel hashes catch identical decoded pairs even if JPEG bytes differ.
    def pair(image):
        path=Path(image['file_name']);visible=raw/'images'/path
        thermal=raw/'images'/path.parts[0]/'01'/path.name
        rgb=cv2.imread(str(visible));ir=cv2.imread(str(thermal),0)
        if rgb is None or ir is None or rgb.shape[:2]!=ir.shape or ir.shape!=(image['height'],image['width']):raise ValueError('Native pair shape mismatch')
        h=hashlib.sha256();h.update(rgb.tobytes());h.update(ir.tobytes())
        return dict(rgb=str(visible),thermal=str(thermal),rgb_sha256=sha256(visible),thermal_sha256=sha256(thermal),pixel_pair_sha256=h.hexdigest())
    legacy_hashes=set()
    for index,image in enumerate(legacy['images'],1):
        legacy_hashes.add(pair(image)['pixel_pair_sha256'])
        if index%2000==0:status('auditing_legacy_test_content',frames=index,total=len(legacy['images']))
    annotations=defaultdict(list)
    for box in train['annotations']:annotations[box['image_id']].append(box)
    frames=[];quarantined={};parents={}
    def find(k):
        parents.setdefault(k,k)
        if parents[k]!=k:parents[k]=find(parents[k])
        return parents[k]
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parents[max(x,y)]=min(x,y)
    duplicate_owner={}
    for index,image in enumerate(train['images'],1):
        seq=sequence(image);group=prefix(seq)
        if group in legacy_prefixes:
            quarantined[group]='prefix occurs in already inspected official test';continue
        record=dict(image_id=image['id'],sequence=seq,prefix=group,**pair(image));frames.append(record);find(group)
        digest=record['pixel_pair_sha256']
        if digest in legacy_hashes:quarantined[group]='exact decoded pair occurs in already inspected official test'
        if digest in duplicate_owner:union(group,duplicate_owner[digest])
        else:duplicate_owner[digest]=group
        if index%2000==0:status('auditing_candidate_training_content',frames=index,total=len(train['images']))
    bad_roots={find(k) for k in quarantined if k in parents}
    groups={}
    for frame in frames:
        group=find(frame['prefix'])
        if group in bad_roots:continue
        frame['group']=group
        value=groups.setdefault(group,dict(frames=0,classes=[0]*7,prefixes=[]))
        value['frames']+=1
        if frame['prefix'] not in value['prefixes']:value['prefixes'].append(frame['prefix'])
        for box in annotations[frame['image_id']]:
            if not box.get('ignore',0) and not box.get('iscrowd',0):value['classes'][box['category_id']]+=1
    assignment=choose_split(groups)
    split_by_group={g:s for s,gs in assignment.items() for g in gs}
    frames=[f for f in frames if 'group' in f and f['group'] in split_by_group]
    for f in frames:f['split']=split_by_group[f['group']]
    for x in ['train','val','holdout']:
        hx={f['pixel_pair_sha256'] for f in frames if f['split']==x}
        for y in ['train','val','holdout']:
            if x<y and hx&{f['pixel_pair_sha256'] for f in frames if f['split']==y}:raise ValueError('Decoded-pair leakage')
    # Freeze selection, budget and all evaluation actions before model fitting.
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=42,classes=TINY_NAMES,
        class_count=7,channel_order='B,G,R,IR',initial_weights=str(a.weights),initial_weights_sha256=sha256(a.weights),
        train_annotation_sha256=sha256(train_path),legacy_test_annotation_sha256=sha256(legacy_path),
        split_algorithm='5000 seeded whole-group permutations; minimize frame-fraction error +0.2 class-fraction error; each split contains all7 classes',
        target_fractions=[.70,.15,.15],groups=groups,assignment=assignment,quarantined_prefixes=quarantined,
        counts=dict(Counter(f['split'] for f in frames)),training=dict(epochs=100,max_hours=6,batch=16,workers=4,patience=20,
            optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,warmup_epochs=3,
            imgsz=640,multi_scale_sizes=[480,544,576,640],mosaic=0.0,mixup=0.0,scale=.2,amp=True,cache=False),
        checkpoint_selection='Ultralytics8.4.14 validation mAP50-95 fitness at640; exact native COCO AP reported separately',
        development_fixed_resolutions=[320,480,544,576,640],development_policies=['sequence','guarded','deadline','aimd'],
        development_budget_ms=25,development_mode='offline; no streaming/drop claims',
        reported_person_recall='IoU0.5, confidence0.25, maxDets100; person and small-person recall',
        holdout_unlocked=False,holdout_policy='Do not evaluate until the new controller and final protocol are locked. Official test remains previously examined exploratory data.',
        grouping_limit='DJI prefixes are conservative candidate capture groups; exact pixel duplicates are merged. This does not prove location/weather independence.',
        source_sha256=sha256(__file__))
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'frame_inventory.json').write_text(json.dumps(frames))
    image_by_id={i['id']:i for i in train['images']}
    for split in ['train','val','holdout']:
        chosen=sorted([f for f in frames if f['split']==split],key=lambda f:(f['sequence'],image_by_id[f['image_id']]['file_name']))
        ids={f['image_id'] for f in chosen}
        gt=dict(info={},images=[image_by_id[f['image_id']] for f in chosen],annotations=[dict(b,iscrowd=int(bool(b.get('ignore',0) or b.get('iscrowd',0)))) for b in train['annotations'] if b['image_id'] in ids],categories=train['categories'])
        (a.output/(split+'_annotations.json')).write_text(json.dumps(gt))
        (a.output/(split+'_manifest.json')).write_text(json.dumps(dict(split=split,frames=chosen),indent=2))
    data=a.output/'data';clipped=0
    for frame in frames:
        split=frame['split']
        if split=='holdout':continue
        image=image_by_id[frame['image_id']];stem=frame['sequence']+'__'+Path(image['file_name']).stem
        for folder,key in [('images','rgb'),('thermal','thermal')]:
            dst=data/folder/split/(stem+'.jpg');dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(frame[key])
        lines=[];w,h=image['width'],image['height']
        for b in annotations[frame['image_id']]:
            if b.get('iscrowd',0) or b.get('ignore',0):continue
            x,y,bw,bh=b['bbox'];x1,y1=max(0,x),max(0,y);x2,y2=min(w,x+bw),min(h,y+bh)
            clipped+=int([x1,y1,x2-x1,y2-y1]!=[x,y,bw,bh])
            if x2<=x1 or y2<=y1:continue
            lines.append(f"{b['category_id']} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}")
        dst=data/'labels'/split/(stem+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text('\n'.join(lines)+'\n')
    dataset=dict(path=str(data),train='images/train',val='images/val',channels=4,names=TINY_NAMES)
    (a.output/'dataset.yaml').write_text(yaml.safe_dump(dataset,sort_keys=False))
    (a.output/'PREPARED.json').write_text(json.dumps(dict(counts=plan['counts'],groups={k:len(v) for k,v in assignment.items()},
        plan_sha256=sha256(a.output/'plan.json'),frame_inventory_sha256=sha256(a.output/'frame_inventory.json'),
        clipped_training_validation_boxes=clipped,no_exact_decoded_pairs_cross_splits=True,holdout_in_training_yaml=False),indent=2)+'\n')
    status('prepared',counts=plan['counts'],groups={k:len(v) for k,v in assignment.items()})


if __name__=='__main__':main()
