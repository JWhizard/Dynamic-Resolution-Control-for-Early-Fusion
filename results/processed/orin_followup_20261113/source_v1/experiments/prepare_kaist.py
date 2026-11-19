"""Prepare a frozen-model, pedestrian-only COCO evaluation of the KAIST preview.

The preview contains one original training sequence (set04/V001, night), not
an official test split. No fitting or policy tuning is performed on these data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    lines=(a.root/'imageSets/train-preview-01.txt').read_text().splitlines()
    frames=[];images=[];annotations=[];environments=set();annotation_digest=hashlib.sha256()
    for i,line in enumerate(lines):
        relative=Path(line);seq=str(relative.parent);stem=relative.name
        rgb=a.root/'images'/relative.parent/'visible'/(stem+'.jpg')
        ir=a.root/'images'/relative.parent/'lwir'/(stem+'.jpg')
        label=a.root/'annotations-xml-new-sanitized'/relative.with_suffix('.xml')
        tree=ET.fromstring(label.read_bytes());annotation_digest.update(label.read_bytes())
        image=cv2.imread(str(rgb));thermal=cv2.imread(str(ir))
        h,w=image.shape[:2]
        if thermal.shape[:2]!=(h,w):raise ValueError('modality dimensions differ')
        if (int(tree.findtext('size/height')),int(tree.findtext('size/width')))!=(h,w):raise ValueError('label coordinates and image dimensions differ')
        environment='night' if int(relative.parts[0][3:]) in [3,4,5,9,10,11] else 'day'
        environments.add(environment)
        frames.append({'id':i,'rgb':str(rgb),'thermal':str(ir),'sequence':seq,'scene':'unannotated',
                       'environment':environment,'timestamp_s':int(stem[1:])/20})
        images.append({'id':i,'file_name':str(rgb),'width':w,'height':h})
        for obj in tree.findall('object'):
            name=obj.findtext('name');box=obj.find('bndbox')
            if name not in ['person','people','person?','cyclist']:raise ValueError('unknown category '+str(name))
            x,y,bw,bh=[float(box.findtext(v)) for v in ['x','y','w','h']]
            x1=max(0,x);y1=max(0,y);x2=min(w,x+bw);y2=min(h,y+bh)
            if x2<=x1 or y2<=y1:continue
            ignored=name!='person' or obj.findtext('difficult','0')=='1'
            annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':1,
                                'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),
                                'iscrowd':int(ignored),'source_name':name})
    gt={'info':{'description':'KAIST preview pedestrian-only COCO adaptation; not official KAIST protocol'},
        'images':images,'annotations':annotations,'categories':[{'id':1,'name':'person'}]}
    (a.output/'annotations.json').write_text(json.dumps(gt))
    manifest={'schema_version':1,'dataset':'KAIST preview set04/V001','source':'https://github.com/SoonminHwang/rgbt-ped-detection',
              'license':'CC BY-NC-SA 4.0 (dataset); retain attribution and publisher terms',
              'timestamp_kind':'source frame index / nominal 20 Hz; not hardware-timestamp verification',
              'registered':'publisher aligned RGB/T','coco_annotations':'annotations.json',
              'class_to_category':{'4':1},'scored_model_classes':[4],
              'annotation_source_sha256':annotation_digest.hexdigest(),'frames':frames,
              'evaluation_protocol':'All person boxes, native coordinates. Other human-like/uncertain categories and difficult person boxes are COCO crowd ignore regions; non-person model classes excluded.',
              'limitations':['One night training-preview sequence, not official KAIST test evaluation.',
                             'Frozen M3FD-trained detector and controller; no target-data fitting.',
                             'No day data, no independent sequence-level robustness inference.',
                             'COCO AP adaptation is not official KAIST log-average miss rate.',
                             'Timing inferred from frame IDs; sensor synchronization not independently measured.']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'manifest':str(a.output/'manifest.json'),'frames':len(frames),
                      'environments':sorted(environments),'person_boxes':sum(not a['iscrowd'] for a in annotations),
                      'ignored_boxes':sum(a['iscrowd'] for a in annotations)}))


if __name__=='__main__':main()
