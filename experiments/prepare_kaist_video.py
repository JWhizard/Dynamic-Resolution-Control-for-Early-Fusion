"""Validate archive-extracted dense KAIST prefixes and build labeled manifests."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
from experiments.evaluate_families import sha256
from experiments.kaist_selection import select


def convert_objects(tree,image_id,width,height,start_id):
    annotations=[]
    for obj in tree.findall('object'):
        name=obj.findtext('name');box=obj.find('bndbox')
        if name not in ['person','people','person?','cyclist']:
            raise ValueError('unknown label '+str(name))
        x,y,w,h=[float(box.findtext(k)) for k in ['x','y','w','h']]
        x1,y1=max(0,x),max(0,y);x2,y2=min(width,x+w),min(height,y+h)
        if x2<=x1 or y2<=y1:continue
        ignored=name!='person' or obj.findtext('difficult','0')=='1'
        annotations.append(dict(id=start_id+len(annotations),image_id=image_id,category_id=1,
            bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=int(ignored),
            source_name=name,source_occlusion=obj.findtext('occlusion'),source_truncated=obj.findtext('truncated')))
    return annotations


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    selection=json.loads((a.root/'selection.json').read_text())
    extraction=json.loads((a.root/'EXTRACTION.json').read_text())
    for f in extraction['files']:
        if sha256(a.root/f['path'])!=f['sha256']:raise ValueError('archive copy mismatch '+f['path'])
    def lines(name):return (a.root/'imageSets'/(name+'.txt')).read_text().splitlines()
    chosen=select(lines('test-all-01'),lines('test-day-01'),lines('test-night-01'),lines('train-all-01'),selection.get('selection_length',600))
    if chosen!=selection['sequences']:raise ValueError('selection differs from official archive split lists')
    a.output.mkdir(parents=True,exist_ok=False);inventory=[];all_images=[];all_annotations=[]
    for seq_index,seq in enumerate(chosen):
        directory=a.output/seq['sequence'].replace('/','_');directory.mkdir()
        frames=[];images=[];annotations=[]
        for index,name in enumerate(seq['frames']):
            path=Path(name);stem=path.name;image_id=len(all_images)+index
            rgb=a.root/'images'/path.parent/'visible'/(stem+'.jpg')
            ir=a.root/'images'/path.parent/'lwir'/(stem+'.jpg')
            label=a.root/'annotations-xml-new-sanitized'/path.with_suffix('.xml')
            image=cv2.imread(str(rgb));thermal=cv2.imread(str(ir),cv2.IMREAD_GRAYSCALE)
            if image is None or thermal is None or image.shape[:2]!=thermal.shape:raise ValueError('invalid paired frame')
            height,width=thermal.shape;tree=ET.fromstring(label.read_bytes())
            if (int(tree.findtext('size/width')),int(tree.findtext('size/height')))!=(width,height):
                raise ValueError('native annotation dimensions differ')
            frames.append(dict(id=image_id,rgb=str(rgb),thermal=str(ir),sequence=seq['sequence'],
                scene=seq['sequence'],environment=seq['environment'],timestamp_s=int(stem[1:])/20,
                source_frame=stem,rgb_sha256=sha256(rgb),thermal_sha256=sha256(ir),annotation_sha256=sha256(label)))
            images.append(dict(id=image_id,file_name=str(rgb),width=width,height=height))
            annotations+=convert_objects(tree,image_id,width,height,len(annotations)+1)
        gt=dict(info={},images=images,annotations=annotations,categories=[dict(id=1,name='person')])
        (directory/'annotations.json').write_text(json.dumps(gt))
        manifest=dict(schema_version=1,dataset='KAIST official test dense sequence selection',sequence=seq['sequence'],
            source='https://github.com/SoonminHwang/rgbt-ped-detection',license='CC BY-NC-SA4.0',
            publisher_md5=extraction['verification']['publisher_md5'],
            timestamp_kind='source index / nominal20Hz; no independent sensor timestamp verification',
            registered='publisher beam-splitter alignment; residual registration not independently certified',
            coco_annotations='annotations.json',class_to_category={'4':1},scored_model_classes=[4],
            annotation_version='archive annotations-xml-new-sanitized; native x/y/w/h clipped to image bounds',
            annotation_convention='Person valid unless difficult; people/person?/cyclist/difficult become COCO crowd-ignore regions. Not official KAIST reasonable/log-average-miss-rate scoring.',
            frames=frames)
        mp=directory/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
        inventory.append(dict(sequence=seq['sequence'],environment=seq['environment'],manifest=str(mp),
            manifest_sha256=sha256(mp),annotations_sha256=sha256(directory/'annotations.json'),frames=len(frames),
            person_boxes=sum(not x['iscrowd'] for x in annotations),ignored_boxes=sum(x['iscrowd'] for x in annotations),
            native_dimensions=sorted({(x['width'],x['height']) for x in images})))
        all_images+=images
        for annotation in annotations:all_annotations.append(dict(annotation,id=len(all_annotations)+1))
    (a.output/'annotations_all.json').write_text(json.dumps(dict(info={},images=all_images,
        annotations=all_annotations,categories=[dict(id=1,name='person')])))
    (a.output/'inventory.json').write_text(json.dumps(dict(sequences=inventory,
        unique_pairs=len(all_images),train_test_disjoint=True,archive_file_hashes_verified=len(extraction['files'])),indent=2))
    print(json.dumps(inventory,indent=2),flush=True)


if __name__=='__main__':main()
