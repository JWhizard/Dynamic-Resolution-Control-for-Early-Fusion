"""GPU-only additional-family training; use the authorized workstation, not Jetson CPU.

No training has been run merely by creating this script. Selection uses the
provided grouped validation split; test images are never loaded by this entrypoint.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from experiments.detector_families import build,NAMES


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self,root,ids,augment=False):self.root,self.ids,self.augment=Path(root),ids,augment
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        stem=self.ids[index]
        rgb=cv2.imread(str(self.root/'Vis'/(stem+'.png')));ir=cv2.imread(str(self.root/'Ir'/(stem+'.png')),cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None or rgb.shape[:2]!=ir.shape:raise ValueError('missing or unregistered pair '+stem)
        h,w=ir.shape;boxes=[];labels=[]
        for line in (self.root/'labels_yolo'/(stem+'.txt')).read_text().splitlines():
            if not line.strip():continue
            cls,cx,cy,bw,bh=map(float,line.split()[:5]);x1=max(0,(cx-bw/2)*w);y1=max(0,(cy-bh/2)*h);x2=min(w,(cx+bw/2)*w);y2=min(h,(cy+bh/2)*h)
            if x2<=x1 or y2<=y1:raise ValueError('degenerate label '+stem)
            if cls!=int(cls) or not 0<=int(cls)<6:raise ValueError('invalid class '+stem)
            boxes.append([x1,y1,x2,y2]);labels.append(int(cls)+1)
        image=np.dstack([rgb,ir]);boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        if self.augment and random.random()<.5:
            image=image[:,::-1].copy();boxes[:,[0,2]]=w-boxes[:,[2,0]]
        target={'boxes':boxes,'labels':torch.tensor(labels,dtype=torch.int64),'image_id':torch.tensor(index),
                'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),'iscrowd':torch.zeros(len(labels),dtype=torch.int64)}
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float()/255,target


def collate(batch):return tuple(zip(*batch))


def validate_splits(spec):
    splits=spec['splits'];sets=[set(splits[k]) for k in ['train','val','test']]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('overlapping train/validation/test IDs')
    if 'groups' not in spec:raise ValueError('group mapping required for grouped training')
    mapping=spec['groups']
    if any(i not in mapping for values in sets for i in values):raise ValueError('missing group membership')
    groups=[{mapping[i] for i in values} for values in sets]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('group leakage across train/validation/test')
    return splits


def evaluate(model,dataset,device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images=[];annotations=[];predictions=[];model.eval();model.transform.min_size=(576,);model.transform.max_size=576
    with torch.inference_mode():
        for index in range(len(dataset)):
            image,target=dataset[index];h,w=image.shape[-2:];images.append({'id':index,'height':h,'width':w})
            for box,label in zip(target['boxes'].tolist(),target['labels'].tolist()):
                x1,y1,x2,y2=box;annotations.append({'id':len(annotations)+1,'image_id':index,'category_id':label,'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
            result=model([image.to(device)])[0]
            for box,label,score in zip(result['boxes'].cpu().tolist(),result['labels'].cpu().tolist(),result['scores'].cpu().tolist()):
                if label==0 or score<.001:continue
                x1,y1,x2,y2=box;predictions.append({'image_id':index,'category_id':label,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset={'info':{},'images':images,'annotations':annotations,'categories':[{'id':i+1,'name':n} for i,n in enumerate(NAMES)]};gt.createIndex()
        if predictions:dt=gt.loadRes(predictions)
        else:dt=COCO();dt.dataset={'images':images,'annotations':[],'categories':gt.dataset['categories']};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    return {'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP_small':float(ev.stats[3])}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--family',required=True,choices=['fasterrcnn','retinanet'])
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--splits',type=Path,default=Path('splits/m3fd_grouped_seed42.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-hours',type=float,default=2);a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():p.error('training requires an available CUDA GPU on the authorized workstation')
    if a.epochs<1 or a.batch_size<1 or a.max_hours<=0:p.error('epochs, batch size and time budget must be positive')
    for directory in ['Vis','Ir','labels_yolo']:
        if not (a.data_root/directory).is_dir():p.error('missing dataset directory '+directory)
    try:splits=validate_splits(json.loads(a.splits.read_text()))
    except ValueError as exc:p.error(str(exc))
    a.output.mkdir(parents=True,exist_ok=False);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=build(a.family,True).to(a.device)
    train=PairedDataset(a.data_root,splits['train'],True);validation=PairedDataset(a.data_root,splits['val'],False)
    loader=torch.utils.data.DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,collate_fn=collate,
                                       generator=torch.Generator().manual_seed(a.seed),pin_memory=True)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[int(a.epochs*.6),int(a.epochs*.8)],gamma=.1)
    metadata={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'channel_order':'B,G,R,IR',
              'class_names':NAMES,'family':a.family,'split_sha256':hashlib.sha256(a.splits.read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__,
              'builder_sha256':hashlib.sha256(Path('experiments/detector_families.py').read_bytes()).hexdigest(),
              'gpu':torch.cuda.get_device_name(a.device),'test_data_loaded':False,
              'recipe_status':'Provisional comparable-data recipe; not yet matched to YOLO training/tuning budget or instructor-approved.'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2));start=time.monotonic();best=-1
    for epoch in range(a.epochs):
        model.train();losses=[]
        for images,targets in loader:
            if time.monotonic()-start>a.max_hours*3600:break
            resolution=random.choice([320,480,544,576,640]);model.transform.min_size=(resolution,);model.transform.max_size=resolution
            images=[im.to(a.device,non_blocking=True) for im in images];targets=[{k:v.to(a.device) for k,v in t.items()} for t in targets]
            loss=sum(model(images,targets).values())
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();losses.append(float(loss.detach()))
        if not losses:break
        scheduler.step();metrics=evaluate(model,validation,a.device)
        row={'epoch':epoch+1,'epoch_complete':len(losses)==len(loader),'mean_loss':float(np.mean(losses)),'validation':metrics,'seconds':time.monotonic()-start}
        with (a.output/'epochs.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        saved={'model_state':model.state_dict(),'family':a.family,'class_names':NAMES,'channel_order':'B,G,R,IR','epoch':epoch+1,'validation':metrics}
        torch.save(saved,a.output/'last.pt')
        if metrics['AP']>best:best=metrics['AP'];torch.save(saved,a.output/'best.pt')
        print(json.dumps(row),flush=True)
        if time.monotonic()-start>a.max_hours*3600:break


if __name__=='__main__':main()
