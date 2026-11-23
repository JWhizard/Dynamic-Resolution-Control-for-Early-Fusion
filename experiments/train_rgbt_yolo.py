"""Fine-tune one seven-class, four-channel model; never load the sealed holdout."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

import torch
import ultralytics
from ultralytics import YOLO,settings

from common.repro import set_global_seed
from experiments.evaluate_families import sha256
from fusion.paired_dataset import make_paired_trainer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Training requires workstation CUDA')
    plan=json.loads((a.study/'plan.json').read_text());prepared=json.loads((a.study/'PREPARED.json').read_text())
    if sha256(a.study/'plan.json')!=prepared['plan_sha256']:raise ValueError('Frozen protocol changed')
    if plan['holdout_unlocked']:raise ValueError('This development stage requires a sealed holdout')
    if sha256(plan['initial_weights'])!=plan['initial_weights_sha256']:raise ValueError('Initial checkpoint changed')
    if ultralytics.__version__!='8.4.14':raise ValueError('Use the frozen Ultralytics8.4.14 environment')
    output=a.study/'training';output.mkdir(exist_ok=False)
    # YOLO_CONFIG_DIR is set to this study by the pipeline before imports.
    for key in ['wandb','comet','mlflow','clearml','neptune','hub','sync']:
        if key in settings:settings.update({key:False})
    set_global_seed(plan['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-8:])
    model=YOLO(plan['initial_weights']);source_stem=model.model.model[0].conv.weight.detach().float().cpu().clone()
    started=[None];batches=[0]
    def status(stage,**fields):
        record=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**fields)
        temp=a.study/'training_status.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(a.study/'training_status.json')
        print(json.dumps(record),flush=True)
    def verify(trainer):
        stem=trainer.model.model[0].conv.weight.detach().float().cpu()
        if stem.shape[1]!=4 or not torch.equal(stem,source_stem):raise ValueError('Four-channel pretrained stem was not preserved')
        if trainer.model.model[-1].nc!=7:raise ValueError('Target head must have seven classes')
        actual=[trainer.data['names'][i] for i in range(7)]
        if actual!=plan['classes']:raise ValueError('Target class order mismatch')
        for split,loader in [('train',trainer.train_loader),('val',trainer.test_loader)]:
            rows=json.loads((a.study/(split+'_manifest.json')).read_text())['frames']
            expected={r['sequence']+'__'+Path(r['rgb']).stem for r in rows}
            if {Path(f).stem for f in loader.dataset.im_files}!=expected:raise ValueError('Unexpected split loaded: '+split)
        sample=trainer.train_loader.dataset[0]
        if sample['img'].shape[0]!=4:raise ValueError('Training data did not provide four channels')
        (output/'PREFLIGHT.json').write_text(json.dumps(dict(stem_preserved=True,channels=4,class_order=actual,
            training_images=len(trainer.train_loader.dataset),validation_images=len(trainer.test_loader.dataset),
            holdout_loaded=False,ultralytics=ultralytics.__version__,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),source_sha256=sha256(__file__)),indent=2)+'\n')
    def begin(trainer):
        started[0]=time.monotonic();status('training',epoch=0,max_hours=plan['training']['max_hours'])
    def epoch_begin(trainer):batches[0]=0
    def batch_end(trainer):
        batches[0]+=1
        elapsed=time.monotonic()-started[0]
        if elapsed>=plan['training']['max_hours']*3600:trainer.stop=True
        if batches[0]%25==0:
            status('training',epoch=trainer.epoch+1,batches=batches[0],total_batches=len(trainer.train_loader),
                elapsed_hours=elapsed/3600,batch_size=trainer.batch_size,gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30)
    def epoch_end(trainer):
        record=dict(epoch=trainer.epoch+1,epoch_complete=batches[0]==len(trainer.train_loader),
            batches=batches[0],fitness=float(trainer.fitness),best_fitness=float(trainer.best_fitness),
            metrics={k:float(v) for k,v in trainer.metrics.items()},elapsed_hours=(time.monotonic()-started[0])/3600)
        with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        status('epoch_completed',**record)
    model.add_callback('on_pretrain_routine_end',verify)
    model.add_callback('on_train_start',begin)
    model.add_callback('on_train_epoch_start',epoch_begin)
    model.add_callback('on_train_batch_end',batch_end)
    model.add_callback('on_fit_epoch_end',epoch_end)
    cfg=plan['training']
    status('initializing',training=cfg)
    model.train(data=str(a.study/'dataset.yaml'),trainer=make_paired_trainer(cfg['multi_scale_sizes']),
        project=str(output),name='yolo26n_target',exist_ok=False,resume=False,
        device='0',seed=plan['seed'],deterministic=True,epochs=cfg['epochs'],batch=cfg['batch'],
        imgsz=cfg['imgsz'],workers=cfg['workers'],optimizer=cfg['optimizer'],lr0=cfg['lr0'],lrf=cfg['lrf'],
        weight_decay=cfg['weight_decay'],warmup_epochs=cfg['warmup_epochs'],cos_lr=True,
        patience=cfg['patience'],mosaic=cfg['mosaic'],mixup=cfg['mixup'],scale=cfg['scale'],
        amp=cfg['amp'],cache=False,plots=True,save=True,save_period=10,val=True,
        hsv_h=.015,hsv_s=.7,hsv_v=.4,fliplr=.5,flipud=0,copy_paste=0,translate=.1)
    best=Path(model.trainer.best)
    if not best.is_file():raise ValueError('Training did not produce a validation-selected best checkpoint')
    backup=output/'best_frozen.pt';shutil.copy2(best,backup)
    frozen=dict(checkpoint=str(backup),sha256=sha256(backup),classes=plan['classes'],channels=4,
        selection='Ultralytics validation mAP50-95 fitness at640; no holdout inference',
        plan_sha256=sha256(a.study/'plan.json'),original_best=str(best))
    (a.study/'BEST_MODEL.json').write_text(json.dumps(frozen,indent=2)+'\n')
    status('training_completed',**frozen)


if __name__=='__main__':main()
