"""Durable first-stage training/development pipeline; the fresh holdout stays sealed."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,default=3);a=p.parse_args()
    root=a.root.resolve();source=root/'src';study=root/'study'
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,YOLO_CONFIG_DIR=str(root/'ultralytics_settings'),PYTHONHASHSEED='42',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_MODE='disabled',COMET_MODE='DISABLED')
    def status(stage,**fields):
        record=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**fields)
        temp=root/'pipeline_status.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(root/'pipeline_status.json')
        print(json.dumps(record),flush=True)
    def run(module,arguments,log):
        with (root/log).open('x') as output:
            subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],cwd=source,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        status('preparing_capture_groups_and_fresh_holdout')
        run('experiments.prepare_rgbt_training',['--dataset',a.dataset,'--weights',a.weights,'--output',study],'prepare.log')
        observations=0
        while observations<3:
            data=subprocess.check_output(['nvidia-smi',f'--id={a.gpu_index}','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
            uuid,used,total=next(csv.reader(data.splitlines(),skipinitialspace=True))
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
            ready=uuid not in {s.strip() for s in apps} and float(used)<1024 and float(total)-float(used)>=16384
            observations=observations+1 if ready else 0
            status('waiting_for_free_gpu',gpu_index=a.gpu_index,gpu_uuid=uuid,free_observations=observations)
            if observations<3:time.sleep(30)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=uuid)
        (root/'GPU_ASSIGNMENT.json').write_text(json.dumps(dict(index=a.gpu_index,uuid=uuid),indent=2))
        status('training',gpu_uuid=uuid,study=str(study),max_hours=6,epochs_max=100)
        run('experiments.train_rgbt_yolo',['--study',study],'training.log')
        status('evaluating_development_baselines',study=str(study),holdout_evaluated=False)
        run('experiments.evaluate_rgbt_development',['--study',study,'--controller',source/'results/processed/sequence/controller.json','--nvml-index',a.gpu_index],'development.log')
        status('first_stage_completed',study=str(study),holdout_evaluated=False,
            next_step='Analyze development results and design/freeze the reliability-aware controller before unlocking the holdout.')
    except BaseException as error:
        status('failed',error=repr(error),study=str(study))
        raise


if __name__=='__main__':main()
