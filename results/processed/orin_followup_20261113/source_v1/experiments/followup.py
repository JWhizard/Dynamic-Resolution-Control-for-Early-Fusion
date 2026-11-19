"""Frozen, versioned GPU follow-up; no target-data tuning or host reconfiguration."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from experiments.pilot import control


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--engine',required=True,type=Path);p.add_argument('--kaist',required=True,type=Path)
    p.add_argument('--flir',type=Path,default=Path('captures/research_flir_pilot/manifest.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policies=['fixed:480','fixed:544','fixed:576','fixed:640','sequence','deadline','aimd',
              'guarded','guarded-no-hysteresis','guarded-no-switch-cost']
    jobs=[]
    for policy in policies:
        jobs.append({'name':'kaist_'+policy.replace(':','-'),'manifest':str(a.kaist),'score':True,
                     'args':['--policy',policy,'--max-frames','1200','--thread-trace','4,4,4']})
    for rep in range(5):
        jobs.append({'name':f'endurance_sequence_rep{rep}','manifest':str(a.flir),'score':False,
                     'args':['--policy','sequence','--cycles','20','--seed',str(rep)]})
    for policy in ['sequence','deadline','guarded']:
        for kind in ['cpu','gpu','memory']:
            jobs.append({'name':f'resource_{kind}_{policy}','manifest':str(a.flir),'score':False,
                         'args':['--policy',policy,'--thread-trace','4,4,4','--contenders',('2' if kind=='cpu' else '1'),
                                 '--contention-kind',kind,'--contention-engine',str(a.engine)]})
    for policy in ['fixed:576','sequence','guarded']:
        jobs.append({'name':'stream_'+policy.replace(':','-'),'manifest':str(a.kaist),'score':True,
                     'args':['--policy',policy,'--max-frames','1200','--mode','realtime','--thread-trace','4,4,4']})
    random.Random(20261113).shuffle(jobs)
    plan={'created_before_runs':datetime.now().isoformat(),'budget_ms':50,'engine':str(a.engine),
          'protocol':'Frozen M3FD detector/controller; first 1200 KAIST preview frames. Not official KAIST test. No target fitting.',
          'guarded_method':'Scene preference capped by online cost plus 1ms switch reserve and four-frame dwell; budget priority.',
          'endurance':'5 process repetitions × 20 cycles × 120 frames; 120 unique frames only.',
          'jobs':jobs}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    original=control();(a.output/'ui_before.json').write_text(json.dumps(original,indent=2));control({'running':False})
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'}
    try:
        for i,job in enumerate(jobs):
            print(f'[{i+1}/{len(jobs)}] {job["name"]}',flush=True)
            cmd=[sys.executable,'-m','experiments.run','--manifest',job['manifest'],'--output',str(a.output/(job['name']+'.json')),
                 '--weights',str(a.engine),'--adapter','experiments.tensorrt_backend:create','--device','tensorrt',
                 '--budget-ms','50','--idle-seconds','2',*job['args']]
            with (a.output/(job['name']+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        print('Paired GPU switching costs',flush=True)
        subprocess.run([sys.executable,'-m','experiments.switch_cost','--manifest',str(a.flir),'--output',str(a.output/'paired_gpu_switch.json'),
                        '--weights',str(a.engine),'--device','tensorrt','--adapter','experiments.tensorrt_backend:create','--reps','10'],env=env,check=True)
    finally:
        control({'running':original['running'],'view':original['view'],'policy':original['policy'],'confidence':original['confidence']})
    for job in jobs:
        if job['score']:
            print('Scoring '+job['name'],flush=True)
            subprocess.run([sys.executable,'-m','experiments.score','--manifest',job['manifest'],
                            '--result',str(a.output/(job['name']+'.json')),'--output',str(a.output/(job['name']+'_accuracy.json'))],check=True,env=env)


if __name__=='__main__':main()
