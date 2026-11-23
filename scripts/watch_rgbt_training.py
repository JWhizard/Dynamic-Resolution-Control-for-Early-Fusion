"""Mirror a long-running training pipeline and archive its results on completion."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--host',required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--queue-status',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    code="""import json,sys,pathlib
r=pathlib.Path(sys.argv[1]);out={}
for name in ['pipeline_status.json','study/prepare_status.json','study/training_status.json','study/PREPARED.json','study/BEST_MODEL.json','study/development/progress.json']:
 p=r/name
 if p.exists():
  try:out[name]=json.loads(p.read_text())
  except json.JSONDecodeError:pass
print(json.dumps(out))
"""
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host,
        'python3 -c '+shlex.quote(code)+' '+shlex.quote(a.remote_root)]
    while True:
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
            state=json.loads(result.stdout)
            temp=a.output/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(a.output/'status.json')
            queue=json.loads(a.queue_status.read_text()) if a.queue_status.exists() else {}
            queue['target_training']=dict(host=a.host,remote_root=a.remote_root,local_status=str(a.output/'status.json'),**state)
            queue['note']='M3FD additional-family runs are complete. RGBT-Tiny target YOLO pipeline status is recorded in target_training.'
            temp=a.queue_status.with_suffix('.tmp');temp.write_text(json.dumps(queue,indent=2));temp.replace(a.queue_status)
            pipeline=state.get('pipeline_status.json',{}).get('stage')
            if 'study/PREPARED.json' in state and not (a.output/'plan.json').exists():
                for name in ['plan.json','PREPARED.json']:
                    subprocess.run(['scp','-o','BatchMode=yes',a.host+':'+a.remote_root+'/study/'+name,str(a.output/name)],check=True,capture_output=True)
            if pipeline in ['first_stage_completed','failed']:
                subprocess.run(['rsync','-a','--exclude=data','--exclude=*.cache','-e','ssh -o BatchMode=yes',
                    a.host+':'+a.remote_root+'/study/',str(a.output/'artifacts')+'/'],check=True)
                print('Archived terminal state:',pipeline,flush=True);break
        except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as error:
            (a.output/'connection_error.txt').write_text(repr(error))
            print('Connection/archive retry:',repr(error),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
