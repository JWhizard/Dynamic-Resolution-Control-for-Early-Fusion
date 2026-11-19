"""Run the frozen day/night KAIST real-time study and score all input frames."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    for model in plan['models']:
        config=json.loads(Path(model['engine']).read_text())
        gate=json.loads(Path(model['validation']).read_text())
        gate_hash=gate.get('engine_sha256',gate.get('engine',{}).get('engine_sha256'))
        if not gate['passed'] or gate_hash!=config['engine_sha256']:
            raise ValueError('failed/mismatched engine gate')
        if sha256(Path(model['engine']).parent/config['engine'])!=config['engine_sha256']:
            raise ValueError('engine hash changed')
    for sequence in plan['sequences']:
        if sha256(sequence['manifest'])!=sequence['manifest_sha256']:raise ValueError('manifest changed')
    if sha256(plan['controller'])!=plan['controller_sha256']:raise ValueError('controller changed')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'plan.json').write_text(a.plan.read_text())
    records=[]
    for index,job in enumerate(plan['jobs']):
        model=next(x for x in plan['models'] if x['name']==job['model'])
        sequence=next(x for x in plan['sequences'] if x['sequence']==job['sequence'])
        name=f"{index:03d}_{model['name']}_{job['policy']}_{job['condition']}_{job['sequence'].replace('/','_')}"
        output=a.output/(name+'.json');scored=a.output/(name+'_scores.json')
        command=['taskset','-c','8-11','bash','scripts/gpu_python.sh','-m','experiments.run',
            '--manifest',sequence['manifest'],'--output',str(output),'--weights',model['engine'],
            '--adapter',model['adapter'],'--device','tensorrt','--controller',plan['controller'],
            '--mode','realtime','--policy',job['policy'],'--thread-trace',plan['conditions'][job['condition']],
            '--budget-ms',str(plan['budget_ms']),'--warmup','3','--idle-seconds','1']
        print('Starting',name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('inference failed: '+name)
            score_command=[sys.executable,'-m','experiments.score','--manifest',sequence['manifest'],
                '--result',str(output),'--output',str(scored)]
            result=subprocess.run(score_command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('scoring failed: '+name)
        raw=json.loads(output.read_text());scores=json.loads(scored.read_text())
        if raw['processed_frames']+raw['dropped_frames']!=raw['input_frames']:
            raise ValueError('invalid source-frame accounting')
        records.append(dict(job=job,result=output.name,scores_file=scored.name,
            input_frames=raw['input_frames'],processed_frames=raw['processed_frames'],dropped_frames=raw['dropped_frames'],
            summary=raw['summary'],accuracy=scores))
        (a.output/'summary.json').write_text(json.dumps(records,indent=2))
        progress=dict(completed=len(records),total=len(plan['jobs']),latest=name)
        tmp=a.output/'progress.tmp';tmp.write_text(json.dumps(progress,indent=2));tmp.replace(a.output/'progress.json')
        print('Completed',len(records),'/',len(plan['jobs']),flush=True)
    (a.output/'COMPLETED.json').write_text(json.dumps(dict(jobs=len(records),
        input_frame_evaluations=sum(x['input_frames'] for x in records),
        processed_frames=sum(x['processed_frames'] for x in records),
        dropped_frames=sum(x['dropped_frames'] for x in records)),indent=2))


if __name__=='__main__':main()
