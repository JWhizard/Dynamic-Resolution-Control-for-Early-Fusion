"""Run frozen Jetson GPU cases only after numerical validation passes."""
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
        actual=sha256(Path(model['engine']).parent/config['engine'])
        if actual!=config['engine_sha256']:raise ValueError('engine hash mismatch')
        if model.get('validation'):
            gate=json.loads(Path(model['validation']).read_text())
            if not gate['passed']:raise ValueError('failed numerical gate')
            gate_hash=gate.get('engine_sha256',gate.get('engine',{}).get('engine_sha256'))
            if gate_hash!=actual:raise ValueError('validated engine differs')
        else:raise ValueError('every benchmark engine requires validation')
    if sha256(plan['manifest'])!=plan['manifest_sha256'] or sha256(plan['controller'])!=plan['controller_sha256']:
        raise ValueError('input manifest/controller mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'plan.json').write_text(a.plan.read_text())
    results=[]
    for i,job in enumerate(plan['jobs']):
        model=next(m for m in plan['models'] if m['name']==job['model'])
        name=f"{i:03d}_{model['name']}_{job['policy'].replace(':','-')}_rep{job['repeat']}"
        cmd=['taskset','-c','8-11','bash','scripts/gpu_python.sh','-m','experiments.run',
             '--manifest',plan['manifest'],'--output',str(a.output/(name+'.json')),
             '--weights',model['engine'],'--adapter',model['adapter'],'--device','tensorrt',
             '--policy',job['policy'],'--controller',plan['controller'],
             '--thread-trace','4,4,4','--budget-ms',str(plan['budget_ms']),
             '--warmup','3','--idle-seconds','1','--max-frames','120']
        print('Starting',name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            (a.output/'FAILED.json').write_text(json.dumps(dict(job=job,returncode=completed.returncode,log=name+'.log')))
            raise RuntimeError('benchmark failed: '+name)
        raw=json.loads((a.output/(name+'.json')).read_text())
        results.append(dict(job=job,result_file=name+'.json',summary=raw['summary']))
        (a.output/'progress.json').write_text(json.dumps(dict(completed_jobs=len(results),
            total_jobs=len(plan['jobs']),latest=name),indent=2))
        (a.output/'summary.json').write_text(json.dumps(results,indent=2))
        print('Completed',i+1,'/',len(plan['jobs']),flush=True)
    (a.output/'COMPLETED.json').write_text(json.dumps(dict(jobs=len(results),
        processed_frames=sum(x['summary']['frames'] for x in results)),indent=2))


if __name__=='__main__':main()
