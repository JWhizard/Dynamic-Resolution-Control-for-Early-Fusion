"""Prepare full held-out sequences from the authorized workstation and run replay.

Creates a new study; never resumes into or overwrites an older study directory.
Progress and subprocess output remain on disk if the interactive session ends.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from experiments.evaluate_families import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    previous = Path('/media/imsel/4TBSSD/EMCH585_Artifacts/kaist_video_20261116')
    source_data = Path('/media/imsel/4TBSSD/EMCH585_Artifacts/datasets/kaist/heldout_video_20261116')
    remote = 'whisenaj@10.173.92.11'
    remote_root = '/mnt/12TB_Drive/whisenaj/EMCH585_training/' + root.name
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', remote]

    def state(stage, **extra):
        value = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temp = root/'status.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(root/'status.json')
        print(json.dumps(value), flush=True)

    def run(command):
        print('Running:', command, flush=True)
        subprocess.run(command, check=True, cwd=project)

    try:
        state('freezing_selection')
        run([sys.executable, '-m', 'experiments.kaist_selection', '--root', str(source_data),
             '--output', str(root/'selection.json'), '--full-sequences'])
        selection = json.loads((root/'selection.json').read_text())
        protocol = dict(selection=selection['protocol'], unique_pairs=selection['unique_frames'],
            sequence_lengths={s['sequence']:len(s['frames']) for s in selection['sequences']},
            fps=20, primary_policies=['fixed:480','fixed:544','fixed:576','fixed:640',
                'sequence','guarded','deadline','aimd'], primary_threads='4,1,4',
            control_policies=['fixed:576'], control_threads='4,4,4', budget_ms=100,
            tuning=False, seed=20261118,
            limitations=['Extends previously scored 600-frame prefixes; overlapping data, not an untouched confirmatory test.',
                'Separate day/night sequences do not demonstrate natural day-to-night transitions.',
                'Nominal timestamps; COCO person scoring with crowd-ignore regions, not official KAIST scoring.',
                'CPU preprocessing thread intervention does not establish GPU contention robustness.',
                'One run per case; named-rail energy includes background services.'])
        (root/'protocol_before_scores.json').write_text(json.dumps(protocol, indent=2))
        snapshot = root/'source'
        for directory in ['experiments','fusion','scripts']:
            shutil.copytree(project/directory, snapshot/directory,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(project/'live_dashboard.py', snapshot/'live_dashboard.py')
        # Remote extraction validates the full archive before selecting regular files.
        remote_script = root/'remote_extract.py'
        remote_script.write_text('''import hashlib,json,pathlib,subprocess,sys
archive=pathlib.Path('/mnt/12TB_Drive/whisenaj/Datasets/KAIST/kaist-cvpr15.tar')
h=hashlib.md5()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
expected='fcfc7cfcc1237ad414af64b972f8dc97'
if h.hexdigest()!=expected:raise ValueError('publisher archive checksum mismatch')
stat=archive.stat()
pathlib.Path('verification.json').write_text(json.dumps(dict(actual_md5=h.hexdigest(),publisher_md5=expected,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)))
subprocess.run([sys.executable,'extract_kaist_selection.py','--archive',str(archive),'--selection','selection.json','--verification','verification.json','--output','data'],check=True)
''')
        state('extracting_on_workstation', remote_directory=remote_root)
        run(ssh+['mkdir', remote_root])
        run(['scp','-o','BatchMode=yes',str(root/'selection.json'),str(remote_script),
             str(project/'experiments/extract_kaist_selection.py'),remote+':'+remote_root+'/'])
        run(ssh+['cd '+remote_root+' && python3 remote_extract.py'])
        state('copying_selected_frames')
        run(['rsync','-a','--info=progress2','-e','ssh -o BatchMode=yes',
             remote+':'+remote_root+'/data/',str(root/'data')+'/'])
        state('validating_data_and_building_manifests')
        run([sys.executable,'-m','experiments.prepare_kaist_video','--root',str(root/'data'),
             '--output',str(root/'evaluation')])
        plan = json.loads((previous/'plan.json').read_text())
        plan.update(created_utc=datetime.now(timezone.utc).isoformat(),
            dataset='KAIST full versions of six previously selected official test sequences',
            sequences=json.loads((root/'evaluation/inventory.json').read_text())['sequences'])
        jobs = []
        for model in plan['models']:
            for sequence in plan['sequences']:
                for policy in protocol['primary_policies']:
                    jobs.append(dict(model=model['name'],sequence=sequence['sequence'],
                        policy=policy,condition='thread_reduction'))
                jobs.append(dict(model=model['name'],sequence=sequence['sequence'],
                    policy='fixed:576',condition='constant_threads'))
        random.Random(protocol['seed']).shuffle(jobs)
        plan['jobs'] = jobs
        plan['long_video_protocol'] = protocol
        (root/'plan.json').write_text(json.dumps(plan,indent=2))
        shutil.copy2(root/'evaluation/annotations_all.json',root/'annotations_all.json')
        state('running_model_cases', jobs=len(jobs), unique_pairs=selection['unique_frames'])
        run([sys.executable,'-m','experiments.run_kaist_video','--plan',str(root/'plan.json'),
             '--output',str(root/'runs')])
        completion=json.loads((root/'runs/COMPLETED.json').read_text())
        state('model_runs_completed', **completion)
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
