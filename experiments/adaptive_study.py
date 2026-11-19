"""Predeclared cross-family adaptive-resolution study, without test fitting.

Native aligned pairs are processed serially. Scene history resets at manifest
group boundaries. These boundaries need not represent continuous video.
"""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch

from experiments.detector_families import Detector
from experiments.evaluate_families import score, sha256
from experiments.policies import ACTIONS, Policy
from experiments.power import RailSampler
from fusion.preprocess import detect, load_model


def load_pair(row):
    rgb = cv2.imread(row['rgb'])
    ir = cv2.imread(row['thermal'], cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None or rgb.shape[:2] != ir.shape:
        raise ValueError('invalid registered input pair')
    return np.ascontiguousarray(np.dstack([rgb, ir]))


def summary(values):
    return dict(n=len(values), mean=float(np.mean(values)),
                **{'p'+str(q): float(np.percentile(values, q)) for q in [50, 95, 99]})


def configure_policy(name, budget, controller, groups, profile):
    policy = Policy(name, budget_ms=budget, seed=42, controller=controller)
    if policy.guarded:
        policy.cost.update(profile)
    if policy.sequence:
        policy.sequence.groups = groups
    return policy


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    device = plan['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    for entry in plan['models']:
        if sha256(entry['checkpoint']) != entry['sha256']:
            raise ValueError('checkpoint mismatch')
    if sha256(plan['controller']) != plan['controller_sha256']:
        raise ValueError('controller mismatch')
    manifest_path = Path(plan['manifest'])
    if sha256(manifest_path) != plan['manifest_sha256']:
        raise ValueError('manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    frames = manifest['frames']
    if not frames or len({f['id'] for f in frames}) != len(frames):
        raise ValueError('empty or duplicated inputs')
    gt = json.loads(Path(manifest['ground_truth']).read_text()) if manifest.get('ground_truth') else None
    if gt and {f['id'] for f in frames} != {f['id'] for f in gt['images']}:
        raise ValueError('GT/input ID mismatch')
    groups = {str(f['id']): str(f['group']) for f in frames}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'plan.json').write_text(a.plan.read_text())
    torch.set_num_threads(plan['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    environment = dict(platform=platform.platform(), host=platform.node(),
                       torch=torch.__version__, cuda=torch.version.cuda,
                       execution_device=device, cpu_affinity=sorted(os.sched_getaffinity(0)),
                       source_sha256=sha256(__file__), plan_sha256=sha256(a.plan),
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if device.startswith('cuda'):
        environment['gpu'] = torch.cuda.get_device_name(device)
    (a.output/'environment.json').write_text(json.dumps(environment, indent=2))
    resident = None
    if plan.get('input_mode', 'disk') == 'predecoded':
        before = time.monotonic()
        resident = {f['id']:load_pair(f) for f in frames}
        (a.output/'prefetch.json').write_text(json.dumps(dict(
            seconds=time.monotonic()-before, frames=len(resident),
            host_bytes=sum(f.nbytes for f in resident.values()),
            scope='Pixels only; no predictions/labels supplied to policy. Outside timed windows.'),indent=2))
    random = np.random.default_rng(plan['order_seed'])
    jobs = [(m, policy) for m in plan['models'] for policy in plan['policies']]
    jobs = [jobs[i] for i in random.permutation(len(jobs))]
    (a.output/'job_order.json').write_text(json.dumps([(m['name'],p) for m,p in jobs]))
    results = []
    for entry, name in jobs:
        print('Starting', entry['name'], name, flush=True)
        tag = entry['name']+'_'+name.replace(':', '-')
        if entry['family'] == 'yolo':
            model = load_model(entry['checkpoint'], device)
            infer = lambda f, r: detect(model, torch.from_numpy(f).to(device), r, conf=.001).cpu().numpy()
        else:
            model = Detector(entry['checkpoint'], device)
            infer = model.infer
        sync = (lambda: torch.cuda.synchronize(device)) if device.startswith('cuda') else (lambda: None)
        profile = {}
        profile_frame = load_pair(manifest['profile_frame'])
        profile_start = time.monotonic()
        with torch.inference_mode():
            for resolution in ACTIONS:
                costs = []
                for _ in range(plan['warmup_per_resolution']):
                    start = time.monotonic()
                    infer(profile_frame, resolution)
                    sync()
                    costs.append((time.monotonic()-start)*1000)
                profile[resolution] = float(np.median(costs[-2:]))
        profile_seconds = time.monotonic()-profile_start
        policy = configure_policy(name, plan['budget_ms'], plan['controller'], groups, profile)
        sampler = RailSampler()
        sampler.start()
        rows = []
        try:
            start = time.monotonic()
            with torch.inference_mode():
                previous_resolution, previous_group = None, None
                for index, frame in enumerate(frames):
                    before = time.monotonic()
                    fused = resident[frame['id']] if resident is not None else load_pair(frame)
                    decoded = time.monotonic()
                    resolution = policy.choose(str(frame['id']))
                    chosen = time.monotonic()
                    detections = np.asarray(infer(fused, resolution)).reshape(-1,6)
                    sync()
                    inferred = time.monotonic()
                    if not np.isfinite(detections).all():
                        raise ValueError('nonfinite predictions')
                    policy.update(str(frame['id']), detections, resolution, (inferred-before)*1000)
                    end = time.monotonic()
                    rows.append(dict(image_id=frame['id'], group=frame['group'],
                                     resolution=resolution, detections=detections.tolist(),
                                     decode_ms=(decoded-before)*1000,
                                     controller_ms=(chosen-decoded+end-inferred)*1000,
                                     detector_ms=(inferred-chosen)*1000, service_ms=(end-before)*1000,
                                     switched=previous_resolution is not None and previous_resolution!=resolution,
                                     within_group_switch=previous_group==frame['group'] and previous_resolution!=resolution))
                    previous_resolution, previous_group = resolution, frame['group']
                    if (index+1)%100==0:
                        print(tag,index+1,'/',len(frames),flush=True)
            end = time.monotonic()
        finally:
            sampler.stop()
        energy = sampler.energy(start,end)
        total = sum(energy.values()) if energy and all(v is not None for v in energy.values()) else None
        record = dict(model=entry['name'], family=entry['family'], policy=name,
                      budget_ms=plan['budget_ms'], checkpoint_sha256=entry['sha256'],
                      profile_ms=profile, profile_seconds=profile_seconds, frames=len(rows),
                      service_ms=summary([r['service_ms'] for r in rows]),
                      detector_ms=summary([r['detector_ms'] for r in rows]),
                      controller_ms=summary([r['controller_ms'] for r in rows]),
                      miss_fraction=float(np.mean([r['service_ms']>plan['budget_ms'] for r in rows])),
                      mean_resolution=float(np.mean([r['resolution'] for r in rows])),
                      resolution_mix=dict(Counter(r['resolution'] for r in rows)),
                      switches=sum(r['switched'] for r in rows),
                      within_group_switches=sum(r['within_group_switch'] for r in rows),
                      rail_joules=energy, named_rail_sum_J=total,
                      named_rail_sum_J_per_frame=total/len(rows) if total is not None else None,
                      power_read_errors=sampler.errors, rows=rows,
                      power_samples=sampler.samples)
        if gt:
            predictions = []
            for row in rows:
                for x1,y1,x2,y2,conf,cls in row['detections']:
                    predictions.append(dict(image_id=row['image_id'],category_id=int(cls)+1,
                                            bbox=[x1,y1,x2-x1,y2-y1],score=conf))
            record['metrics'] = score(gt,predictions)
        (a.output/(tag+'.json')).write_text(json.dumps(record))
        result = {k:v for k,v in record.items() if k not in ['rows','power_samples']}
        results.append(result)
        (a.output/'summary.json').write_text(json.dumps(results,indent=2))
        print('Completed',tag,'p95',result['service_ms']['p95'],flush=True)
        del infer, model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    (a.output/'COMPLETED.json').write_text(json.dumps(dict(jobs=len(results),
        processed_frames=sum(r['frames'] for r in results),
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))


if __name__ == '__main__':
    main()
