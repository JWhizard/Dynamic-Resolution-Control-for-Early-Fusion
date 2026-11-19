"""Run one continuous-video policy with raw per-frame costs and sampled rail energy.

Offline mode processes every frame in original order. Realtime mode retains only
latest due input, counts skips, and measures output age against recorded arrivals.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import cv2
import numpy as np
import torch

from experiments.power import RailSampler
from experiments.policies import ACTIONS, Policy
from fusion.preprocess import load_model, detect
from live_dashboard import fuse_frames, DEFAULT_CONTROLLER, DEFAULT_WEIGHTS


def stats(values):
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    return {'n': len(x), 'mean': float(x.mean()), **{f'p{q}': float(np.percentile(x, q)) for q in [50, 95, 99]}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturb(rgb, ir, kind):
    if kind == 'rgb_dark':
        rgb = (rgb.astype(float) * .2).astype(np.uint8)
    elif kind == 'rgb_blur':
        rgb = cv2.GaussianBlur(rgb, (15, 15), 4)
    elif kind == 'ir_dropout':
        ir = np.zeros_like(ir)
    elif kind == 'ir_shift':
        matrix = np.float32([[1, 0, 12], [0, 1, 0]])
        ir = cv2.warpAffine(ir, matrix, (ir.shape[1], ir.shape[0]))
    return rgb, ir


class ProjectDetector:
    def __init__(self, weights, device):
        self.device = torch.device(device)
        self.model = load_model(str(weights), self.device)

    def infer(self, frame, resolution):
        # Feedback retained to .001 permits COCO AP later; controller.update uses
        # its own .10/.25 filters. Copies/synchronization are included in cost.
        return detect(self.model, torch.from_numpy(frame).to(self.device), resolution, conf=.001).cpu().numpy()


def hardware():
    def read(path):
        try:
            return Path(path).read_text().strip('\0\n')
        except OSError:
            return None
    return {'model': read('/proc/device-tree/model'), 'platform': platform.platform(),
            'host': platform.node(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'nvpmodel': subprocess.run(['/usr/sbin/nvpmodel', '-q'], capture_output=True, text=True).stdout
            if Path('/usr/sbin/nvpmodel').exists() else None}


def telemetry():
    def read_sensor(path):
        try:
            return path.read_text().strip()
        except (OSError, ValueError, TypeError):
            return None
    return {'cpu_khz': {p.parent.parent.name: read_sensor(p) for p in
                         Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')},
            'temperatures_mC': {p.parent.name: read_sensor(p) for p in
                               Path('/sys/class/thermal').glob('thermal_zone*/temp')}}


def energy_summary(sampler, start, end, idle):
    rails = sampler.energy(start, end)
    total = sum(rails.values()) if rails and all(v is not None for v in rails.values()) else None
    return {'duration_s': end - start, 'rail_joules': rails, 'named_rail_sum_joules': total,
            'background_subtracted_named_rail_joules': total - idle * (end - start)
            if total is not None and idle is not None else None}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument('--adapter', default='project', help='project or module:factory(weights, device) returning infer(BGRIR, resolution)->Nx6 native xyxy/conf/class')
    p.add_argument('--device', default='cpu')
    p.add_argument('--policy', default='sequence', choices=['sequence','deadline','aimd','random','switch'] + [f'fixed:{r}' for r in ACTIONS])
    p.add_argument('--controller', type=Path, default=DEFAULT_CONTROLLER)
    p.add_argument('--thread-trace', default='4,1,4')
    p.add_argument('--contenders', type=int, default=0, help='owned busy CPU processes in middle third; no system settings changed')
    p.add_argument('--budget-ms', type=float, default=120)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=3, help='per resolution, excluded from measurements')
    p.add_argument('--idle-seconds', type=float, default=2)
    p.add_argument('--mode', choices=['offline', 'realtime'], default='offline')
    p.add_argument('--corruption', choices=['clean','rgb_dark','rgb_blur','ir_dropout','ir_shift'], default='clean')
    p.add_argument('--corruption-middle-only', action='store_true')
    p.add_argument('--max-frames', type=int)
    a = p.parse_args()
    thread_trace = [int(v) for v in a.thread_trace.split(',')]
    if min(thread_trace) < 1 or a.budget_ms <= 0 or a.warmup < 0 or a.idle_seconds < 0 or a.contenders < 0:
        p.error('invalid nonpositive experiment setting')
    if a.output.exists():
        p.error('output already exists; preserve completed experiments')
    manifest = json.loads(a.manifest.read_text())
    frames = manifest['frames'][:a.max_frames]
    if not frames:
        p.error('manifest has no frames')
    timestamps = np.array([f['timestamp_s'] for f in frames], float)
    if np.any(np.diff(timestamps) <= 0):
        p.error('timestamps must be strictly increasing (concatenated videos need a monotonic timeline)')
    allowed = sorted(os.sched_getaffinity(0))[:max(thread_trace)]
    os.sched_setaffinity(0, allowed)
    torch.set_num_threads(thread_trace[0])
    torch.set_num_interop_threads(1)
    root = a.manifest.parent
    def decode(row, corruption):
        rgb, ir = [cv2.imread(str(root / row[name]), cv2.IMREAD_COLOR) for name in ['rgb','thermal']]
        if rgb is None or ir is None:
            raise ValueError('missing/corrupt paired input')
        return fuse_frames(*perturb(rgb, ir, corruption))[0]
    if a.adapter == 'project':
        detector = ProjectDetector(a.weights, a.device)
    else:
        module, factory = a.adapter.split(':')
        detector = getattr(importlib.import_module(module), factory)(a.weights, a.device)
    sync = (lambda: torch.cuda.synchronize(a.device)) if a.device.startswith('cuda') else (lambda: None)
    first = decode(frames[0], 'clean')
    with torch.inference_mode():
        for r in ACTIONS:
            for _ in range(a.warmup):
                detector.infer(first, r)
        sync()
    policy = Policy(a.policy, a.budget_ms, a.seed, a.controller)
    if policy.sequence:
        policy.sequence.groups = {f'frame-{i}': f['sequence'] for i,f in enumerate(frames)}
    sampler = RailSampler()
    sampler.start()
    rows, children = [], []
    try:
        idle_start = time.monotonic()
        time.sleep(a.idle_seconds)
        idle_end = time.monotonic()
        sampler.sample()
        idle_energy = sampler.energy(idle_start, idle_end)
        idle_watts = sum(idle_energy.values()) / (idle_end - idle_start) if idle_end > idle_start and idle_energy and all(v is not None for v in idle_energy.values()) else None
        run_start = time.monotonic()
        sampler.sample()
        i, previous_res, previous_phase = 0, None, None
        with torch.inference_mode():
            while i < len(frames):
                dropped = 0
                if a.mode == 'realtime':
                    time.sleep(max(0, timestamps[i] - timestamps[0] - (time.monotonic() - run_start)))
                    due = int(np.searchsorted(timestamps - timestamps[0], time.monotonic() - run_start, side='right') - 1)
                    due = min(len(frames) - 1, max(i, due))
                    dropped, i = due - i, due
                start = time.monotonic()
                phase = min(len(thread_trace)-1, i * len(thread_trace) // len(frames))
                if phase != previous_phase:
                    torch.set_num_threads(thread_trace[phase])
                    for child in children:
                        child.terminate()
                        child.wait(timeout=5)
                    children = []
                    if a.contenders and len(frames)//3 <= i < 2*len(frames)//3:
                        for k in range(a.contenders):
                            core = allowed[k % len(allowed)]
                            code = f'import os; os.sched_setaffinity(0, {{{core}}}); print("ready", flush=True)\nwhile True: pass'
                            child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
                            children.append(child)
                            if child.stdout.readline().strip() != 'ready':
                                raise RuntimeError('contention worker failed')
                    state = telemetry()
                setup_done = time.monotonic()
                active_corruption = a.corruption
                if a.corruption_middle_only and not len(frames)//3 <= i < 2*len(frames)//3:
                    active_corruption = 'clean'
                fused = decode(frames[i], active_corruption)
                decoded = time.monotonic()
                identifier = f'frame-{i}'
                resolution = policy.choose(identifier)
                chosen = time.monotonic()
                detections = np.asarray(detector.infer(fused, resolution)).reshape(-1, 6)
                sync()
                inferred = time.monotonic()
                if not np.isfinite(detections).all():
                    raise ValueError('nonfinite detector output')
                policy.update(identifier, detections, resolution, (inferred - start)*1000)
                end = time.monotonic()
                rows.append({'frame_index': i, 'image_id': frames[i]['id'], 'sequence': frames[i]['sequence'],
                             'scene': frames[i].get('scene'), 'environment': frames[i].get('environment'),
                             'phase': phase, 'threads': thread_trace[phase], 'contenders': len(children),
                             'corruption': active_corruption, 'resolution': resolution,
                             'switched': previous_res is not None and resolution != previous_res,
                             'resource_transition': previous_phase is not None and phase != previous_phase,
                             'start': start, 'end': end, 'setup_ms': (setup_done-start)*1000,
                             'decode_ms': (decoded-setup_done)*1000, 'choose_ms': (chosen-decoded)*1000,
                             'detector_ms': (inferred-chosen)*1000, 'update_ms': (end-inferred)*1000,
                             'service_ms': (end-start)*1000, 'budget_miss': (end-start)*1000 > a.budget_ms,
                             'output_age_ms': (end-run_start-(timestamps[i]-timestamps[0]))*1000 if a.mode=='realtime' else None,
                             'dropped_before': dropped, 'detections': detections.tolist(),
                             'telemetry_at_phase_start': state if phase != previous_phase else None})
                previous_res, previous_phase = resolution, phase
                i += 1
        run_end = time.monotonic()
        sampler.sample()
    finally:
        for child in children:
            child.terminate()
            child.wait(timeout=5)
        sampler.stop()
    def summarize(part):
        return {'frames': len(part), 'service_ms': stats([r['service_ms'] for r in part]),
                'detector_ms': stats([r['detector_ms'] for r in part]),
                'controller_ms': stats([r['choose_ms']+r['update_ms'] for r in part]),
                'budget_miss_fraction': sum(r['budget_miss'] for r in part)/len(part),
                'resolution_mix': dict(Counter(r['resolution'] for r in part)),
                'energy': energy_summary(sampler, part[0]['start'], part[-1]['end'], idle_watts)}
    summary = summarize(rows)
    summary['energy'] = energy_summary(sampler, run_start, run_end, idle_watts)
    summary['energy']['named_rail_sum_joules_per_frame'] = summary['energy']['named_rail_sum_joules']/len(rows) if summary['energy']['named_rail_sum_joules'] is not None else None
    record = {'schema_version': 1, 'timestamp': datetime.now().isoformat(), 'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'hardware': hardware(), 'weights_sha256': sha(a.weights), 'manifest_sha256': sha(a.manifest),
              'source_sha256': {str(p):sha(p) for p in [Path(__file__),Path('experiments/policies.py'),Path('experiments/power.py'),Path('fusion/preprocess.py'),Path('controller_sequence.py')]},
              'input_frames': len(frames), 'processed_frames':len(rows), 'dropped_frames':sum(r['dropped_before'] for r in rows),
              'idle_named_rail_sum_watts': idle_watts, 'summary': summary,
              'phases': {str(phase):summarize([r for r in rows if r['phase']==phase]) for phase in sorted({r['phase'] for r in rows})},
              'switch_frames_service_ms':stats([r['service_ms'] for r in rows if r['switched']]),
              'steady_frames_service_ms':stats([r['service_ms'] for r in rows if not r['switched']]),
              'power_samples':sampler.samples, 'power_errors':sampler.errors, 'rows':rows,
              'limitations': ['Named rail sum is not wall-plug energy or isolated model energy.',
                'Background subtraction is an estimate and includes other process variation.',
                'Offline service-budget misses are not real-time delivery deadline misses.',
                'Synthetic corruptions do not establish cross-environment robustness.',
                'Tail percentiles from short pilot runs are descriptive, not stable estimates.',
                'Detection accuracy is unmeasured unless a separate labeled evaluation is supplied.']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps({'output':str(a.output),'summary':summary},indent=2))


if __name__ == '__main__':
    main()
