"""Summarize matched-input deployments, keeping power measurement scopes separate."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.raw/'plan.json').read_text());plan_hash=sha256(a.raw/'plan.json')
    cases=[];groups=defaultdict(list);reference=None;index=[]
    for platform,backend in [('orin','tensorrt'),('workstation','pytorch')]:
        files=sorted((a.raw/platform).glob('*.json'))
        if len(files)!=9:raise ValueError('Need nine completed runs per platform')
        for path in files:
            r=json.loads(path.read_text())
            if r['frames']!=120 or r['resolution']!=576 or r['backend']!=backend or r['plan_sha256']!=plan_hash or r['power_errors']:raise ValueError('Benchmark contract failed')
            if reference is None:reference=r['input_hashes']
            if reference!=r['input_hashes']:raise ValueError('Input content/order differ')
            if len(r['rows'])!=120 or r['J_per_frame']<=0:raise ValueError('Missing frame/energy data')
            c=dict(platform=platform,model=r['model'],backend=backend,run=path.stem,
                mean_ms=r['mean_ms'],p95_ms=r['p95_ms'],p99_ms=r['p99_ms'],J_per_frame=r['J_per_frame'],energy_scope=r['energy_scope'])
            cases.append(c);groups[(platform,r['model'])].append(c)
            index.append(dict(path=str(path),sha256=sha256(path)))
    expected={(platform,m['name']) for platform in ['orin','workstation'] for m in plan['models']}
    if set(groups)!=expected or any(len(v)!=3 for v in groups.values()):raise ValueError('Model/repetition matrix incomplete')
    summary=[]
    for (platform,model),rows in sorted(groups.items()):
        summary.append(dict(platform=platform,model=model,backend=rows[0]['backend'],runs=3,
            mean_service_ms=float(np.mean([r['mean_ms'] for r in rows])),
            mean_run_p95_ms=float(np.mean([r['p95_ms'] for r in rows])),
            min_run_mean_ms=min(r['mean_ms'] for r in rows),max_run_mean_ms=max(r['mean_ms'] for r in rows),
            mean_J_per_frame=float(np.mean([r['J_per_frame'] for r in rows])),energy_scope=rows[0]['energy_scope']))
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('cases',cases),('summary',summary)]:
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps(dict(runs=18,models=3,repetitions_each=3,
        frames_each=120,total_timed_frames=2160,unique_pairs=120,identical_input_hashes_and_order=True,
        common_plan_hash=True,all_power_measurements_present=True,
        energy_scopes_comparable_across_platforms=False),indent=2)+'\n')
    lines=['# Matched input deployment cost comparison','',
        'Three fixed576 detectors ran on the same 120 recorded indoor RGB/thermal pairs, with three fresh processes per detector and platform: 18 runs and 2,160 timed frames. Input arrays were decoded into host RAM before timing. H2D, backend transforms, inference, native-coordinate output conversion and synchronization remain timed. This is not end-to-end camera latency.','',
        '| Platform | Model | Backend | Mean service ms | Mean run p95 ms | Run-mean range ms | Energy/frame J | Energy scope |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"| {r['platform']} | {r['model']} | {r['backend']} | {r['mean_service_ms']:.2f} | {r['mean_run_p95_ms']:.2f} | {r['min_run_mean_ms']:.2f}–{r['max_run_mean_ms']:.2f} | {r['mean_J_per_frame']:.3f} | {r['energy_scope']} |")
    lines+=['','## Interpretation','',
        'The deployments differ in both hardware and backend: Jetson AGX Orin uses previously validated TensorRT 10.3 FP32 engines; RTX 6000 Ada uses native FP32 PyTorch with the same frozen checkpoint identities. Four CPU threads and the first four cores within the permitted process affinity are used. No host power/clock settings or existing camera services were changed. The workstation cost jobs wait for the dataset evaluation to release their assigned GPU.','',
        'Do not interpret service-time differences as a pure hardware effect. Orin energy integrates named host rails including background activity; workstation energy is the NVML GPU-device total-energy counter delta. Those scopes differ, so no cross-platform energy-efficiency ratio is computed. Short 120-frame runs and three repetitions do not establish stable p99, unseen-scene behavior or model-only power. The clip is unannotated; this benchmark does not measure detection accuracy.','',
        'Input SHA-256 lists and ordering matched across all 18 runs, and every run contains 120 timed outputs with energy measurements. Warmup (12 fixed576 calls), model loading, prefetch and one idle second precede timing. The backend adapters retain their trained preprocessing; fixed long-side size does not imply identical tensor shape or kernel work.','',
        'Raw plans, per-frame detections/timings, Orin rail samples, source snapshots and launch logs: '+str(a.raw)+'. Recreate the export using `python -m experiments.report_platform_cost --raw /path/to/raw --output /path/to/summary`.','',
        '![Deployment service costs](service_cost.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=[m['name'] for m in plan['models']];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for platform,offset,color in [('orin',-.18,'#298b7d'),('workstation',.18,'#477aa6')]:
        rs=[next(r for r in summary if r['model']==m and r['platform']==platform) for m in models]
        ax.bar(np.arange(3)+offset,[r['mean_service_ms'] for r in rs],width=.36,color=color,label=platform)
    ax.set_xticks(np.arange(3),models);ax.set_ylabel('Mean service time (ms)');ax.set_title('Identical predecoded inputs; hardware and backends differ');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(a.output/'service_cost.png',dpi=180);plt.close(fig)
    print(a.output/'REPORT.md')


if __name__=='__main__':main()
