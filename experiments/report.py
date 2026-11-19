"""Summarize completed pilot runs without claiming unlabeled accuracy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    runs=[]
    for path in sorted(a.directory.glob('*.json')):
        record=json.loads(path.read_text())
        if isinstance(record,dict) and 'summary' in record:runs.append((path.stem,record))
    if not runs:raise ValueError('no complete experiment results')
    columns=['run','policy','mode','corruption','frames','p50_ms','p95_ms','p99_ms','controller_mean_ms',
             'budget_miss_fraction','rail_sum_J_per_frame','switches','dropped_frames']
    rows=[]
    for name,r in runs:
        s=r['summary'];arg=r['arguments']
        rows.append(dict(zip(columns,[name,arg['policy'],arg['mode'],arg['corruption'],r['processed_frames'],
                    s['service_ms']['p50'],s['service_ms']['p95'],s['service_ms']['p99'],
                    s['controller_ms']['mean'],s['budget_miss_fraction'],
                    s['energy']['named_rail_sum_joules_per_frame'],sum(f['switched'] for f in r['rows']),r['dropped_frames']])))
    with (a.directory/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    fixed={}
    for name,r in runs:
        arg=r['arguments']
        if name.endswith('_threads') and arg['policy'].startswith('fixed:'):
            for row in r['rows']:fixed[(arg['seed'],row['frame_index'],row['phase'],row['resolution'])]=row
    switches=[]
    for name,r in runs:
        if name.endswith('_threads') and not r['arguments']['policy'].startswith('fixed:'):
            for row in r['rows']:
                target=fixed.get((r['arguments']['seed'],row['frame_index'],row['phase'],row['resolution']))
                if row['switched'] and not row['resource_transition'] and target:
                    switches.append({'run':name,'frame_index':row['frame_index'],'resolution':row['resolution'],
                                     'service_delta_vs_fixed_ms':row['service_ms']-target['service_ms'],
                                     'detector_delta_vs_fixed_ms':row['detector_ms']-target['detector_ms']})
    (a.directory/'switch_matched_deltas.json').write_text(json.dumps(switches,indent=2))
    lines=['# Jetson AGX Orin pilot results','',
           f'{len(runs)} completed runs; {sum(r[1]["processed_frames"] for r in runs)} processed frames.',
           '', 'These are CPU pilot measurements on one unannotated indoor FLIR preview clip.',
           'Accuracy, natural scene-transition robustness, and cross-platform generality are unmeasured.',
           'Short-run p99 values are descriptive. Named-rail energy is not wall-plug energy.',
           'Thread-trace and CPU-contention tests are separate conditions. The C++ capture service remains active.',
           '', '| Run | p50 / p95 / p99 ms | Controller mean ms | Budget misses | Rail-sum J/frame |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        energy=row['rail_sum_J_per_frame']
        lines.append(f"| {row['run']} | {row['p50_ms']:.2f} / {row['p95_ms']:.2f} / {row['p99_ms']:.2f} | {row['controller_mean_ms']:.4f} | {row['budget_miss_fraction']:.1%} | {energy:.4f} |" if energy is not None else f"| {row['run']} | {row['p50_ms']:.2f} / {row['p95_ms']:.2f} / {row['p99_ms']:.2f} | {row['controller_mean_ms']:.4f} | {row['budget_miss_fraction']:.1%} | unavailable |")
    lines+=['','Switch deltas match frame, repetition, resource phase, and destination resolution against a fixed run.',
            'They are descriptive paired comparisons across separate runs; DVFS, temperature, and order can confound them.',
            'Use a randomized same-frame switch/stay microbenchmark for a causal switching-cost estimate.',
            f'{len(switches)} matched switch observations recorded in `switch_matched_deltas.json`.']
    (a.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for name,r in runs:
        if not name.endswith('rep0_threads') or r['arguments']['policy'] in ['switch','random']:continue
        x=[f['frame_index'] for f in r['rows']]
        axes[0].plot(x,[f['service_ms'] for f in r['rows']],label=r['arguments']['policy'],alpha=.8,lw=.9)
        axes[1].plot(x,[f['resolution'] for f in r['rows']],alpha=.8,lw=.9)
        axes[2].plot(x,[f['choose_ms']+f['update_ms'] for f in r['rows']],alpha=.8,lw=.9)
    axes[0].axhline(120,color='black',ls='--',lw=1,label='120 ms budget')
    axes[0].set_ylabel('Service time (ms)');axes[1].set_ylabel('Resolution (px)');axes[2].set_ylabel('Controller (ms)')
    axes[2].set_xlabel('Ordered recorded frame');axes[0].legend(ncol=3,fontsize=8)
    length=max(r['input_frames'] for _,r in runs)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvspan(length/3,2*length/3,color='gray',alpha=.12)
    fig.suptitle('Orin CPU pilot: 4 → 1 → 4 inference threads; one indoor FLIR clip')
    fig.tight_layout();fig.savefig(a.directory/'resource_trace.png',dpi=160);plt.close(fig)
    print(a.directory/'REPORT.md')


if __name__=='__main__':main()
