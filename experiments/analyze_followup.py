"""Curate actual GPU follow-up results; never pool repeats as independent scenes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def ci_mean(values,seed=20261113):
    x=np.asarray(values,float)
    if len(x)<2:return None
    rng=np.random.default_rng(seed);means=rng.choice(x,size=(10000,len(x)),replace=True).mean(1)
    return {'n_runs':len(x),'mean':float(x.mean()),'bootstrap_mean_ci95':np.percentile(means,[2.5,97.5]).tolist(),
            'definition':'Percentile bootstrap of run-level statistics, not frame-level independent samples.'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);runs=[];index={}
    for path in sorted(a.raw.glob('*.json')):
        r=json.loads(path.read_text());index[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        if isinstance(r,dict) and 'summary'in r and 'rows'in r:
            assert r['processed_frames']+r['dropped_frames']==r['input_frames']
            assert r['detector_metadata']['execution_device']=='Jetson GPU'
            runs.append((path.stem,r))
    if not runs:raise ValueError('no complete GPU runs')
    columns=['run','policy','mode','frames','unique_source_frames','p50_ms','p95_ms','p99_ms','controller_mean_ms',
             'preprocess_mean_ms','gpu_execute_mean_ms','postprocess_mean_ms','rail_sum_J_per_frame','budget_miss_fraction',
             'dropped_frames','switches','u_turns','AP','AP_small','person_recall50_conf25','causal_aligned_AP']
    rows=[];accuracy_records={}
    for name,r in runs:
        s=r['summary'];stages=[f['detector_stages_ms'] for f in r['rows']];res=[f['resolution'] for f in r['rows']]
        apath=a.raw/(name+'_accuracy.json');accuracy=json.loads(apath.read_text()) if apath.exists() else {}
        if accuracy:accuracy_records[name]=accuracy
        row={'run':name,'policy':r['arguments']['policy'],'mode':r['arguments']['mode'],'frames':r['processed_frames'],
             'unique_source_frames':r['unique_source_frames'],'p50_ms':s['service_ms']['p50'],'p95_ms':s['service_ms']['p95'],
             'p99_ms':s['service_ms']['p99'],'controller_mean_ms':s['controller_ms']['mean'],
             'preprocess_mean_ms':float(np.mean([f['preprocess_ms'] for f in stages])),
             'gpu_execute_mean_ms':float(np.mean([f['gpu_execute_ms'] for f in stages])),
             'postprocess_mean_ms':float(np.mean([f['postprocess_ms'] for f in stages])),
             'rail_sum_J_per_frame':s['energy']['named_rail_sum_joules_per_frame'],
             'budget_miss_fraction':s['budget_miss_fraction'],'dropped_frames':r['dropped_frames'],
             'switches':sum(f['switched'] for f in r['rows']),
             'u_turns':sum(res[i]==res[i-2] and res[i]!=res[i-1] for i in range(2,len(res))),
             'AP':accuracy.get('overall',{}).get('AP'),'AP_small':accuracy.get('overall',{}).get('AP_small'),
             'person_recall50_conf25':accuracy.get('overall',{}).get('person_recall_iou50_conf25'),
             'causal_aligned_AP':accuracy.get('causal_time_aligned',{}).get('AP')}
        rows.append(row)
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    endurance=[r for r in rows if r['run'].startswith('endurance')]
    intervals={key:ci_mean([r[key] for r in endurance]) for key in ['p95_ms','p99_ms','controller_mean_ms','rail_sum_J_per_frame']}
    (a.output/'endurance_intervals.json').write_text(json.dumps(intervals,indent=2))
    (a.output/'accuracy.json').write_text(json.dumps(accuracy_records,indent=2))
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2))
    for name in ['plan.json','paired_gpu_switch.json','supplement_plan.json']:
        if (a.raw/name).exists():shutil.copy2(a.raw/name,a.output/name)
    lines=['# Jetson GPU follow-up — November 13, 2026','',
           f'{len(runs)} completed TensorRT GPU runs; {sum(r["frames"] for r in rows)} processed frames.',
           '', 'This study uses one detector family on one physical Jetson AGX Orin.',
           'The KAIST data are a frozen-model check on the first 1,200 frames of one night training-preview sequence, not official KAIST test evaluation.',
           'The FLIR endurance data repeat 120 unique indoor pairs. These are not independent environments.',
           'Strict FP32 TensorRT disables TF32; preprocessing remains the original CPU torch path.',
           'The proposed guarded policy is experimental and is not a published-baseline reproduction.',
           '', '## Measured runs','',
           '| Run | p95 / p99 ms | Controller mean ms | Rail-sum J/frame | Person AP | Causal aligned AP |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.4f}'
    for r in rows:
        lines.append(f"| {r['run']} | {r['p95_ms']:.2f} / {r['p99_ms']:.2f} | {r['controller_mean_ms']:.4f} | {fmt(r['rail_sum_J_per_frame'])} | {fmt(r['AP'])} | {fmt(r['causal_aligned_AP'])} |")
    lines+=['','## Repeated-run uncertainty','',
            'Intervals below bootstrap means of run-level statistics across fresh process repetitions. They do not provide sequence-level or cross-environment confidence intervals.']
    for key,value in intervals.items():
        if value:lines.append(f"- {key}: mean {value['mean']:.4f}; 95% bootstrap interval {value['bootstrap_mean_ci95']}; n={value['n_runs']} runs.")
    lines+=['','## Measurement scope','',
            '- Stage timing includes CPU preprocessing/postprocessing, synchronized GPU execution span and host transfers. Service adds JPEG decoding and controller work.',
            '- Rail sums are not wall-plug/model-only energy. CPU/GPU/memory contention energy includes the competing workload.',
            '- Phase setup is included in service cost; worker initialization can produce a large transition outlier. Warmup/profile wall time is separate in each raw record.',
            '- COCO evaluation uses native coordinates, an explicit People→person mapping, and ignores other human-like/uncertain regions. This is not official KAIST log-average miss rate.',
            '- Causal aligned AP uses only the latest completed result at each recorded arrival; no tracking or future predictions. It is distinct from scoring skipped frames as empty.',
            '- Resource conditions are short diagnostics. Correlation, shared-host load, thermal drift and small repetition counts limit causal/performance claims.',
            '- Natural scene transition annotations, external power calibration, independently held-out domains and trained additional detector families are still required.',
            '- Workstation 10.173.92.11 is authorized for training, but SSH access must be configured before remote jobs can run.',
            '', 'The source study plan was saved before execution. No model or controller parameters were fitted to KAIST labels.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labeled=[r for r in rows if r['run'].startswith('kaist_') and r['AP'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for row in labeled:
        axes[0].scatter(row['p95_ms'],row['AP']);axes[0].annotate(row['policy'],(row['p95_ms'],row['AP']),fontsize=7,xytext=(3,3),textcoords='offset points')
        axes[1].scatter(row['rail_sum_J_per_frame'],row['AP']);axes[1].annotate(row['policy'],(row['rail_sum_J_per_frame'],row['AP']),fontsize=7,xytext=(3,3),textcoords='offset points')
    axes[0].set_xlabel('Service p95 (ms)');axes[1].set_xlabel('Named-rail energy (J/frame)')
    for ax in axes:ax.set_ylabel('Person COCO AP');ax.grid(alpha=.2)
    fig.suptitle('Frozen detector on KAIST night preview prefix — diagnostic, not official benchmark')
    fig.tight_layout();fig.savefig(a.output/'accuracy_cost.png',dpi=160);plt.close(fig)
    validation={'gpu_runs':len(runs),'processed_frames':sum(r['frames'] for r in rows),
                'endurance_runs':len(endurance),'endurance_timed_frames':sum(r['frames'] for r in endurance),
                'power_read_errors':sum(len(r['power_errors']) for _,r in runs),
                'all_rail_energy_covered':all(all(v is not None for v in r['summary']['energy']['rail_joules'].values()) for _,r in runs),
                'all_frame_accounting_valid':True,'labeled_runs':len(accuracy_records),
                'source_versions':{key:sorted({r['source_sha256'][key] for _,r in runs}) for key in runs[0][1]['source_sha256']}}
    (a.output/'validation.json').write_text(json.dumps(validation,indent=2));print(a.output/'REPORT.md')


if __name__=='__main__':main()
