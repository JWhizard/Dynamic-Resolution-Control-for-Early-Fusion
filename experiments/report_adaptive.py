"""Validate and summarize the frozen cross-family adaptive study."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from experiments.evaluate_families import sha256


def inspect(root, expected_jobs, expected_frames):
    completed=json.loads((root/'COMPLETED.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert completed['jobs']==len(summary)==expected_jobs
    assert completed['processed_frames']==expected_frames*expected_jobs
    assert len({(r['model'],r['policy']) for r in summary})==expected_jobs
    for r in summary:
        tag=r['model']+'_'+r['policy'].replace(':','-')
        raw=json.loads((root/(tag+'.json')).read_text())
        assert r['frames']==len(raw['rows'])==expected_frames
        assert len({x['image_id'] for x in raw['rows']})==expected_frames
        assert sum(r['resolution_mix'].values())==expected_frames
        assert not r['power_read_errors']
        if plan['device']=='cpu':
            assert r['named_rail_sum_J_per_frame'] is not None
            assert all(x is not None for x in r['rail_joules'].values())
    return summary,plan


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--workstation',type=Path,required=True)
    p.add_argument('--buffered',type=Path,required=True)
    p.add_argument('--jetson',type=Path,required=True,help='CPU study runs directory')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ws,wp=inspect(a.workstation,24,419)
    bs,bp=inspect(a.buffered,24,419)
    js,jp=inspect(a.jetson,12,32)
    assert wp['controller_sha256']==jp['controller_sha256']
    assert bp['controller_sha256']==wp['controller_sha256']
    assert {r['checkpoint_sha256'] for r in ws}=={r['checkpoint_sha256'] for r in js}
    assert {r['checkpoint_sha256'] for r in ws}=={r['checkpoint_sha256'] for r in bs}
    a.output.mkdir(parents=True,exist_ok=False)
    for label,root in [('workstation',a.workstation),('buffered',a.buffered),('jetson_cpu',a.jetson)]:
        for name in ['summary.json','plan.json','environment.json','job_order.json','COMPLETED.json']:
            shutil.copy2(root/name,a.output/(label+'_'+name))
    def write_csv(rows,path):
        fields=['model','policy','frames','AP','AP50','AP_small','mean_service_ms','mean_decode_ms','mean_detector_ms','p95_ms','p99_ms',
                'controller_mean_ms','miss_fraction','mean_resolution','switches','within_group_switches','named_rail_sum_J_per_frame']
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in rows:
                w.writerow(dict(model=r['model'],policy=r['policy'],frames=r['frames'],
                    **{k:r.get('metrics',{}).get(k) for k in ['AP','AP50','AP_small']},
                    mean_service_ms=r['service_ms']['mean'],
                    mean_decode_ms=r['service_ms']['mean']-r['detector_ms']['mean']-r['controller_ms']['mean'],
                    mean_detector_ms=r['detector_ms']['mean'],p95_ms=r['service_ms']['p95'],p99_ms=r['service_ms']['p99'],
                    controller_mean_ms=r['controller_ms']['mean'],miss_fraction=r['miss_fraction'],
                    mean_resolution=r['mean_resolution'],switches=r['switches'],within_group_switches=r['within_group_switches'],
                    named_rail_sum_J_per_frame=r['named_rail_sum_J_per_frame']))
    write_csv(ws,a.output/'workstation.csv');write_csv(bs,a.output/'buffered.csv');write_csv(js,a.output/'jetson_cpu.csv')
    lines=['# Cross-family adaptive resolution — November 16, 2026','',
        'Completed 24 primary disk-replay and 24 supplemental buffered-input workstation GPU cases (20,112 timed frames), plus 12 Jetson CPU cost cases (384 timed frames). No model/controller fitting or policy selection used the resulting test scores.', '',
        '## Workstation GPU: same grouped M3FD inputs','',
        '419 test pairs per case, grouped and ordered by original ID within group. Common 25 ms service budget; warmup excluded. Table shows AP, measured mean/p95 service cost and deadline misses. One trial per case, on a shared workstation.', '',
        '| Model | Policy | AP % | Mean ms | p95 ms | Miss % | Mean resolution |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(ws,key=lambda r:(r['model'],r['policy'])):
        lines.append(f"| {r['model']} | {r['policy']} | {r['metrics']['AP']*100:.2f} | {r['service_ms']['mean']:.2f} | {r['service_ms']['p95']:.2f} | {r['miss_fraction']*100:.1f} | {r['mean_resolution']:.1f} |")
    lines += ['', '### Pipeline bottleneck at fixed 576','',
        'PNG input reading/decoding dominates this workstation disk-replay pipeline. The 25 ms service budget is largely infeasible even before policy improvements. These service times are not detector-only GPU latency or camera streaming latency. CSV files separate mean decode, detector and controller costs.', '']
    for model in sorted({r['model'] for r in ws}):
        r=next(x for x in ws if x['model']==model and x['policy']=='fixed:576')
        decode=r['service_ms']['mean']-r['detector_ms']['mean']-r['controller_ms']['mean']
        lines.append(f"- {model}: mean decode/read {decode:.2f} ms; detector {r['detector_ms']['mean']:.2f} ms; controller {r['controller_ms']['mean']:.3f} ms.")
    lines += ['', '### Frozen sequence controller versus fixed 576','']
    deltas=[]
    for model in sorted({r['model'] for r in ws}):
        by={r['policy']:r for r in ws if r['model']==model};s,f=by['sequence'],by['fixed:576']
        d=dict(model=model,AP_percentage_point_change=100*(s['metrics']['AP']-f['metrics']['AP']),
               mean_service_percent_change=100*(s['service_ms']['mean']/f['service_ms']['mean']-1))
        deltas.append(d)
        lines.append(f"- {model}: AP change {d['AP_percentage_point_change']:+.2f} percentage points; mean service cost change {d['mean_service_percent_change']:+.1f}%. Descriptive single-run comparison.")
    lines += ['', '## Supplemental GPU benchmark: predecoded host inputs','',
        'Declared after observing the disk-decoding bottleneck. All 419 native BGRIR arrays are loaded into host RAM before timing; prefetch time/bytes are recorded separately. Same checkpoints, policies and25ms budget; no retuning. This isolates inference/controller/H2D costs and is not end-to-end camera latency.', '',
        '| Model | Policy | AP % | Mean ms | p95 ms | Miss % | Mean resolution |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(bs,key=lambda r:(r['model'],r['policy'])):
        lines.append(f"| {r['model']} | {r['policy']} | {r['metrics']['AP']*100:.2f} | {r['service_ms']['mean']:.2f} | {r['service_ms']['p95']:.2f} | {r['miss_fraction']*100:.1f} | {r['mean_resolution']:.1f} |")
    lines += ['', '## Jetson CPU: short unannotated indoor replay','',
        '32 unique recorded FLIR pairs per case; all three detectors run with CPU PyTorch, four threads on cores 8–11. Budget is 1,000 ms for this CPU pilot. CPU results are separate from prior YOLO TensorRT GPU measurements. No new-family Jetson GPU deployment is claimed.', '',
        '| Model | Policy | Mean ms | p95 ms | Controller ms | Named rail sum J/frame | Miss % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(js,key=lambda r:(r['model'],r['policy'])):
        lines.append(f"| {r['model']} | {r['policy']} | {r['service_ms']['mean']:.2f} | {r['service_ms']['p95']:.2f} | {r['controller_ms']['mean']:.3f} | {r['named_rail_sum_J_per_frame']:.3f} | {r['miss_fraction']*100:.1f} |")
    lines += ['', '## Interpretation and limits','',
        '- These are actual serial inference runs, not selection from future predictions or cached costs. Fixed resolutions are 480, 544, 576 and 640. Sequence, guarded, deadline EMA and AIMD use only past predictions/costs. Policies and configurations were saved before execution; job order was randomized with seed 20261116.',
        '- The same frozen YOLO-derived scene controller is transferred unchanged across families. Warmup cost profiles use one validation frame on the workstation; they do not fit to test labels. Jetson cost-only warmup uses the first unannotated indoor pair.',
        '- M3FD has 218 short/static test groups. Scene history resets at group changes; group ordering does not establish true video continuity. Resource-cost history may persist across groups. Most groups are shorter than the sequence controller interval, limiting adaptation opportunities.',
        '- The fixed-resolution test scores were already examined before this study. This is a predeclared follow-up on the same test split, not a new independent holdout. No parameters were changed based on these outcomes.',
        '- Local EMA/AIMD/guarded implementations are engineering baselines, not faithful reproductions of published adaptive systems. One run per case does not establish statistical superiority, stable p99 or robustness across environments.',
        '- Service includes decoding, policy choice/update, detector preprocessing/inference/postprocessing and synchronization. Outer result-record serialization is excluded from per-frame service but included in run energy. Controller timing also includes the finite-output check. Quadratic cost projection across resolutions can be inaccurate for detector-specific proposal/ROI work.',
        '- Model preprocessing and training budgets differ. No architecture-superiority conclusion follows from the absolute AP ranking.',
        '- Jetson CPU data have no detection labels. Named-rail integrals include dashboard/camera/background workload; they are neither isolated model energy nor wall-plug power. Rail-level values and samples are preserved. Profiles/model loading are outside the timed energy window.',
        '- Shared-host load, thermal state and ordered jobs confound small runtime differences. The CPU pilot uses only 32 frames per case, so its percentiles are exploratory. No matched CPU-versus-GPU speedup or continuous-video streaming claim is made.',
        '- Next: validate GPU exports of the additional families on Jetson, run longer repeated cost experiments, and evaluate all families on labeled continuous video and real environment transitions.', '',
        '## Reproduction and artifacts','',
        'Source entrypoint: `experiments/adaptive_study.py`. Each copied plan freezes inputs, models, controller, budget, CPU threads and job order seed. Use the same plan with a NEW output directory:', '',
        '```bash','python -m experiments.adaptive_study --plan /path/to/plan.json --output /NEW/output','```','',
        'For the recorded CPU affinity, prefix the Jetson command with `taskset -c 8-11`. For GPU execution, set CUDA_VISIBLE_DEVICES to the assigned GPU UUID. Check available resources first and preserve completed outputs.', '',
        'Workstation raw data: '+str(a.workstation),
        '', 'Jetson raw data: '+str(a.jetson), '',
        'Buffered raw data: '+str(a.buffered),'',
        'Raw files retain per-frame native detections, resolution choices, controller/decode/inference timings, switching, warmup profiles and rail samples. Source snapshots preserve the primary implementation and its later predecode extension separately. See raw_index.json for hashes and validation.json for integrity checks.']
    lines += ['', '## Figures', '',
              '![Disk-replay accuracy/cost](accuracy_cost.png)', '',
              '![Buffered GPU accuracy/cost](buffered_accuracy_cost.png)', '',
              '![Jetson CPU host-energy pilot](jetson_cpu_energy.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.output/'sequence_deltas.json').write_text(json.dumps(deltas,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=sorted({r['model'] for r in ws})
    palette={p:plt.get_cmap('tab10')(i) for i,p in enumerate(sorted({r['policy'] for r in ws}))}
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,model in zip(axes,families):
        for r in [x for x in ws if x['model']==model]:
            fixed=r['policy'].startswith('fixed:')
            ax.scatter(r['service_ms']['mean'],r['metrics']['AP']*100,
                       marker='s' if fixed else 'o',s=42,color=palette[r['policy']],label=r['policy'])
        ax.set_title(model);ax.set_xlabel('Mean service time (ms)')
        ax.set_ylabel('COCO AP (%)');ax.grid(alpha=.25)
    fig.suptitle('Workstation GPU: single-run accuracy/cost observations')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=8)
    fig.savefig(a.output/'accuracy_cost.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,model in zip(axes,families):
        for r in [x for x in bs if x['model']==model]:
            ax.scatter(r['service_ms']['mean'],r['metrics']['AP']*100,s=42,
                       marker='s' if r['policy'].startswith('fixed:') else 'o',
                       color=palette[r['policy']],label=r['policy'])
        ax.set_title(model);ax.set_xlabel('Mean inference/controller time (ms)')
        ax.set_ylabel('COCO AP (%)');ax.grid(alpha=.25)
    fig.suptitle('Supplemental GPU benchmark: host inputs predecoded, transfer included')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=8)
    fig.savefig(a.output/'buffered_accuracy_cost.png',dpi=180,bbox_inches='tight');plt.close(fig)
    policies=['fixed:576','sequence','deadline','guarded']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,model in zip(axes,families):
        by={r['policy']:r for r in js if r['model']==model}
        ax.bar(range(4),[by[p]['named_rail_sum_J_per_frame'] for p in policies],color='#477dab')
        ax.set_xticks(range(4),['576','sequence','deadline','guarded'],rotation=25)
        ax.set_ylabel('Named-rail sum (J/frame)');ax.set_title(model)
        ax.grid(axis='y',alpha=.25)
    fig.suptitle('Jetson CPU: 32-pair pilot; host/background energy included')
    fig.savefig(a.output/'jetson_cpu_energy.png',dpi=180);plt.close(fig)
    index={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for root in [a.workstation,a.buffered,a.jetson.parent] for p in root.rglob('*') if p.is_file()}
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps(dict(workstation_jobs=24,workstation_frames=10056,
        buffered_jobs=24,buffered_frames=10056,
        jetson_cpu_jobs=12,jetson_cpu_frames=384,all_counts_and_ids_valid=True,
        all_jetson_energy_windows_covered=True,power_read_errors=0,
        frozen_controller_and_checkpoint_sets_match=True,targeted_tests_passed=10),indent=2)+'\n')


if __name__=='__main__':
    main()
