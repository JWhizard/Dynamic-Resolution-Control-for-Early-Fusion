"""Verify completed Jetson GPU deployment measurements and write review artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import csv
import shutil
import numpy as np
from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=a.raw/'benchmarks';plan=json.loads((b/'plan.json').read_text())
    done=json.loads((b/'COMPLETED.json').read_text());summary=json.loads((b/'summary.json').read_text())
    if done['jobs']!=72 or done['processed_frames']!=8640 or len(summary)!=72:
        raise ValueError('incomplete measurement matrix')
    groups=defaultdict(list);raws={};source_hashes={sha256(f) for f in (a.raw/'source').rglob('*') if f.is_file()}
    for entry in summary:
        path=b/entry['result_file'];raw=json.loads(path.read_text());job=entry['job']
        if raw['input_frames']!=120 or raw['processed_frames']!=120 or raw['dropped_frames']:
            raise ValueError('invalid frame accounting')
        if len(raw['rows'])!=120 or len({r['image_id'] for r in raw['rows']})!=120:
            raise ValueError('duplicate/missing frames')
        if raw['power_errors'] or any(v is None for v in raw['summary']['energy']['rail_joules'].values()):
            raise ValueError('incomplete rail measurement')
        if raw['summary']['energy']['named_rail_sum_joules_per_frame'] is None:
            raise ValueError('energy not available')
        if raw['manifest_sha256']!=plan['manifest_sha256']:
            raise ValueError('input manifest mismatch')
        if not all(h in source_hashes for h in raw['source_sha256'].values()):
            raise ValueError('source snapshot mismatch')
        model=next(m for m in plan['models'] if m['name']==job['model'])
        config=json.loads(Path(model['engine']).read_text())
        if raw['detector_metadata']['engine_sha256']!=config['engine_sha256']:
            raise ValueError('unexpected engine')
        if job['model']!='yolo26n_ms' and raw['detector_metadata']['adapter_source_sha256']!=sha256(a.raw/'source/experiments/family_trt_backend.py'):
            raise ValueError('adapter source mismatch')
        groups[(job['model'],job['policy'])].append(raw);raws[entry['result_file']]=raw
    if len(groups)!=24 or any(len(v)!=3 for v in groups.values()):
        raise ValueError('missing fresh-process repetitions')
    a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for (model,policy),runs in sorted(groups.items()):
        def mean(fn):return float(np.mean([fn(r) for r in runs]))
        row=dict(model=model,policy=policy,repetitions=3,unique_input_pairs=120,timed_frames=360,
            mean_service_ms=mean(lambda r:r['summary']['service_ms']['mean']),
            mean_run_p95_ms=mean(lambda r:r['summary']['service_ms']['p95']),
            min_run_p95_ms=min(r['summary']['service_ms']['p95'] for r in runs),
            max_run_p95_ms=max(r['summary']['service_ms']['p95'] for r in runs),
            mean_run_p99_ms=mean(lambda r:r['summary']['service_ms']['p99']),
            mean_controller_ms=mean(lambda r:r['summary']['controller_ms']['mean']),
            mean_detector_ms=mean(lambda r:r['summary']['detector_ms']['mean']),
            mean_named_rail_J_per_frame=mean(lambda r:r['summary']['energy']['named_rail_sum_joules_per_frame']),
            mean_budget_miss_fraction=mean(lambda r:r['summary']['budget_miss_fraction']),
            mean_switches_per_run=mean(lambda r:sum(x['switched'] for x in r['rows'])))
        all_rows=[x for r in runs for x in r['rows']]
        row['mean_resolution']=float(np.mean([r['resolution'] for r in all_rows]))
        for name in ['decode_ms','preprocess_ms','postprocess_ms','gpu_execute_ms']:
            values=[r[name] if name=='decode_ms' else r['detector_stages_ms'][name] for r in all_rows]
            row['mean_'+name]=float(np.mean(values))
        records.append(row)
    (a.output/'summary.json').write_text(json.dumps(records,indent=2))
    with (a.output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for name in ['retinanet_validation.json','fasterrcnn_validation.json']:
        shutil.copy2(a.raw/name,a.output/name)
    diagnostics=[]
    for model in ['retinanet','fasterrcnn']:
        path=a.raw/(model+'_degenerate_diagnostic.json')
        if path.exists():
            data=json.loads(path.read_text());shutil.copy2(path,a.output/path.name)
            diagnostics.append(dict(model=model,cases=len(data['rows']),
                passed_by_threshold={c:sum(r['comparisons'][c]['passed'] for r in data['rows']) for c in ['0.1','0.01','0.001']},
                zero_area_coordinate_tolerance_pixels=.01))
    shutil.copy2(b/'plan.json',a.output/'plan.json')
    lines=['# Jetson GPU deployment of three detector families — November 16, 2026','',
        'Both frozen four-channel additional-family models were exported to ONNX and built as TensorRT10.3 strict-FP32/no-TF32 engines on Jetson AGX Orin. The complete CNN, proposal/ROI heads and NMS run on GPU; original CPU transforms and native-coordinate restoration remain outside the engine. Existing YOLO26n is measured alongside them.', '',
        '## Prediction validation','',
        'The primary predeclared gate requires identical class-aware detection counts at confidence0.10, IoU≥0.90 and score difference≤0.02. Each model was compared at four resolutions on12 M3FD validation pairs,4 indoor FLIR pairs and one zero-image edge case:68 comparisons per model.', '',
        '| Model | Passed cases | Matched detections | Minimum matched IoU | Maximum score error | Largest validation AP difference (points) |',
        '|---|---:|---:|---:|---:|---:|']
    for model in ['fasterrcnn','retinanet']:
        gate=json.loads((a.raw/(model+'_validation.json')).read_text())
        if not gate['passed']:raise ValueError('failed model gate')
        matches=[m for r in gate['rows'] for m in r['comparisons']['0.1']['matches']]
        lines.append(f"| {model} | {len(gate['rows'])} | {len(matches)} | {min(m['iou'] for m in matches):.6f} | {max(m['score_error'] for m in matches):.6g} | {max(r['AP_abs_difference'] for r in gate['validation_AP'])*100:.4f} |")
    lines += ['', 'Labeled AP comparisons use only the12 selected M3FD validation pairs, separately at each resolution. This is sampled numerical/backend validation, not proof of full-test AP equivalence or environmental robustness.', '',
        'Supplemental low-confidence diagnostic: positive-area boxes retain the original IoU/count/score gate; clipped zero-area boxes are compared by class, count and coordinates within0.01 pixel, with the same score tolerance. This resolves undefined-IoU bookkeeping without changing models, thresholds or the primary gate. The original diagnostics remain saved.', '',
        '```json',json.dumps(diagnostics,indent=2),'```','',
        '## Repeated latency/controller/energy measurements','',
        '72 fresh-process runs:3 detectors ×8 policies ×3 repetitions, each on the same120 recorded indoor RGB/IR pairs (8,640 timed frames total). Warmup, model loading and profiling are excluded. All use a100ms offline service budget,4 CPU threads pinned to cores8–11, the same frozen sequence controller and randomized job order.', '',
        '| Model | Policy | Mean service ms | Mean run p95 ms | Mean run p99 ms | Controller ms | Named-rail J/frame | Budget misses % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['model']} | {r['policy']} | {r['mean_service_ms']:.2f} | {r['mean_run_p95_ms']:.2f} | {r['mean_run_p99_ms']:.2f} | {r['mean_controller_ms']:.3f} | {r['mean_named_rail_J_per_frame']:.3f} | {r['mean_budget_miss_fraction']*100:.1f} |")
    lines += ['', 'CSV includes per-case p95 ranges, stage means, mean resolution and switches. Percentiles shown are means of run-level percentiles; three repeats describe local repeat variation, not independent scenes or reliable population confidence intervals.', '',
        '## Scope and limits','',
        '- All models use the same frozen checkpoints previously tested. No training, class-map changes, score-threshold fitting, power-mode changes or live-camera movement occurred.',
        '- TensorRT executes complete exported detection graphs, including ROI processing and suppression; this is not a CPU detector reported as GPU. Input normalization/resize/padding and coordinate conversion are CPU operations. CUDA events measure the execution span, including possible host/device shape synchronization gaps; they are not isolated kernel-active time.',
        '- The export pads outputs to300 detections with zero-score rows discarded afterwards. It replaces an unsupported empty-NMS If branch with negative-score dummy padding and specifies known Faster R-CNN classifier dimensions for empty-ROI reshapes. No learned weights or thresholds change. Export helper tests and numerical gates protect these transformations.',
        '- RetinaNet low-confidence diagnostics initially show16 IoU-only matching failures at0.01/0.001 despite equal counts: the reference outputs contain clipped zero-area boxes, for which IoU is undefined. These are retained separately from the primary0.10 gate; no arbitrary low-confidence AP equivalence is claimed.',
        '- The cost clip is unannotated. Lower latency/energy at lower resolution does not establish an acceptable accuracy tradeoff. Learned sequence/guarded controllers and EMA/AIMD baselines are local implementations, not faithful reproductions of published adaptive systems.',
        '- Named-rail sums include background acquisition/dashboard and other host load; they are neither wall-plug energy nor isolated model power. Rail traces/errors are retained. Services remained active. Hardware mode, clocks, temperatures and source hashes are recorded per run.',
        '- Only120 unique indoor pairs appear in every repetition. Tail percentiles from120-frame runs remain preliminary. Offline budget misses do not measure real-time output age, dropped live frames, resource-transition recovery or multi-environment robustness.',
        '- Service time includes the harness phase-setup/telemetry work at three boundaries. The4,4,4 thread trace is constant; phase flags are bookkeeping, not actual resource changes. Per-frame setup_ms is retained. Controller cost includes output-finiteness checking plus policy choice/update. Instrumentation can influence short-run tails.',
        '- Engines are target-specific. Rebuild for another TensorRT/device environment and repeat prediction validation before using those timings.', '',
        '## Artifacts and reproduction','',
        'Raw exports, engines, build logs, validation inputs/references and per-frame timing/rail samples: '+str(a.raw), '',
        'Source snapshots and plan hashes preserve exact measurement inputs. See `raw_index.json` and `validation.json`. ONNX/engine hashes and complete builder commands are in each engine.json.', '',
        '```bash',
        '# From the project root, with matching native TensorRT bindings:',
        'bash scripts/gpu_python.sh -m experiments.run \\',
        '  --manifest captures/research_flir_pilot/manifest.json --output /NEW/result.json \\',
        '  --weights /path/to/validated/engine.json --adapter experiments.family_trt_backend:create \\',
        '  --device tensorrt --policy sequence --thread-trace 4,4,4 --budget-ms 100',
        '```', '',
        'Full matrix: `python -m experiments.benchmark_family_trt --plan /path/to/benchmark_plan.json --output /NEW/output`. It refuses engines whose saved numerical gate does not pass or whose hashes differ.', '',
        'Next: longer labeled continuous-video evaluation with scene/resource transitions and held-out environments.']
    lines += ['', '## Figures', '', '![Latency and energy](latency_energy.png)', '',
              '![Fixed576 pipeline stages](pipeline_stages.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=sorted({r['model'] for r in records});policies=plan['policy_names']
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,name in zip(axes,names):
        for i,policy in enumerate(policies):
            r=next(x for x in records if x['model']==name and x['policy']==policy)
            ax.scatter(r['mean_run_p95_ms'],r['mean_named_rail_J_per_frame'],label=policy,
                       color=plt.get_cmap('tab10')(i),marker='s' if policy.startswith('fixed:') else 'o')
        ax.set_title(name);ax.set_xlabel('Mean run p95 service time (ms)');ax.set_ylabel('Named-rail sum (J/frame)');ax.grid(alpha=.25)
    fig.suptitle('Jetson GPU: three fresh-process runs per case; same120-pair indoor clip')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=8)
    fig.savefig(a.output/'latency_energy.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    rows=[next(x for x in records if x['model']==name and x['policy']=='fixed:576') for name in names]
    stages=[('mean_decode_ms','Decode'),('mean_preprocess_ms','CPU transform'),('mean_gpu_execute_ms','GPU execution span'),('mean_postprocess_ms','CPU coordinates'),('mean_controller_ms','Controller')]
    bottom=np.zeros(len(names))
    for key,label in stages:
        values=np.array([r[key] for r in rows]);ax.bar(names,values,bottom=bottom,label=label);bottom+=values
    ax.set_ylabel('Mean measured stage time (ms)');ax.set_title('Fixed576 stages; GPU span excludes transfer/other host overhead')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
    fig.savefig(a.output/'pipeline_stages.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (a.output/'validation.json').write_text(json.dumps(dict(completed_jobs=72,timed_frames=8640,
        unique_source_pairs=120,repetitions_per_case=3,frame_accounting_valid=True,
        all_rail_windows_covered=True,power_read_errors=0,source_and_engine_hashes_match=True,
        new_family_numerical_cases_passed=136),indent=2))
    index={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in a.raw.rglob('*') if p.is_file()}
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()
