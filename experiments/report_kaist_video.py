"""Validate and summarize frozen-model, native-arrival KAIST replay results."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.evaluate_families import score,sha256
from experiments.streaming import align_predictions
from experiments.content_transitions import presence_events


def person_predictions(rows):
    output=[]
    for row in rows:
        for x1,y1,x2,y2,confidence,cls in row['detections']:
            if int(cls)!=4:continue
            output.append(dict(image_id=row['image_id'],category_id=1,
                bbox=[x1,y1,x2-x1,y2-y1],score=confidence))
    return output


def subset_gt(gt,image_ids):
    ids=set(image_ids)
    return dict(info=gt.get('info',{}),categories=gt['categories'],
        images=[x for x in gt['images'] if x['id'] in ids],
        annotations=[x for x in gt['annotations'] if x['image_id'] in ids])


def restore_recovery(rows,budget_ms):
    restored=[r for r in rows if r['phase']==2]
    if not restored:return None
    for index in range(2,len(restored)):
        if all(r['service_ms']<=budget_ms for r in restored[index-2:index+1]):
            return (restored[index]['end']-restored[0]['start'])*1000
    return None


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.raw/'runs';plan=json.loads((root/'plan.json').read_text())
    completed=json.loads((root/'COMPLETED.json').read_text());entries=json.loads((root/'summary.json').read_text())
    if completed['jobs']!=90 or len(entries)!=90 or completed['input_frame_evaluations']!=54000:
        raise ValueError('incomplete study')
    a.output.mkdir(parents=True,exist_ok=False)
    manifests={s['sequence']:json.loads(Path(s['manifest']).read_text()) for s in plan['sequences']}
    all_gt=json.loads((a.raw/'annotations_all.json').read_text())
    source_hashes={sha256(f) for f in (a.raw/'source').rglob('*') if f.is_file()}
    records=[];pools=defaultdict(lambda:dict(processed=[],processed_ids=[],aligned=[],cases=[]));events=[]
    for seq in plan['sequences']:
        manifest=manifests[seq['sequence']]
        labels=json.loads((Path(seq['manifest']).parent/'annotations.json').read_text())
        for event in presence_events(manifest['frames'],labels['annotations']):
            events.append(dict(sequence=seq['sequence'],environment=seq['environment'],**event))
    phase_records=[]
    for entry in entries:
        job=entry['job'];raw_path=root/entry['result'];raw=json.loads(raw_path.read_text())
        scored=json.loads((root/entry['scores_file']).read_text());frames=manifests[job['sequence']]['frames']
        seq=next(s for s in plan['sequences'] if s['sequence']==job['sequence'])
        if raw['input_frames']!=600 or raw['processed_frames']+raw['dropped_frames']!=600:
            raise ValueError('frame accounting mismatch')
        if raw['power_errors'] or raw['summary']['energy']['named_rail_sum_joules'] is None:
            raise ValueError('invalid energy integration')
        if raw['manifest_sha256']!=seq['manifest_sha256'] or scored['provenance']['result_sha256']!=sha256(raw_path):
            raise ValueError('manifest/result provenance mismatch')
        if scored['provenance']['annotations_sha256']!=seq['annotations_sha256'] or scored['provenance']['scorer_sha256']!=sha256(a.raw/'source/experiments/score.py'):
            raise ValueError('annotation/scorer provenance mismatch')
        if not all(h in source_hashes for h in raw['source_sha256'].values()):
            raise ValueError('inference source snapshot mismatch')
        model=next(m for m in plan['models'] if m['name']==job['model'])
        engine=json.loads(Path(model['engine']).read_text())
        if raw['weights_sha256']!=sha256(model['engine']) or raw['detector_metadata']['engine_sha256']!=engine['engine_sha256']:
            raise ValueError('engine provenance mismatch')
        rows=raw['rows'];indices=[r['frame_index'] for r in rows]
        if len(indices)!=len(set(indices)) or indices!=sorted(indices):raise ValueError('invalid processed order')
        expected=[int(x) for x in plan['conditions'][job['condition']].split(',')]
        for r in rows:
            if r['image_id']!=frames[r['frame_index']]['id'] or r['threads']!=expected[r['phase']]:
                raise ValueError('source ID or thread budget mismatch')
        aligned=align_predictions(frames,rows)
        age=[r['output_age_ms'] for r in rows]
        by_id={f['id']:f for f in frames}
        display_age=[(by_id[r['image_id']]['timestamp_s']-frames[r['source_frame_index']]['timestamp_s'])*1000 for r in aligned]
        energy=raw['summary']['energy']['named_rail_sum_joules']
        record=dict(model=job['model'],policy=job['policy'],condition=job['condition'],sequence=job['sequence'],
            environment=seq['environment'],source_frames=600,processed_frames=len(rows),dropped_frames=raw['dropped_frames'],
            dropped_fraction=raw['dropped_frames']/600,AP_missing_empty=scored['overall']['AP'],
            AP50_missing_empty=scored['overall']['AP50'],AP_causal=scored['causal_time_aligned']['AP'],
            AP50_causal=scored['causal_time_aligned']['AP50'],AP_small_causal=scored['causal_time_aligned']['AP_small'],
            recall50_conf25_causal=scored['causal_time_aligned']['person_recall_iou50_conf25'],
            mean_service_ms=raw['summary']['service_ms']['mean'],p95_service_ms=raw['summary']['service_ms']['p95'],
            p99_service_ms=raw['summary']['service_ms']['p99'],mean_controller_ms=raw['summary']['controller_ms']['mean'],
            mean_completion_age_ms=float(np.mean(age)),p95_completion_age_ms=float(np.percentile(age,95)),
            mean_display_age_ms=float(np.mean(display_age)),p95_display_age_ms=float(np.percentile(display_age,95)),
            initial_unavailable_fraction=1-len(aligned)/600,budget_miss_fraction=raw['summary']['budget_miss_fraction'],
            named_rail_J=energy,J_per_source_frame=energy/600,J_per_completed_inference=energy/len(rows),
            mean_resolution=float(np.mean([r['resolution'] for r in rows])),switches=sum(r['switched'] for r in rows),
            recovery_three_on_budget_ms=restore_recovery(rows,plan['budget_ms']) if job['condition']=='thread_reduction' else None)
        records.append(record)
        key=(job['model'],job['policy'],job['condition']);pool=pools[key]
        pool['processed']+=person_predictions(rows);pool['processed_ids'] += [r['image_id'] for r in rows]
        pool['aligned']+=person_predictions(aligned);pool['cases'].append(record)
        for phase in range(3):
            part=[r for r in rows if r['phase']==phase]
            phase_records.append(dict(model=job['model'],policy=job['policy'],condition=job['condition'],sequence=job['sequence'],
                phase=phase,threads=expected[phase],source_frames=200,processed_frames=len(part),dropped_frames=200-len(part),
                mean_service_ms=float(np.mean([r['service_ms'] for r in part])),
                mean_completion_age_ms=float(np.mean([r['output_age_ms'] for r in part])),
                mean_resolution=float(np.mean([r['resolution'] for r in part]))))
    groups=[];pooled_scores={}
    env_ids={env:[f['id'] for m in manifests.values() for f in m['frames'] if f['environment']==env] for env in ['day','night']}
    for key,pool in sorted(pools.items()):
        model,policy,condition=key;cases=pool['cases']
        if len(cases)!=6:raise ValueError('missing sequence in comparison cell')
        micro=score(all_gt,pool['aligned']);missing=score(all_gt,pool['processed'])
        processed_only=score(subset_gt(all_gt,pool['processed_ids']),pool['processed'])
        env_scores={}
        for env,ids in env_ids.items():
            allowed=set(ids);pred=[r for r in pool['aligned'] if r['image_id'] in allowed]
            env_scores[env]=score(subset_gt(all_gt,ids),pred)
        tag=model+'_'+policy.replace(':','-')+'_'+condition
        pooled_scores[tag]=dict(causal=micro,missing_empty=missing,processed_only_diagnostic=processed_only,environments=env_scores)
        groups.append(dict(model=model,policy=policy,condition=condition,sequences=6,
            pooled_causal_AP=micro['AP'],pooled_missing_empty_AP=missing['AP'],processed_only_AP_diagnostic=processed_only['AP'],
            macro_causal_AP=float(np.mean([r['AP_causal'] for r in cases])),
            day_pooled_AP=env_scores['day']['AP'],night_pooled_AP=env_scores['night']['AP'],
            day_macro_AP=float(np.mean([r['AP_causal'] for r in cases if r['environment']=='day'])),
            night_macro_AP=float(np.mean([r['AP_causal'] for r in cases if r['environment']=='night'])),
            worst_sequence_AP=min(r['AP_causal'] for r in cases),
            mean_drop_fraction=float(np.mean([r['dropped_fraction'] for r in cases])),
            mean_service_ms=float(np.mean([r['mean_service_ms'] for r in cases])),
            mean_case_p95_service_ms=float(np.mean([r['p95_service_ms'] for r in cases])),
            mean_case_p95_completion_age_ms=float(np.mean([r['p95_completion_age_ms'] for r in cases])),
            mean_case_p95_display_age_ms=float(np.mean([r['p95_display_age_ms'] for r in cases])),
            mean_controller_ms=float(np.mean([r['mean_controller_ms'] for r in cases])),
            mean_J_per_source_frame=float(np.mean([r['J_per_source_frame'] for r in cases])),
            mean_J_per_completed_inference=float(np.mean([r['J_per_completed_inference'] for r in cases]))))
    write_csv(a.output/'cases.csv',records);write_csv(a.output/'summary.csv',groups);write_csv(a.output/'phases.csv',phase_records)
    (a.output/'cases.json').write_text(json.dumps(records,indent=2));(a.output/'pooled_accuracy.json').write_text(json.dumps(pooled_scores,indent=2))
    (a.output/'content_events.json').write_text(json.dumps(dict(definition='Post-hoc stable person-present/no-person transitions;>=10 source frames, never passed to policies.',events=events),indent=2))
    paired=[]
    for model in sorted({r['model'] for r in records}):
        for sequence in manifests:
            pair={r['condition']:r for r in records if r['model']==model and r['sequence']==sequence and r['policy']=='fixed:576'}
            constant,changed=pair['constant_threads'],pair['thread_reduction']
            paired.append(dict(model=model,sequence=sequence,
                causal_AP_point_change=100*(changed['AP_causal']-constant['AP_causal']),
                drop_fraction_change=changed['dropped_fraction']-constant['dropped_fraction'],
                p95_completion_age_ms_change=changed['p95_completion_age_ms']-constant['p95_completion_age_ms'],
                J_per_source_frame_change=changed['J_per_source_frame']-constant['J_per_source_frame']))
    write_csv(a.output/'paired_fixed_controls.csv',paired)
    shutil.copy2(root/'plan.json',a.output/'plan.json');shutil.copy2(a.raw/'data_inventory.json',a.output/'data_inventory.json')
    for name in ['retinanet_gate.json','fasterrcnn_gate.json']:shutil.copy2(a.raw/name,a.output/name)
    lines=['# KAIST held-out day/night real-time replay — November16,2026','',
        'Full official archive verified against publisher MD5 fcfc7cfcc1237ad414af64b972f8dc97 on the workstation. A predeclared subset of3,600 dense paired frames was extracted directly from that archive, hash-verified on Jetson, and evaluated as six independent30-second prefixes at nominal20Hz.', '',
        'Source: [KAIST official release](https://github.com/SoonminHwang/rgbt-ped-detection), Hwang et al., CVPR2015; dataset license CC BY-NC-SA4.0. Actual decoded dimensions are640×512. The six sequences are set06–11/V000; training and test memberships are disjoint. This is a selected subset, not the complete official test benchmark.', '',
        '## Frozen-model results','',
        'All three GPU detectors and the M3FD-derived controller remain frozen. Four policies use a controlled Torch CPU-thread budget4→1→4 over the source thirds (0–10,10–20,20–30s). Fixed576 also has a constant4→4→4 control for each model/sequence:90 cases total. Native20Hz arrivals have50ms periods; the inherited100ms service budget is a separate policy constraint.', '',
        '| Model | Policy | Threads | Pooled causal AP % | Day AP % | Night AP % | Worst-sequence AP % | Drops % | Mean case p95 output age ms | J/source frame |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        lines.append(f"| {r['model']} | {r['policy']} | {r['condition']} | {100*r['pooled_causal_AP']:.3f} | {100*r['day_pooled_AP']:.3f} | {100*r['night_pooled_AP']:.3f} | {100*r['worst_sequence_AP']:.3f} | {100*r['mean_drop_fraction']:.1f} | {r['mean_case_p95_completion_age_ms']:.1f} | {r['mean_J_per_source_frame']:.3f} |")
    lines += ['', '## Metric definitions and scope','',
        '- Only model class4 (People) maps to KAIST person. Other detector classes are outside this benchmark. The archive sanitized XML is converted to native x/y/w/h COCO boxes; people/person?/cyclist/difficult regions are crowd-ignore regions. No reasonable-height filter or official KAIST log-average miss-rate evaluation is claimed.',
        '- Causal AP uses only the latest completed result at each source arrival; no future predictions, tracking or interpolation. Missing-empty AP separately gives skipped source frames no detections. Both score all600 inputs per case.',
        '- Processed-only AP is an additional diagnostic on actual processed inputs. It can expose detector limitations independently of the skipped-empty convention, but its latency-selected subset is not an unbiased full-video benchmark. It is not used to select policies or replace all-arrival metrics.',
        '- Completion age is end-to-source-arrival time for a processed frame. Display age is age of the reused prediction at each source arrival. They are different quantities. The main table shows the mean of case-level p95 completion ages; CSV also retains display ages, service tails, controller overhead and both AP conventions.',
        '- Pooled AP recomputes one COCO curve from all six sequences, preserving their independent IDs. Day/night pooled scores are recomputed separately. Macro means and worst-sequence scores expose unequal pedestrian density; none are official full-dataset benchmark scores.',
        '- J/source frame divides host named-rail energy by600 arrivals; J/completed inference divides by the processed count. Dropping more inputs must not be hidden by an ambiguous energy denominator. Neither metric is isolated model energy or wall-plug power. Background browser/acquisition load remains included.',
        '- The resource intervention changes Torch preprocessing parallelism within the same4-core process affinity. It does not reserve physical cores, throttle GPU clocks or introduce GPU contention. Matched fixed-policy controls are in paired_fixed_controls.csv; the adaptive policies have no constant-thread counterfactual in this study.',
        '- Per-phase source counts are200; phases.csv retains drops, service/age costs and chosen scales. Recovery means the time from the first restored-phase row start until three consecutive completed frames meet100ms service, or null if not reached. Intrinsically infeasible budgets must not be called a controller recovery failure.',
        '- Stable person-presence events are post-hoc content-change proxies, not annotated day/night/weather cuts. Sequences were not joined into fake continuous transitions. Labels were never supplied to a controller.',
        '- One run per case and only three day/three night sequence prefixes limit statistical/generalization claims. Source-content changes, shared-host load and thermal drift constrain interpretation of phase differences. Frozen M3FD→KAIST transfer measures domain shift; no target fitting or favorable-sequence selection occurred.', '',
        '## Reproduction and evidence','',
        'Preparation: experiments/kaist_selection.py, extract_kaist_selection.py and prepare_kaist_video.py. The full archive remains read-only at the verified workstation path; only selected JPEG/XML/metadata files were copied into isolated project directories.', '',
        'Runner: `python -m experiments.run_kaist_video --plan /path/to/plan.json --output /NEW/output`. It checks validated engine and manifest hashes before processing. Source snapshots, raw per-frame costs/predictions, power traces and scoring provenance are retained at '+str(a.raw)+'.', '',
        'New-family gates on the first frame of each new sequence passed24 comparisons per model at all four resolutions, with explicit person-only label mapping. This is backend sanity checking, not accuracy fitting.', '',
        'See validation.json, raw_index.json, cases.csv, phases.csv and paired_fixed_controls.csv for evidence and definitions.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.output/'summary.json').write_text(json.dumps(groups,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=sorted({r['model'] for r in groups});policies=plan['policy_names']
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,constrained_layout=True)
    positions=np.arange(len(policies))
    for ax,model in zip(axes,names):
        selected={r['policy']:r for r in groups if r['model']==model and r['condition']=='thread_reduction'}
        ax.bar(positions-.18,[100*selected[p]['day_pooled_AP'] for p in policies],width=.36,label='Day')
        ax.bar(positions+.18,[100*selected[p]['night_pooled_AP'] for p in policies],width=.36,label='Night')
        ax.set_xticks(positions,['576','sequence','deadline','guarded'],rotation=25)
        ax.set_title(model);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Pooled causal person AP (%)');axes[-1].legend()
    fig.suptitle('Frozen M3FD models on held-out KAIST prefixes; CPU threads 4→1→4')
    fig.savefig(a.output/'day_night_accuracy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,constrained_layout=True)
    palette={p:plt.get_cmap('tab10')(i) for i,p in enumerate(policies)}
    for ax,model in zip(axes,names):
        for r in [x for x in groups if x['model']==model]:
            label=r['policy']+(' (constant)' if r['condition']=='constant_threads' else '')
            ax.scatter(r['mean_case_p95_completion_age_ms'],100*r['pooled_causal_AP'],s=45,
                marker='s' if r['condition']=='constant_threads' else 'o',label=label,
                color=palette[r['policy']],edgecolor='black' if r['condition']=='constant_threads' else 'none')
        ax.set_title(model);ax.set_xlabel('Mean case p95 completion age (ms)');ax.grid(alpha=.25);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Pooled causal person AP (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.1),ncol=3,fontsize=8)
    fig.suptitle('Accuracy and output age; same six independent 30-second prefixes')
    fig.savefig(a.output/'accuracy_age.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Trace examples are fixed by sequence ID, not selected by favorable results.
    examples=[plan['sequences'][0]['sequence'],plan['sequences'][3]['sequence']]
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
    for i,model in enumerate(names):
        for j,sequence in enumerate(examples):
            ax=axes[i,j]
            for policy in policies:
                entry=next(x for x in entries if x['job']==dict(model=model,policy=policy,sequence=sequence,condition='thread_reduction'))
                raw=json.loads((root/entry['result']).read_text());mf=manifests[sequence]
                ax.step([mf['frames'][r['frame_index']]['timestamp_s'] for r in raw['rows']],
                        [r['resolution'] for r in raw['rows']],where='post',label=policy,color=palette[policy],alpha=.8)
            ax.axvspan(10,20,color='grey',alpha=.12);ax.set_title(model+' '+sequence)
            ax.set_yticks([480,544,576,640]);ax.set_ylabel('Long side (pixels)');ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Source time (s)');axes[-1,1].set_xlabel('Source time (s)')
    fig.suptitle('Predeclared first day/night sequence examples; grey = one Torch CPU thread')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.05),ncol=4,fontsize=8)
    fig.savefig(a.output/'resolution_traces.png',dpi=180,bbox_inches='tight');plt.close(fig)
    with (a.output/'REPORT.md').open('a') as report:
        report.write('\n## Figures\n\n![Day/night accuracy](day_night_accuracy.png)\n\n![Accuracy/output age](accuracy_age.png)\n\n![Resolution traces](resolution_traces.png)\n')
    (a.output/'validation.json').write_text(json.dumps(dict(completed_cases=90,unique_pairs=3600,
        input_frame_evaluations=54000,processed_frames=completed['processed_frames'],dropped_frames=completed['dropped_frames'],
        source_accounting_valid=True,power_read_errors=0,person_class_mapping=4,source_fps=20),indent=2))
    index={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in a.raw.rglob('*') if p.is_file()}
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()
