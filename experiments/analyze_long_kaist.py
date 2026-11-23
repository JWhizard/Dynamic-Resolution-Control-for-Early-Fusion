"""Analyze frozen KAIST runs without fitting models or selecting new policies.

Accuracy aggregation is explicitly a sequence macro mean of saved exact COCO
scores. Service/age quantiles pool recorded samples; energy and drops use
explicit source/completed denominators. Bootstrap units are paired sequences.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.streaming import align_predictions

POLICIES=['fixed:480','fixed:544','fixed:576','fixed:640','sequence','guarded','deadline','aimd']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def aggregate_cases(cases):
    """Keep sequence-macro AP distinct from source/processed-weighted costs."""
    source=sum(c['source_frames'] for c in cases)
    processed=sum(c['processed_frames'] for c in cases)
    energy=sum(c['named_rail_J'] for c in cases)
    return dict(sequences=len(cases),source_frames=source,processed_frames=processed,
        dropped_frames=source-processed,
        macro_causal_AP=float(np.mean([c['AP_causal'] for c in cases])),
        macro_missing_empty_AP=float(np.mean([c['AP_missing_empty'] for c in cases])),
        macro_causal_AP50=float(np.mean([c['AP50_causal'] for c in cases])),
        macro_causal_AP_small=float(np.mean([c['AP_small_causal'] for c in cases])),
        macro_person_recall50_conf25=float(np.mean([c['person_recall50_conf25'] for c in cases])),
        worst_sequence_causal_AP=min(c['AP_causal'] for c in cases),
        dropped_fraction=(source-processed)/source,
        initial_unavailable_fraction=sum(c['initial_unavailable_frames'] for c in cases)/source,
        mean_service_ms=sum(c['mean_service_ms']*c['processed_frames'] for c in cases)/processed,
        mean_controller_ms=sum(c['mean_controller_ms']*c['processed_frames'] for c in cases)/processed,
        mean_completion_age_ms=sum(c['mean_completion_age_ms']*c['processed_frames'] for c in cases)/processed,
        budget_miss_fraction=sum(c['budget_misses'] for c in cases)/processed,
        J_per_source_frame=energy/source,J_per_completed_inference=energy/processed,
        named_rail_J=energy,switches=sum(c['switches'] for c in cases))


def pair_cases(cases,reference):
    left={c['sequence']:c for c in cases};right={c['sequence']:c for c in reference}
    if len(left)!=len(cases) or len(right)!=len(reference) or left.keys()!=right.keys():
        raise ValueError('Paired comparison requires the same unique sequences')
    return [(left[k],right[k]) for k in sorted(left)]


def paired_bootstrap(cases,reference,repetitions=1000,seed=20261119):
    pairs=pair_cases(cases,reference)
    metrics={
        'AP_change_points':np.array([100*(a['AP_causal']-b['AP_causal']) for a,b in pairs]),
        'drop_change_points':np.array([100*(a['dropped_frames']/a['source_frames']-b['dropped_frames']/b['source_frames']) for a,b in pairs]),
        'service_change_percent':np.array([100*(a['mean_service_ms']/b['mean_service_ms']-1) for a,b in pairs]),
        'energy_per_source_change_percent':np.array([100*((a['named_rail_J']/a['source_frames'])/(b['named_rail_J']/b['source_frames'])-1) for a,b in pairs]),
        'p95_completion_age_change_percent':np.array([100*(a['p95_completion_age_ms']/b['p95_completion_age_ms']-1) for a,b in pairs])}
    indices=np.random.default_rng(seed).integers(0,len(pairs),(repetitions,len(pairs)))
    record={}
    for name,values in metrics.items():
        lo,hi=np.percentile(values[indices].mean(axis=1),[2.5,97.5])
        record[name]=float(values.mean());record[name+'_lo']=float(lo);record[name+'_hi']=float(hi)
    return record


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.raw/'runs'
    plan=json.loads((root/'plan.json').read_text())
    complete=json.loads((root/'COMPLETED.json').read_text())
    entries=json.loads((root/'summary.json').read_text())
    if Counter(tuple(sorted(j.items())) for j in plan['jobs'])!=Counter(tuple(sorted(e['job'].items())) for e in entries):
        raise ValueError('Job matrix mismatch')
    manifests={};inventory={s['sequence']:s for s in plan['sequences']};annotation_counts={}
    source_roots=[args.raw/'source',args.raw.parent/'kaist_video_20261116/source']
    source_proofs={sha(f):str(f) for directory in reversed(source_roots) for f in directory.rglob('*') if f.is_file()}
    if sha(plan['controller'])!=plan['controller_sha256']:raise ValueError('Frozen controller configuration changed')
    source_evidence={}
    for sequence,s in inventory.items():
        p=Path(s['manifest']);m=json.loads(p.read_text())
        if sha(p)!=s['manifest_sha256'] or sha(p.parent/'annotations.json')!=s['annotations_sha256']:
            raise ValueError('Manifest or annotation file changed')
        manifests[sequence]=m
        gt=json.loads((p.parent/'annotations.json').read_text())
        annotation_counts[sequence]=sum(not a['iscrowd'] for a in gt['annotations'])
    engines={}
    for model in plan['models']:
        config=json.loads(Path(model['engine']).read_text())
        gate=json.loads(Path(model['validation']).read_text())
        gate_hash=gate.get('engine_sha256',gate.get('engine',{}).get('engine_sha256'))
        if not gate['passed'] or gate_hash!=config['engine_sha256'] or sha(Path(model['engine']).parent/config['engine'])!=config['engine_sha256']:
            raise ValueError('Model engine/gate mismatch')
        engines[model['name']]=(sha(model['engine']),config['engine_sha256'])
    records=[];groups=defaultdict(list);samples=defaultdict(lambda:defaultdict(list));phases=[];index=[]
    for number,entry in enumerate(entries,1):
        job=entry['job'];p=root/entry['result'];raw=json.loads(p.read_text())
        score_file=root/entry['scores_file'];scored=json.loads(score_file.read_text())
        s=inventory[job['sequence']];frames=manifests[job['sequence']]['frames'];rows=raw['rows']
        result_hash=sha(p)
        if result_hash!=scored['provenance']['result_sha256'] or scored!=entry['accuracy']:
            raise ValueError('Score/result mismatch')
        if raw['manifest_sha256']!=s['manifest_sha256'] or scored['provenance']['annotations_sha256']!=s['annotations_sha256']:
            raise ValueError('Result/annotation/manifest mismatch')
        if scored['provenance']['scorer_sha256']!=sha(args.raw/'source/experiments/score.py'):
            raise ValueError('Scorer differs from archived source')
        if not all(h in source_proofs for h in raw['source_sha256'].values()):raise ValueError('Archived source mismatch')
        for name,h in raw['source_sha256'].items():source_evidence[name]=dict(sha256=h,archived_copy=source_proofs[h])
        if (raw['weights_sha256'],raw['detector_metadata']['engine_sha256'])!=engines[job['model']]:raise ValueError('Measured engine mismatch')
        if raw['input_frames']!=len(frames) or raw['processed_frames']!=len(rows) or raw['processed_frames']+raw['dropped_frames']!=len(frames):
            raise ValueError('Frame accounting mismatch')
        indices=[r['frame_index'] for r in rows]
        if indices!=sorted(set(indices)) or any(r['image_id']!=frames[r['frame_index']]['id'] for r in rows):
            raise ValueError('Source order/ID mismatch')
        expected=[int(x) for x in plan['conditions'][job['condition']].split(',')]
        if any(r['threads']!=expected[r['phase']] for r in rows):raise ValueError('Thread intervention mismatch')
        if raw['power_errors'] or raw['summary']['energy']['named_rail_sum_joules'] is None:raise ValueError('Invalid power data')
        aligned=align_predictions(frames,rows);by_id={f['id']:f for f in frames}
        ages=[r['output_age_ms'] for r in rows]
        display_ages=[1000*(by_id[r['image_id']]['timestamp_s']-frames[r['source_frame_index']]['timestamp_s']) for r in aligned]
        if any(x<0 for x in ages+display_ages):raise ValueError('Noncausal age')
        service=[r['service_ms'] for r in rows]
        causal=scored['causal_time_aligned']
        record=dict(**job,environment=s['environment'],source_frames=len(frames),processed_frames=len(rows),
            dropped_frames=raw['dropped_frames'],initial_unavailable_frames=len(frames)-len(aligned),
            person_boxes=annotation_counts[job['sequence']],AP_causal=causal['AP'],AP50_causal=causal['AP50'],
            AP_small_causal=causal['AP_small'],person_recall50_conf25=causal['person_recall_iou50_conf25'],
            AP_missing_empty=scored['overall']['AP'],mean_service_ms=float(np.mean(service)),
            p95_service_ms=float(np.percentile(service,95)),p99_service_ms=float(np.percentile(service,99)),
            mean_controller_ms=raw['summary']['controller_ms']['mean'],
            mean_completion_age_ms=float(np.mean(ages)),p95_completion_age_ms=float(np.percentile(ages,95)),
            mean_display_age_ms=float(np.mean(display_ages)),p95_display_age_ms=float(np.percentile(display_ages,95)),
            budget_misses=sum(r['service_ms']>plan['budget_ms'] for r in rows),
            named_rail_J=raw['summary']['energy']['named_rail_sum_joules'],
            mean_resolution=float(np.mean([r['resolution'] for r in rows])),switches=sum(r['switched'] for r in rows))
        if min(record[k] for k in ['AP_causal','AP_small_causal','person_recall50_conf25'])<0:
            raise ValueError('Undefined accuracy metric; explicitly handle missing groups')
        records.append(record)
        for environment in ['all',s['environment']]:
            key=(job['model'],job['policy'],job['condition'],environment)
            groups[key].append(record)
            for name,values in [('service',service),('completion_age',ages),('display_age',display_ages)]:samples[key][name].extend(values)
        # The runtime assigns source phases by floor(index*3/N).
        for phase in range(3):
            part=[r for r in rows if r['phase']==phase]
            source_count=sum(min(2,i*3//len(frames))==phase for i in range(len(frames)))
            phases.append(dict(**job,environment=s['environment'],phase=phase,threads=expected[phase],
                source_frames=source_count,processed_frames=len(part),dropped_frames=source_count-len(part),
                mean_service_ms=float(np.mean([r['service_ms'] for r in part])),
                mean_completion_age_ms=float(np.mean([r['output_age_ms'] for r in part])),
                mean_resolution=float(np.mean([r['resolution'] for r in part]))))
        index.append(dict(result=str(p),sha256=result_hash,scores=str(score_file),scores_sha256=sha(score_file)))
        if number%6==0:
            (args.output/'progress.json').write_text(json.dumps(dict(verified_cases=number,total=len(entries))))
            print(f'Verified {number}/{len(entries)}',flush=True)
    if complete['jobs']!=len(records) or sum(r['source_frames'] for r in records)!=complete['input_frame_evaluations']:
        raise ValueError('Completion mismatch')
    summary=[]
    for key,cases in sorted(groups.items()):
        model,policy,condition,environment=key
        record=dict(model=model,policy=policy,condition=condition,environment=environment,**aggregate_cases(cases))
        for metric,values in samples[key].items():
            for percentile in [50,95,99]:record[f'pooled_p{percentile}_{metric}_ms']=float(np.percentile(values,percentile))
        summary.append(record)
    paired=[];controls=[]
    for model in sorted({r['model'] for r in records}):
        reference=groups[(model,'fixed:576','thread_reduction','all')]
        for policy in POLICIES:
            if policy.startswith('fixed:'):continue
            cases=groups[(model,policy,'thread_reduction','all')]
            paired.append(dict(model=model,policy=policy,reference='fixed:576',sequences=len(cases),
                **paired_bootstrap(cases,reference)))
        changed=reference;constant=groups[(model,'fixed:576','constant_threads','all')]
        controls.append(dict(model=model,comparison='thread_reduction_minus_constant',sequences=len(changed),
            **paired_bootstrap(changed,constant)))
    frontier=[]
    for model in sorted({r['model'] for r in records}):
        cells=[r for r in summary if r['model']==model and r['condition']=='thread_reduction' and r['environment']=='all']
        fixed=[r for r in cells if r['policy'].startswith('fixed:')]
        for cell in cells:
            dominators=[f['policy'] for f in fixed if f['macro_causal_AP']>=cell['macro_causal_AP'] and
                f['mean_service_ms']<=cell['mean_service_ms'] and f['J_per_source_frame']<=cell['J_per_source_frame'] and
                (f['macro_causal_AP']>cell['macro_causal_AP'] or f['mean_service_ms']<cell['mean_service_ms'] or f['J_per_source_frame']<cell['J_per_source_frame'])]
            frontier.append(dict(model=model,policy=cell['policy'],dominated_by_observed_fixed=';'.join(dominators),
                macro_causal_AP=cell['macro_causal_AP'],mean_service_ms=cell['mean_service_ms'],J_per_source_frame=cell['J_per_source_frame']))
    for name,data in [('cases',records),('summary',summary),('paired_vs_fixed576',paired),('paired_fixed_controls',controls),('observed_fixed_frontier',frontier),('phases',phases)]:
        write_csv(args.output/(name+'.csv'),data)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (args.output/'source_evidence.json').write_text(json.dumps(source_evidence,indent=2)+'\n')
    validation=dict(created_utc=datetime.now(timezone.utc).isoformat(),verified_cases=len(records),
        unique_pairs=sum(len(m['frames']) for m in manifests.values()),**complete,
        processed_counts_match=sum(r['processed_frames'] for r in records)==complete['processed_frames'],
        manifest_annotation_engine_source_scorer_hashes_match=True,power_read_errors=0,
        bootstrap_repetitions=1000,bootstrap_seed=20261119,bootstrap_unit='paired sequence (six selected sequences)',
        accuracy_aggregation='Equal-weight sequence macro means of saved exact pycocotools scores; not pooled COCO AP',
        post_hoc_analysis=True,model_or_policy_tuning=False)
    if not validation['processed_counts_match']:raise ValueError('Processed completion mismatch')
    (args.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    make_figures(args.output,summary)
    make_report(args.output,summary,paired,controls,frontier,validation)
    print('Analysis complete:',args.output,flush=True)


def make_figures(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=['yolo26n_ms','fasterrcnn','retinanet']
    names={'yolo26n_ms':'YOLO26n','fasterrcnn':'Faster R-CNN','retinanet':'RetinaNet'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,model in zip(axes,models):
        x=np.arange(len(POLICIES))
        for environment,offset,color in [('day',-.19,'#e0a12b'),('night',.19,'#326aab')]:
            lookup={r['policy']:r for r in summary if r['model']==model and r['condition']=='thread_reduction' and r['environment']==environment}
            ax.bar(x+offset,[100*lookup[p]['macro_causal_AP'] for p in POLICIES],width=.38,label=environment,color=color)
        ax.set_xticks(x,[p.replace('fixed:','') for p in POLICIES],rotation=45,ha='right')
        ax.set_title(names[model]);ax.set_ylabel('Sequence-macro causal person AP (%)');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('KAIST frozen transfer: three day and three night sequences (one run per case)')
    fig.savefig(output/'day_night_accuracy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for column,model in enumerate(models):
        cells=[r for r in summary if r['model']==model and r['condition']=='thread_reduction' and r['environment']=='all']
        for row,(metric,label) in enumerate([('mean_service_ms','Mean service time (ms)'),('J_per_source_frame','Named-rail energy / source arrival (J)')]):
            ax=axes[row,column]
            for r in cells:
                fixed=r['policy'].startswith('fixed:');color=plt.get_cmap('tab10')(POLICIES.index(r['policy']))
                ax.scatter(r[metric],100*r['macro_causal_AP'],marker='s' if fixed else 'o',s=45,color=color)
                offset=(4,4)
                if model=='yolo26n_ms':
                    offsets=({'sequence':(-57,-13),'guarded':(10,-11),'fixed:640':(10,16),'aimd':(10,-2)}
                             if row==0 else {'sequence':(-4,24),'guarded':(-55,-10),'fixed:640':(-2,6)})
                    offset=offsets.get(r['policy'],offset)
                ax.annotate(r['policy'].replace('fixed:',''),(r[metric],100*r['macro_causal_AP']),
                    xytext=offset,textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',lw=.6,color=color) if offset!=(4,4) else None)
            ax.set_title(names[model]);ax.set_xlabel(label);ax.set_ylabel('Sequence-macro causal AP (%)');ax.grid(alpha=.2)
    fig.suptitle('Observed accuracy / cost tradeoffs; squares = fixed, circles = adaptive')
    fig.savefig(output/'accuracy_cost.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for column,model in enumerate(models):
        lookup={r['policy']:r for r in summary if r['model']==model and r['condition']=='thread_reduction' and r['environment']=='all'}
        for row,(metric,label,scale) in enumerate([('dropped_fraction','Dropped source arrivals (%)',100),('pooled_p95_completion_age_ms','Pooled p95 completion age (ms)',1)]):
            ax=axes[row,column];ax.bar(np.arange(8),[scale*lookup[p][metric] for p in POLICIES],color=['#547a9e']*4+['#309b83']*4)
            ax.set_xticks(np.arange(8),[p.replace('fixed:','') for p in POLICIES],rotation=45,ha='right')
            ax.set_title(names[model]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('20 Hz latest-due-input replay; source period 50 ms, service budget 100 ms')
    fig.savefig(output/'drops_output_age.png',dpi=180);plt.close(fig)


def make_report(output,summary,paired,controls,frontier,validation):
    primary=[r for r in summary if r['condition']=='thread_reduction' and r['environment']=='all']
    lines=['# KAIST full-video analysis — November 19, 2026','',
        'All 162 cases completed and their saved evidence was verified. The study uses three frozen M3FD-trained models on six full KAIST sequences (three day, three night), 16,873 unique paired images, and 455,571 source-frame evaluations. This analysis is post-hoc; no model, threshold or controller was fitted to these results.','',
        '## Findings','']
    for model in ['yolo26n_ms','fasterrcnn','retinanet']:
        cells=[r for r in primary if r['model']==model];best=max((r for r in cells if r['policy'].startswith('fixed:')),key=lambda r:r['macro_causal_AP'])
        seq=next(r for r in cells if r['policy']=='sequence');guard=next(r for r in cells if r['policy']=='guarded')
        lines.append(f"- **{model}:** highest observed fixed-resolution macro AP is {100*best['macro_causal_AP']:.3f}% at {best['policy']}; frozen sequence / guarded AP is {100*seq['macro_causal_AP']:.3f}% / {100*guard['macro_causal_AP']:.3f}%. Sequence / guarded source drops are {100*seq['dropped_fraction']:.2f}% / {100*guard['dropped_fraction']:.2f}%.")
    lines+=['','The data do not show a consistent adaptive advantage across detector families. Absolute person AP is low under this frozen cross-dataset transfer. The best observed fixed setting is a descriptive test result, not a validation-selected policy or a newly recommended deployment setting.','',
        '## Fixed and adaptive policies','',
        'AP is the equal-weight mean of six per-sequence **causal** COCO AP scores, in percent; it is not a pooled COCO score. Day/night AP averages three sequences each. Service and controller means weight completed inferences. Drops weight source arrivals. P95 values below pool the actual samples across all six sequences, rather than averaging six percentiles.','',
        '| Model | Policy | AP % | Day AP % | Night AP % | Worst AP % | Mean service ms | p95 service ms | p95 completion age ms | Drops % | J/source | J/completed |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        day=next(s for s in summary if (s['model'],s['policy'],s['condition'],s['environment'])==(r['model'],r['policy'],r['condition'],'day'))
        night=next(s for s in summary if (s['model'],s['policy'],s['condition'],s['environment'])==(r['model'],r['policy'],r['condition'],'night'))
        lines.append(f"| {r['model']} | {r['policy']} | {100*r['macro_causal_AP']:.3f} | {100*day['macro_causal_AP']:.3f} | {100*night['macro_causal_AP']:.3f} | {100*r['worst_sequence_causal_AP']:.3f} | {r['mean_service_ms']:.2f} | {r['pooled_p95_service_ms']:.2f} | {r['pooled_p95_completion_age_ms']:.2f} | {100*r['dropped_fraction']:.2f} | {r['J_per_source_frame']:.3f} | {r['J_per_completed_inference']:.3f} |")
    lines+=['','![Day/night accuracy](day_night_accuracy.png)','',
        '## Paired comparisons against fixed576','',
        'Each comparison pairs the same six sequences under the same 4→1→4 thread intervention. Brackets give the 2.5th–97.5th percentiles of 1,000 paired-sequence bootstrap resamples (seed 20261119). Percentage cost changes are means of six within-sequence relative changes; they differ from ratios of pooled costs above. AP and drop changes are percentage points. Negative cost/drop changes are reductions.','',
        '| Model | Policy | AP change, points [95% interval] | Mean service change % [95% interval] | Energy/source change % [95% interval] | Drop change, points [95% interval] |',
        '|---|---|---:|---:|---:|---:|']
    for p in paired:
        formatted=[f"{p[k]:+.3f} [{p[k+'_lo']:+.3f}, {p[k+'_hi']:+.3f}]" for k in ['AP_change_points','service_change_percent','energy_per_source_change_percent','drop_change_points']]
        lines.append('| '+p['model']+' | '+p['policy']+' | '+' | '.join(formatted)+' |')
    lines+=['','Only six selected sequence units and one run per case are available. These intervals describe sensitivity to the sampled sequence mix; they do not estimate repeat-run variability or justify broad environmental/generalization or significance claims. Many comparisons are exploratory and have no multiplicity adjustment.','',
        '## Fixed-resolution frontier','',
        'A fixed point dominates a measured policy only if its macro causal AP is at least as high, mean service time and J/source are both no higher, and at least one is strictly better. This is an observed three-objective comparison, not a confirmatory frontier or interpolation between fixed sizes.','']
    for model in ['yolo26n_ms','fasterrcnn','retinanet']:
        found=[f"{r['policy']} (by {r['dominated_by_observed_fixed']})" for r in frontier if r['model']==model and not r['policy'].startswith('fixed:') and r['dominated_by_observed_fixed']]
        lines.append('- '+model+': adaptive points dominated by an observed fixed point: '+(', '.join(found) if found else 'none under all three measured objectives')+'.')
    lines+=['','![Accuracy and costs](accuracy_cost.png)','',
        '## Streaming, energy and resource controls','',
        '- Skipped-empty AP assigns no boxes to skipped source inputs. Causal AP reuses only the latest completed boxes at each arrival, without future predictions, tracking or interpolation. Every source arrival is included in scoring. Both measures are retained in the CSVs; neither is official KAIST log-average miss rate.',
        '- Completion age is completion time minus the processed frame’s arrival. Display age is source-time age of the reused boxes at an arrival. Initial arrivals with no available output are counted separately; display-age quantiles condition on an available output. See summary.csv for pooled p50/p95/p99 of each.',
        '- The 50 ms source period and 100 ms service budget differ. A service under 100 ms can still drop frames at 20 Hz. Tail measurements include input reading/decoding, preprocessing, inference, policy updates and harness overhead; they are not isolated GPU kernel timings.',
        '- Energy is integrated named-rail host energy, including background browser/camera activity. Both J/source arrival and J/completed inference are reported: processing fewer inputs must not be presented as model-only energy efficiency.',
        '- Fixed576 has matched constant-thread controls. The primary intervention changes Torch CPU preprocessing threads 4→1→4 in sequence thirds within the same affinity; it does not impose GPU contention or a physical-core quota. Phase differences also coincide with changing scene content.','',
        '| Model | Thread change minus constant: AP points | Service change % | Drop change points | Energy/source change % |',
        '|---|---:|---:|---:|---:|']
    for c in controls:lines.append(f"| {c['model']} | {c['AP_change_points']:+.3f} | {c['service_change_percent']:+.2f} | {c['drop_change_points']:+.2f} | {c['energy_per_source_change_percent']:+.2f} |")
    lines+=['','Control bootstrap intervals are retained in paired_fixed_controls.csv. Adaptive methods have no corresponding constant-thread control in this study. Resource phases are retained in phases.csv; no scene-transition causality or independent recovery generalization is claimed.','',
        '![Drops and completion age](drops_output_age.png)','',
        '## Limits and reproducibility','',
        '- These full sequences extend previously scored 600-frame prefixes. The frozen-transfer protocol and case matrix preceded the long runs, but this analysis and its observed frontier are post-hoc. No target fitting occurred.',
        '- All models score only mapped M3FD People → KAIST person. Crowd, ambiguous, cyclist and difficult regions follow the preserved crowd-ignore convention. Provider alignment is accepted; pixel calibration and hardware synchronization are not independently certified.',
        '- Day and night recordings remain separate sequences. They are not continuous day-to-night transitions, new weather/location coverage or evidence of real-camera accuracy. No new training or inference was performed for this analysis.',
        '- Per-sequence exact scores come from the archived pycocotools evaluation. AP-small and person recall at IoU .5/confidence .25 are retained in cases.csv and summary.csv. Sequence macro averaging prevents longer clips from silently receiving more accuracy weight; it is not pooled COCO AP.',
        '- Raw traces are preserved outside the repository. raw_index.json links all verified result/score files by SHA-256. validation.json records completion, archived source/scorer, engine, manifest and annotation checks. The long-run source directory omitted controller_sequence.py; its recorded hash matches the pre-existing November 16 archived copy exactly. source_evidence.json records the actual archived path for every checked inference source. No source was reconstructed or silently replaced.',
        '', 'Reproduce into a new directory:', '', '```bash',
        '.venv/bin/python -m experiments.analyze_long_kaist \\',
        '  --raw /media/imsel/4TBSSD/EMCH585_Artifacts/kaist_long_video_20261118 \\',
        '  --output /NEW/analysis', '```', '',
        'Files: cases.csv (162 cases), summary.csv (all/day/night cells), paired_vs_fixed576.csv, paired_fixed_controls.csv, observed_fixed_frontier.csv, phases.csv, raw_index.json and validation.json.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
