"""Validate frozen RGBT-Tiny transfer outputs and write a compact comparison."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.evaluate_families import sha256


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    completed=json.loads((a.raw/'COMPLETED.json').read_text());plan=json.loads((a.raw/'plan.json').read_text())
    metrics=json.loads((a.raw/'metrics.json').read_text());stats=json.loads((a.raw/'dataset_statistics.json').read_text())
    prefixes=lambda values:{'_'.join(s.split('_')[:2]) for s in values}
    overlap=sorted(prefixes(plan['train_sequences'])&prefixes(plan['test_sequences']))
    gt=json.loads((a.raw/'target_common_ground_truth.json').read_text());ids={x['id'] for x in gt['images']}
    hashes=json.loads((a.raw/'input_hashes.json').read_text())
    if completed['models']!=3 or len(metrics)!=3 or len(ids)!=12590 or len(hashes)!=12590:raise ValueError('Incomplete comparison')
    if {h['image_id'] for h in hashes}!=ids:raise ValueError('Input hash inventory mismatch')
    if {m['model'] for m in metrics}!={m['name'] for m in plan['models']}:raise ValueError('Model matrix mismatch')
    rows=[];index=[]
    for m in metrics:
        name=m['model'];pred=a.raw/(name+'_predictions.json');timing=a.raw/(name+'_timing.json')
        predictions=json.loads(pred.read_text());times=json.loads(timing.read_text())
        if len(times)!=12590 or {t['image_id'] for t in times}!=ids:raise ValueError('Source-frame accounting mismatch')
        if any(x['image_id'] not in ids or x['category_id'] not in [1,2,5] or not np.isfinite(x['bbox']+[x['score']]).all() for x in predictions):raise ValueError('Prediction contract failed')
        index.append(dict(model=name,predictions=str(pred),predictions_sha256=sha256(pred),timing=str(timing),timing_sha256=sha256(timing),detections=len(predictions)))
        for dataset in ['M3FD','RGBT_Tiny']:
            s=m[dataset]
            rows.append(dict(model=name,dataset=dataset,AP=s['AP'],AP50=s['AP50'],AP_small=s['AP_small'],
                AP_bus=s['AP_per_class']['bus'],AP_car=s['AP_per_class']['car'],AP_person=s['AP_per_class']['person']))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for name in ['COMPLETED.json','plan.json','metrics.json','dataset_statistics.json','environment.json','status.json','RESUME.json']:
        if (a.raw/name).exists():shutil.copy2(a.raw/name,a.output/name)
    (a.output/'raw_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps(dict(models=3,target_pairs=12590,target_sequences=30,
        source_pairs=419,shared_classes=['bus','car','person'],all_target_images_accounted=True,
        finite_predictions=True,predictions_hashed=True,trained_on_target=False,
        cached_fasterrcnn_rescored='RESUME.json preserved; original six-class helper failed before target scoring'),indent=2)+'\n')
    (a.output/'split_audit.json').write_text(json.dumps(dict(official_sequence_ids_disjoint=not(set(plan['train_sequences'])&set(plan['test_sequences'])),
        shared_filename_recording_prefixes=overlap,
        interpretation='Filename prefixes suggest possible related captures; they do not prove identical recording identity. Preserve the official split and verify capture groups before any new target-training protocol.'),indent=2)+'\n')
    lines=['# RGBT-Tiny comparison with M3FD','',
        'Three frozen M3FD-trained detectors were evaluated at a 576-pixel input long side on the complete official RGBT-Tiny test split: 12,590 pairs across 30 sequences. The same saved M3FD test predictions were rescored on the shared bus, car and person classes. No RGBT-Tiny training, threshold fitting or checkpoint selection occurred.','',
        '| Model | M3FD shared-class AP % | RGBT-Tiny shared-class AP % | M3FD AP-small % | RGBT-Tiny AP-small % |',
        '|---|---:|---:|---:|---:|']
    for m in metrics:lines.append(f"| {m['model']} | {100*m['M3FD']['AP']:.3f} | {100*m['RGBT_Tiny']['AP']:.3f} | {100*m['M3FD']['AP_small']:.3f} | {100*m['RGBT_Tiny']['AP_small']:.3f} |")
    lines+=['','AP is standard COCO IoU .50:.95 with maxDets 1/10/100 and native annotation areas. It is distinct from the RGBT-Tiny SAFit metric. The source/target AP difference describes these checkpoints under transfer; it does not rank the intrinsic difficulty of two datasets or compare matched within-dataset training.','',
        '## Object scale and class distribution','',
        '| Shared-class test data | Paired images | Boxes | Area below 16² native pixels | Area below 32² native pixels | Mean boxes/image |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset in ['M3FD','RGBT_Tiny']:
        s=stats[dataset];lines.append(f"| {dataset} | {s['images']:,} | {s['boxes']:,} | {100*s['fraction_below_16_squared']:.2f}% | {100*s['fraction_below_32_squared']:.2f}% | {s['mean_boxes_per_image']:.2f} |")
    lines+=['','These are native-coordinate area thresholds. Native image dimensions differ across datasets; resizing changes the actual input-pixel footprint. Class frequencies, viewpoint, environment, calibration and annotation conventions also differ, so object size alone is not established as the cause of the transfer gap.','',
        'RGBT-Tiny has 46,701 paired images: 34,111 training pairs in 85 sequences and 12,590 test pairs in 30 disjoint sequence IDs. Eight DJI filename prefixes occur in both partitions; this suggests potentially related captures but does not prove recording identity. split_audit.json preserves that distinction. This frozen-transfer study never fits on either RGBT-Tiny partition. Any future target-training protocol should verify capture groups before selecting validation/test units. The frequently quoted roughly 93K frames counts the two modality frames. Our comparison uses original visible-modality 00 boxes with paired thermal-modality 01 images, rather than the local YOLO conversion, which clips some source boxes. Residual cross-camera parallax is not independently calibrated. All images remain in scope, including images without shared-class objects. Cyclist is not mapped to motorcycle; unmatched classes are excluded.','',
        '## Per-class accuracy','',
        '| Model | Dataset | Bus AP % | Car AP % | Person AP % |',
        '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['dataset']} | {100*r['AP_bus']:.3f} | {100*r['AP_car']:.3f} | {100*r['AP_person']:.3f} |")
    lines+=['','## Interpretation and reproduction','',
        'This experiment establishes a frozen-transfer baseline for three architectures. A dataset-trained comparison requires separate training and validation within RGBT-Tiny, followed by an untouched evaluation protocol. This fixed576 accuracy study does not establish adaptive-controller performance on RGBT-Tiny or live latency, dropped-frame or energy results. Timings in the raw files are offline workstation service measurements.','',
        'The original frozen plan, checkpoint and annotation hashes are preserved. An older remote evaluator assumed six classes and failed while summarizing the shared-class source score; its source copy is retained in source_attempt1. The corrected evaluator handles noncontiguous category IDs. Completed Faster R-CNN native predictions were reused and rescored; remaining models used the same frozen settings. RESUME.json records source versions; no favorable test setting was chosen.','',
        'Run `python -m experiments.compare_rgbt_tiny` with the arguments in plan.json to reproduce into a new directory. Recreate this compact export with `python -m experiments.report_cross_dataset --raw /path/to/results --output /path/to/summary`. Full predictions, ground truth, image hashes, timings and logs remain outside Git at '+str(a.raw)+'.','',
        'Source: [RGBT-Tiny official release](https://github.com/XinyiYing/RGBT-Tiny), Ying et al., *Visible-Thermal Tiny Object Detection: A Benchmark Dataset and Baselines*, TPAMI 2025. Existing data-access terms apply; a public-domain license is not assumed.','',
        '![Transfer accuracy and object scale](comparison.png)']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    labels=[m['model'] for m in metrics];x=np.arange(3)
    for dataset,offset,color in [('M3FD',-.18,'#477aa6'),('RGBT_Tiny',.18,'#df9940')]:
        vals=[100*m[dataset]['AP'] for m in metrics];axes[0].bar(x+offset,np.maximum(vals,1e-5),width=.36,label=dataset,color=color)
    axes[0].set_xticks(x,labels,rotation=15);axes[0].set_yscale('log');axes[0].set_ylabel('Shared-class COCO AP (%) — logarithmic axis');axes[0].legend();axes[0].set_title('Frozen M3FD weights; fixed576')
    for dataset,offset,color in [('M3FD',-.18,'#477aa6'),('RGBT_Tiny',.18,'#df9940')]:
        s=stats[dataset];axes[1].bar(np.arange(2)+offset,[100*s['fraction_below_16_squared'],100*s['fraction_below_32_squared']],width=.36,label=dataset,color=color)
    axes[1].set_xticks([0,1],['Area < 16²','Area < 32²']);axes[1].set_ylabel('Shared-class native boxes (%)');axes[1].set_ylim(0,105);axes[1].set_title('Native box area; input footprints differ');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(a.output/'comparison.png',dpi=180);plt.close(fig)
    print(a.output/'REPORT.md')


if __name__=='__main__':main()
