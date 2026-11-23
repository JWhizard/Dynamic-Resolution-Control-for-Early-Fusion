"""Build updated repository review drafts from saved measured results."""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    analysis=repo/'results/processed/kaist_long_video_20261118/analysis_20261119'
    summary=json.loads((analysis/'summary.json').read_text())
    metrics=json.loads((a.comparison/'metrics.json').read_text()) if (a.comparison/'metrics.json').exists() else []
    complete=(a.comparison/'COMPLETED.json').exists()
    platform_path=repo/'results/processed/platform_comparison_20261123/summary.json'
    platform_metrics=json.loads(platform_path.read_text()) if platform_path.exists() else []
    names={'yolo26n_ms':'YOLO26n','fasterrcnn':'Faster R-CNN','retinanet':'RetinaNet'}
    def cell(model,policy):
        return next(r for r in summary if (r['model'],r['policy'],r['condition'],r['environment'])==(model,policy,'thread_reduction','all'))
    lines=['# Dynamic resolution control for RGB thermal detection','',
        'Review draft updated November 23, 2026. Instructor scope, submission requirements and final claims remain to be agreed.','',
        '## Research question and contribution','',
        'Can causal input-resolution control improve the accuracy, timeliness and energy cost of RGB–thermal detection under changing content and compute availability? The implemented contribution is a reproducible evaluation system with common detector adapters, native-coordinate scoring, explicit streaming arrivals, resource interventions, controller accounting and recorded power. Current measurements show useful reductions relative to some fixed settings, but no consistent adaptive-policy advantage across all three detector families and the observed fixed-resolution alternatives.','',
        '## Models and data','',
        'All fusion detectors consume B,G,R,IR. YOLO26n uses the existing multi-scale project checkpoint; Faster R-CNN and RetinaNet use frozen validation-best checkpoints from provisional two-hour training runs. Training budgets differ, so these are checkpoint comparisons rather than a matched-budget architecture study.','',
        '| Dataset | Paired images used | Purpose | Split and scope |',
        '|---|---:|---|---|',
        '| M3FD | 4,200 total; 419 test | Source training and held-out accuracy | Sequence-grouped 2,940/841/419 train/validation/test |',
        '| KAIST | 16,873 | Frozen video transfer across three day and three night sequences | Full versions of six selected official test sequences; earlier prefixes were already examined |',
        '| RGBT-Tiny | 46,701 total; 12,590 test | Frozen transfer and tiny-object comparison | Official 85 training / 30 test sequences; no target fitting |',
        '| Local FLIR | 120 recorded indoor pairs | Deployment cost, repetition and switching | Unannotated; repeated frames do not establish accuracy or independent scene coverage |','',
        'RGBT-Tiny stores two modality frames per pair; the local release has 34,111 training pairs and 12,590 test pairs. Its native release includes modality-specific boxes. The comparison uses original visible-coordinate annotations with paired thermal inputs. Dataset counts and official split membership are audited from the local release. Eight DJI filename prefixes occur in both official partitions; this suggests possibly related captures, not proven frame leakage. Future target training should verify capture identity before selecting validation/test units.','',
        '## Deployment and measurements','',
        'Batch size is one. Jetson AGX Orin uses TensorRT 10.3 strict FP32 GPU engines and retains CPU preprocessing. The workstation uses an RTX 6000 Ada and native FP32 PyTorch. Both additional-family TensorRT exports passed sampled prediction agreement gates before timing. Backends and preprocessing contracts are recorded; cost differences are not attributed solely to hardware.','',
        'Accuracy uses pycocotools COCO AP over IoU .50:.95, AP-small and person recall at IoU .5/confidence .25. The streaming replay runs nominal 20 Hz arrivals, retains the latest due input and counts skipped arrivals. Causal scoring uses only the latest completed boxes at each arrival, without tracking or future results. Service time, completion age and display age are separate metrics. The 100 ms service budget differs from the 50 ms arrival period.','',
        'Named-rail energy includes shared-host browser/camera load. Both joules per source arrival and joules per completed inference are reported; neither is wall-plug or isolated model energy. Model loading, profiling, warmup and common engine builds are recorded separately.','',
        '## M3FD held-out accuracy','',
        '| Frozen checkpoint | Six-class AP % | AP50 % | AP-small % |',
        '|---|---:|---:|---:|',
        '| YOLO26n | 41.21 | 63.58 | 14.58 |',
        '| Faster R-CNN | 40.16 | 65.18 | 18.38 |',
        '| RetinaNet | 30.37 | 49.02 | 4.06 |','',
        'All models were evaluated on the same 419 grouped test pairs at a 576-pixel long side. These six-class scores must not be directly subtracted from person-only KAIST scores or the three-class transfer scores below.','',
        '## KAIST fixed and adaptive comparisons','',
        'All 162 cases completed: three models × six sequences × eight policies plus a fixed576 constant-thread control. The 16,873 unique pairs generate 455,571 source-arrival evaluations, of which 280,026 were processed and 175,545 skipped. The primary CPU preprocessing-thread trace is 4→1→4, within fixed process affinity; this is not a GPU-contention experiment.','',
        '| Model | Policy | Sequence-macro causal person AP % | Mean service ms | Dropped arrivals % | J/source arrival |',
        '|---|---|---:|---:|---:|---:|']
    for model in names:
        for policy in ['fixed:480','fixed:576','sequence','guarded']:
            r=cell(model,policy)
            lines.append(f"| {names[model]} | {policy} | {100*r['macro_causal_AP']:.3f} | {r['mean_service_ms']:.2f} | {100*r['dropped_fraction']:.2f} | {r['J_per_source_frame']:.3f} |")
    lines+=['','Daytime sequence-macro AP exceeds nighttime AP in every tested model/policy combination. YOLO fixed480 has the highest observed person AP (2.742%); its frozen sequence controller reaches 1.558%. Faster R-CNN and RetinaNet have weak absolute transfer accuracy and substantial source drops. Guarded RetinaNet reduces cost relative to fixed576, but the frozen sequence policy has no consistent advantage over the observed fixed alternatives.','',
        'Accuracy aggregates equally across sequences; latency/age percentiles pool actual samples. Paired differences use 1,000 bootstrap resamples of the six matched sequence units. With one run per case and six selected sequences, those intervals describe sequence-mix sensitivity, not repeated-run variability or broad environmental robustness.','',
        '[Detailed KAIST analysis and figures](../../results/processed/kaist_long_video_20261118/analysis_20261119/REPORT.md).','',
        '## RGBT-Tiny comparison with M3FD','',
        'The same frozen M3FD-trained checkpoints are evaluated at fixed576 on the full RGBT-Tiny test split. Both datasets are scored on bus, car and person only; original M3FD predictions are rescored to that subset. Cyclist is not treated as motorcycle. Ship, drone, plane and other unmatched classes are outside this comparison, while all test images remain included as positives or negatives.','']
    if metrics:
        lines+=['| Model | M3FD shared-class AP % | RGBT-Tiny shared-class AP % | M3FD AP-small % | RGBT-Tiny AP-small % |',
            '|---|---:|---:|---:|---:|']
        for m in metrics:lines.append(f"| {names[m['model']]} | {100*m['M3FD']['AP']:.3f} | {100*m['RGBT_Tiny']['AP']:.3f} | {100*m['M3FD']['AP_small']:.3f} | {100*m['RGBT_Tiny']['AP_small']:.3f} |")
    lines += ['','Comparison status: '+('all three models complete.' if complete else f'{len(metrics)} of three model evaluations complete; remaining results are pending.'),
        '', 'In the shared-class test annotations, 80.26% of RGBT-Tiny boxes have area below 16² native pixels, versus 8.25% in M3FD. Native dimensions and resized input footprints differ. Object size, viewpoint, environment and annotation conventions all change, so the size difference alone is not established as the cause of the transfer gap. [Detailed class metrics and scale audit](../../results/processed/rgbt_tiny_comparison_20261123/REPORT.md).',
        '','These are frozen cross-dataset transfer results, not evidence that one dataset is intrinsically harder or a comparison of models trained on each dataset. COCO IoU AP is distinct from RGBT-Tiny’s SAFit metric. Residual cross-camera parallax, object scale and annotation conventions can affect transfer. Test results are not used to choose a new policy.','',
        '## Matched input deployment costs','',
        'A separate fixed576 comparison uses the same 120 predecoded indoor pairs, three fresh processes per model/platform, and four pinned CPU threads. Both input hashes and ordering match. These are short deployment-stack comparisons: Orin TensorRT and RTX 6000 Ada PyTorch differ in backend as well as hardware.','']
    if platform_metrics:
        lines+=['| Platform | Model | Mean service ms | Mean run p95 ms | J/frame | Energy scope |','|---|---|---:|---:|---:|---|']
        for r in platform_metrics:lines.append(f"| {r['platform']} | {names[r['model']]} | {r['mean_service_ms']:.2f} | {r['mean_run_p95_ms']:.2f} | {r['mean_J_per_frame']:.3f} | {r['energy_scope']} |")
    lines+=['','Workstation NVML measures GPU-device energy; Orin rails include host/background load. These scopes do not support a cross-platform energy-efficiency ratio. Prefetch, loading and warmup are excluded, so the table is not end-to-end camera latency. The 120-frame repetitions do not establish stable p99 or unseen-scene behavior. [Protocol and validation](../../results/processed/platform_comparison_20261123/REPORT.md).','',
        '## Related adaptive systems','',
        '[ApproxDet](https://arxiv.org/abs/2010.10754) and [Chanakya](https://www.microsoft.com/en-us/research/publication/chanakya-learning-runtime-decisions-for-adaptive-real-time-perception/) already study content/resource-aware inference. Local deadline, AIMD and guarded controls are diagnostic implementations, not reproductions of those published systems. The preserved upstream environments require older dependencies and original model/data assets; faithful execution and a documented multimodal adaptation remain open.','',
        '## Remaining evaluation and paper direction','',
        '- Measure longer live-camera sessions with verified synchronization, controlled changes and human-checked labels for accuracy claims.',
        '- Extend the short matched-input platform measurements to independent scenes and longer runs; cross-check power with an external meter where available.',
        '- Reproduce at least one published adaptive baseline under declared comparable capabilities and full overhead accounting.',
        '- Investigate why feedback fails under transfer before designing reliability-aware probing or fallback control. Confirm any new method on untouched data.',
        '- Agree on instructor scope, submission requirements and success margins. The evidence supports an empirical system study; it does not yet establish a new state-of-the-art controller.','',
        '## Evidence and reproduction','',
        '- [Project checklist](../EMCH585_PROJECT_TODO.txt)',
        '- [GPU export and numerical validation](../FAMILY_TENSORRT.md)',
        '- [KAIST video protocol](../KAIST_VIDEO_PROTOCOL.md)',
        '- [M3FD held-out report](../../results/processed/heldout_families_20261116/REPORT.md)',
        '- [RGBT-Tiny official release](https://github.com/XinyiYing/RGBT-Tiny) and [paper](https://arxiv.org/abs/2406.14482). Existing data-access terms remain applicable; the repository does not establish a public-domain dataset license.',
        '','The dated November 13 drafts remain preserved. This review draft incorporates the November 16 training/deployment and November 19 full-video analysis.']
    report=a.output/'EMCH585_REPORT_REVIEW_20261123.md';report.write_text('\n'.join(lines)+'\n')
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    def text(s,x,y,w,h,value,size=23,color=(230,236,246)):
        tf=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame;tf.word_wrap=True
        for i,line in enumerate(value.split('\n')):
            q=tf.paragraphs[0] if i==0 else tf.add_paragraph();q.text=line;q.font.size=Pt(size);q.font.color.rgb=RGBColor(*color);q.space_after=Pt(14)
    def slide(title,body=None):
        s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor(12,20,34)
        text(s,.55,.35,12.2,.8,title,30,(86,224,204))
        if body:text(s,.65,1.5,12,5.1,'\n'.join(body))
        text(s,.65,7.02,12,.3,'EMCH585 | November 23, 2026 | REVIEW DRAFT',11,(156,172,194));return s
    slide('Dynamic resolution control for RGB–thermal detection',[
        'Question: When does changing input resolution improve useful detection per unit cost?',
        'Three detector families, source accuracy and frozen video transfer.',
        'Current result: useful tradeoffs, but no consistent adaptive advantage.'])
    slide('Models and data',[
        'YOLO26n, Faster R-CNN and RetinaNet: four-channel B,G,R,IR inputs.',
        'M3FD: 4,200 pairs; grouped train / validation / test = 2,940 / 841 / 419.',
        'KAIST: six full day/night test sequences, 16,873 paired frames.',
        'RGBT-Tiny: full official test, 12,590 pairs across 30 sequences; frozen transfer.'])
    slide('Platform and metrics',[
        'Batch one; Jetson AGX Orin TensorRT 10.3 FP32 and RTX 6000 Ada PyTorch.',
        'Accuracy: COCO AP, AP-small and person recall; explicit shared-class mapping.',
        'Streaming: mean/p95 latency, completion age, display age and dropped arrivals.',
        'Energy: named Jetson rails per source arrival and per completed inference.'])
    slide('Completed experiments',[
        'Three frozen checkpoints scored on 419 grouped M3FD test pairs.',
        'Jetson deployment: 72 short GPU cases, three repetitions per model/policy.',
        'KAIST: 162 long-video cases, 455,571 source-frame evaluations.',
        'All long-video cases verified by hashes, source accounting and power-read checks.'])
    slide('M3FD source accuracy at fixed576',[
        'YOLO26n: AP 41.21%; AP-small 14.58%.',
        'Faster R-CNN: AP 40.16%; AP-small 18.38%.',
        'RetinaNet: AP 30.37%; AP-small 4.06%.',
        'Six classes; checkpoint recipes and training budgets differ.'])
    slide('KAIST person accuracy and streaming drops',[
        *[f"{names[m]} sequence policy: AP {100*cell(m,'sequence')['macro_causal_AP']:.3f}%; drops {100*cell(m,'sequence')['dropped_fraction']:.2f}%." for m in names],
        'AP is averaged equally across six sequences; all models remain frozen.',
        '20 Hz arrivals have a 50 ms period; the service budget is 100 ms.'])
    s=slide('Day and night transfer')
    s.shapes.add_picture(str(analysis/'day_night_accuracy.png'),Inches(.55),Inches(1.55),width=Inches(12.2))
    text(s,.7,5.6,12,1.1,'Daytime AP exceeds nighttime AP across the tested policies.\nAbsolute transfer accuracy is weak; separate recordings do not establish natural transitions.',20)
    slide('What the controller comparison shows',[
        'YOLO fixed480: highest observed macro person AP, 2.742%; sequence: 1.558%.',
        'Guarded RetinaNet lowers service cost, drops and energy versus fixed576.',
        'Observed fixed settings dominate several adaptive points across accuracy and costs.',
        'A best observed test setting is descriptive; it is not a validation-selected deployment policy.'])
    for filename,title in [('kaist_controller_detections.gif','Recorded daytime controller replay'),('kaist_controller_night_detections.gif','Recorded nighttime controller replay')]:
        image=repo/'figures'/filename
        if image.exists():
            s=slide(title)
            s.shapes.add_picture(str(image),Inches(2.5),Inches(1.15),height=Inches(5.6))
            text(s,.7,6.7,12,.25,'Saved predictions and controller decisions; illustrative clips. KAIST, Hwang et al., CVPR 2015; CC BY-NC-SA 4.0.',11)
    slide('RGBT-Tiny and M3FD comparison',[
        'Use the same frozen weights and fixed576 input long side.',
        'Rescore both datasets on bus, car and person; retain negative images.',
        'Use original RGBT-Tiny visible-coordinate labels and paired thermal images.',
        'Completed: '+str(len(metrics))+' / 3 model evaluations.'+(' All results saved.' if complete else ' Remaining results pending.')])
    if metrics:
        slide('Shared-class frozen transfer results',[
            *[f"{names[m['model']]}: M3FD AP {100*m['M3FD']['AP']:.3f}%; RGBT-Tiny AP {100*m['RGBT_Tiny']['AP']:.3f}%." for m in metrics],
            'COCO IoU AP, not SAFit; source-trained transfer, not matched within-dataset training.'])
    if platform_metrics:
        slide('Matched input deployment costs',[
            *[f"{names[m]} mean service: Orin {next(r['mean_service_ms'] for r in platform_metrics if r['platform']=='orin' and r['model']==m):.2f} ms; workstation {next(r['mean_service_ms'] for r in platform_metrics if r['platform']=='workstation' and r['model']==m):.2f} ms." for m in names],
            'Same 120 predecoded pairs, fixed576, three fresh processes per model/platform.',
            'Hardware and backends differ. Host rails and GPU energy scopes are not comparable.'])
    slide('Rigor and interpretation',[
        'Models selected on validation; no KAIST or RGBT-Tiny target fitting.',
        'Plans frozen before inference, randomized case order and preserved source hashes.',
        'KAIST: 1,000 paired-sequence bootstrap resamples; only six sequence units.',
        'One long-video run per case; no repeat-run or broad robustness claim.'])
    slide('Remaining work and conference direction',[
        'Finish faithful published baselines; extend platform costs to longer independent clips.',
        'Record longer synchronized camera sessions; human-check labels for accuracy.',
        'Explain transfer failures, then investigate reliability-aware probing and fallback.',
        'Confirm any new controller on untouched data; agree on final scope with the instructor.'])
    deck=a.output/'EMCH585_PROGRESS_REVIEW_20261123.pptx';prs.save(deck)
    print(report);print(deck)


if __name__=='__main__':main()
