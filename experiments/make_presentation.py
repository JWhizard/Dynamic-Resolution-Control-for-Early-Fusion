"""Create a reviewable progress deck from measured artifacts (not fabricated results)."""
import argparse
import csv
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--results',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    rows=list(csv.DictReader((a.results/'summary.csv').open()));intervals=json.loads((a.results/'endurance_intervals.json').read_text())
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    def text(slide,left,top,width,height,content,size=24,color=(232,235,245)):
        box=slide.shapes.add_textbox(Inches(left),Inches(top),Inches(width),Inches(height));tf=box.text_frame;tf.word_wrap=True
        for i,line in enumerate(content.split('\n')):
            paragraph=tf.paragraphs[0] if i==0 else tf.add_paragraph();paragraph.text=line
            paragraph.font.name='Aptos';paragraph.font.size=Pt(size);paragraph.font.color.rgb=RGBColor(*color)
            paragraph.space_after=Pt(14)
        return box
    def slide(title,lines=None):
        s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor(15,17,25)
        text(s,.55,.35,12.2,.8,title,30,(170,160,255))
        if lines:text(s,.65,1.55,12,5.15,'\n'.join(lines),23)
        text(s,.65,7.0,12,.3,'EMCH585 • November 13, 2026 • PROGRESS DRAFT — instructor review pending',11,(160,168,184))
        return s
    slide('RGB/IR resolution control: what the evidence supports',[
        'Research question: Can causal resolution control improve accuracy, latency and energy under changing resources?',
        'Working prototype + measured Jetson CPU and TensorRT GPU studies.',
        'Broad robustness and cross-detector/platform generality remain unproven.'])
    slide('Method and comparisons',[
        'Registered B,G,R,IR → resolution choice → detector → native-coordinate predictions.',
        'Existing scene sequence policy; online deadline feedback; AIMD; four fixed resolutions.',
        'Experimental guarded policy: scene preference + online cost cap + four-frame dwell + 1 ms switch reserve.',
        'Ablations remove scene feedback, resource feedback, dwell or switch reserve. No KAIST fitting.'])
    slide('Data and protocol',[
        'FLIR: 120 unique indoor pairs; repeated for cost/endurance measurements; no ground truth.',
        'KAIST: official checksum-verified preview; first 1,200 frames of one night training-preview sequence.',
        'Person-only COCO adaptation, explicit class mapping and ignore regions; not official KAIST benchmark scoring.',
        'One detector family, one physical Orin. These data do not establish multiple independent environments.'])
    slide('Validated GPU deployment',[
        'TensorRT 10.3 / CUDA runtime 12.6 on Jetson AGX Orin; strict FP32, TF32 disabled.',
        'CPU preprocessing preserved; GPU span, transfer/host cost, postprocessing and controller timed separately.',
        'Sampled numerical gates: 32 FLIR cases, 32 KAIST cases, plus 64 KAIST low-confidence cases passed.',
        'Numerical agreement is not proof of full-dataset AP equivalence or sensor synchronization.'])
    slide('GPU endurance: five fresh-process runs',[
        '12,000 timed frames, but only 120 unique source pairs. Resource trace: 4 → 1 → 4 CPU threads.',
        f"Mean run p95: {intervals['p95_ms']['mean']:.2f} ms; mean run p99: {intervals['p99_ms']['mean']:.2f} ms.",
        f"Controller mean: {intervals['controller_mean_ms']['mean']:.3f} ms; rail-sum energy: {intervals['rail_sum_J_per_frame']['mean']:.3f} J/frame.",
        'Run-level bootstrap intervals are in the report; they do not describe unseen scenes/platforms. Rail sums are not wall-plug energy.'])
    selected=[r for r in rows if r['run'] in ['kaist_fixed-480','kaist_fixed-576','kaist_sequence','kaist_guarded']]
    slide('Labeled night transfer: a failure case',[
        *[f"{r['policy']}: person AP {float(r['AP']):.4f}; recall@.5/conf .25 {float(r['person_recall50_conf25']):.1%}." for r in selected],
        'All are weak absolute results. Faster execution does not make this frozen detector robust.',
        'Class names, image/label dimensions, native-shape numerical agreement and sample overlays were checked.'])
    s=slide('Accuracy–cost observations (diagnostic)')
    s.shapes.add_picture(str(a.results/'accuracy_cost.png'),Inches(.6),Inches(1.45),width=Inches(12.1))
    text(s,.8,6.6,11.8,.35,'Data: KAIST preview (CC BY-NC-SA 4.0), Soonmin Hwang et al.; one night prefix, frozen model.',11,(170,178,190))
    s=slide('Guarded-policy ablations: no generality claim')
    s.shapes.add_picture(str(a.results/'ablations.png'),Inches(.6),Inches(1.45),width=Inches(12.1))
    pressure=[r for r in rows if r['run'] in ['resource_cpu_sequence','resource_gpu_sequence','resource_memory_sequence']]
    slide('Resource pressure and streaming semantics',[
        *[f"{r['run'].replace('resource_','')}: service p95 {float(r['p95_ms']):.1f} ms; budget misses {float(r['budget_miss_fraction']):.1%}." for r in pressure],
        'Separate CPU, GPU and bounded memory workers; raw transition traces and source hashes preserved.',
        'Streaming AP holds latest completed boxes at each input arrival. It never uses future predictions or tracking.'])
    slide('Reproducibility and honest limits',[
        'Source/config snapshots, raw predictions, power samples, hashes, job order and environment metadata archived.',
        'Warm switching, process-cold startup and engine-build wall time are separate measurements.',
        'No faithful published adaptive-baseline reproduction yet; upstream code snapshots and plans prepared.',
        'External power calibration, real scene-transition annotations and independent environment test sets remain.'])
    slide('Next actions and ownership',[
        'Instructor: confirm deadline, claims, minimum detector/dataset/platform counts and success margins.',
        'Workstation 10.173.92.11: authorized for training; SSH username/key access still required.',
        'GPU-only Faster R-CNN/RetinaNet recipes prepared; construction/data-contract tested; no GPU training launched.',
        'Then: train/validate additional families, reproduce adaptive baselines, evaluate held-out domains and additional physical platforms.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);prs.save(a.output);print(a.output)


if __name__=='__main__':main()
