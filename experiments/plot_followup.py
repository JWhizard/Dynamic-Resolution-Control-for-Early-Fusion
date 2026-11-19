"""Readable main-policy and ablation panels; preserve low absolute AP labels."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def panel(rows,selection,destination,title):
    values={r['policy']:r for r in rows if r['run'].startswith('kaist_')}
    selected=[values[key] for key,label in selection];labels=[label for key,label in selection]
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
    settings=[('AP','Person AP (all < 0.02)',.02),('p95_ms','Service p95 (ms)',65),('rail_sum_J_per_frame','Named-rail energy (J/frame)',1.3)]
    for ax,(key,label,limit) in zip(axes,settings):
        numbers=[float(r[key]) for r in selected]
        ax.barh(range(len(numbers)),numbers,color=['#625bd6' if r['policy'].startswith('fixed') else '#238c82' for r in selected],height=.68)
        ax.set_xlim(0,limit);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for i,value in enumerate(numbers):ax.text(value+limit*.015,i,f'{value:.4f}' if key=='AP' else f'{value:.2f}',va='center',fontsize=9)
    axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
    axes[1].axvline(50,color='#9c3848',ls='--',lw=1)
    fig.suptitle(title,fontsize=13)
    fig.text(.5,.015,'One frozen detector, one night training-preview prefix; single runs, no superiority/significance claim. KAIST: CC BY-NC-SA 4.0.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(destination,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('results',type=Path);a=p.parse_args()
    rows=list(csv.DictReader((a.results/'summary.csv').open()))
    panel(rows,[(f'fixed:{r}',f'Fixed {r}') for r in [480,544,576,640]]+[('sequence','Scene sequence'),('deadline','Deadline feedback'),('guarded','Guarded')],
          a.results/'accuracy_cost.png','Frozen-model KAIST night transfer — all observed person AP values below 0.02')
    panel(rows,[('guarded','Guarded'),('guarded-no-scene','No scene feedback'),('guarded-no-resource','No resource feedback'),('guarded-no-hysteresis','No dwell'),('guarded-no-switch-cost','No switch reserve')],
          a.results/'ablations.png','Guarded-policy ablations — diagnostic, not an official KAIST benchmark')
    print(a.results/'accuracy_cost.png');print(a.results/'ablations.png')


if __name__=='__main__':main()
