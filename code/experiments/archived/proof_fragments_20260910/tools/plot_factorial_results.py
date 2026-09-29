"""Plot saved independent aggregates; never choose a checkpoint or read proof payloads."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/proof-fragments-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARMS=['selected_full','selected_reset','uniform_full','uniform_reset']
LABELS=['Selected, full starts','Selected, reset curriculum','Uniform, full starts','Uniform, reset curriculum']
COLORS=['#0072B2','#009E73','#D55E00','#CC79A7']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['development','final'],required=True); a=p.parse_args()
    path=ROOT/'reports'/('main_development_seed1_v1.json' if a.phase=='development' else 'final_seed1_v1.json')
    d=json.loads(path.read_text()); outputs=[]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:
            target=ROOT/'figures'/f'{a.phase}_{name}_v1.{ext}'; fig.savefig(target,bbox_inches='tight'); outputs.append(str(target))
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    for ax,metric,title,scale in zip(axes,['coverage_at_k','attempt_success','whole_proof_bits'],['Theorems solved in eight attempts (%)','Successful attempts (%)','Reference loss (bits / proof)'],[100,100,1]):
        baseline=d['models']['adapt'][metric]['balanced']
        ax.axhline(scale*baseline['mean'],color='#555555',ls=':',label='Adapted starting model')
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            vals=[d['models'][f'{arm}_s1_{b}'][metric]['balanced'] for b in ['f05','f10','f20','f40']]
            ax.plot([0,.5,1,2,4],[baseline['mean']*scale]+[v['mean']*scale for v in vals],marker='o',color=color,label=label,linewidth=1.8,markersize=4)
            # The endpoint is primary; full endpoint intervals stay legible without overlapping ribbons.
            lo,hi=vals[-1]['context_bootstrap95']; ax.errorbar([4],[vals[-1]['mean']*scale],yerr=[[scale*(vals[-1]['mean']-lo)],[scale*(hi-vals[-1]['mean'])]],color=color,alpha=.55,capsize=3)
        ax.set(xlabel='Direct training compute proxy (10¹⁵ units)',ylabel=title,xticks=[0,.5,1,2,4]); ax.grid(alpha=.18)
    axes[0].legend(fontsize=8,loc='best')
    fig.suptitle(f'{a.phase.capitalize()} contexts · one online training seed · all frozen main checkpoints',fontsize=14,y=1.03)
    save(fig,'main_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    rows=[('Selection, averaged across starts',d['contrasts']['selection_main_effect_f40']['metrics']),('Reset, averaged across source banks',d['contrasts']['reset_main_effect_f40']['metrics']),('Selection × reset interaction',d['contrasts']['interaction_f40']['metrics'])]
    for arm,label in zip(ARMS,LABELS): rows.append((label+' − adapted',d['comparisons'][f'{arm}_s1_f40_minus_adapt']))
    for ax,metric,label in zip(axes,['coverage_at_k','attempt_success'],['Difference in coverage at eight attempts (pp)','Difference in successful attempts (pp)']):
        for i,(name,v) in enumerate(rows):
            s=v[metric]['balanced']; lo,hi=s['context_bootstrap95']; color='#0072B2' if i<3 else '#555555'
            ax.errorbar(s['mean']*100,i,xerr=[[100*(s['mean']-lo)],[100*(hi-s['mean'])]],fmt='o',color=color,capsize=3)
        ax.axvline(0,color='black',lw=.8); ax.set(yticks=range(len(rows)),yticklabels=[r[0] for r in rows],xlabel=label); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    axes[1].set_yticklabels([])
    fig.suptitle(f'{a.phase.capitalize()} · primary 4 × 10¹⁵ endpoint · paired context 95% intervals',fontsize=13,y=1.02)
    save(fig,'main_endpoint_contrasts')
    if a.phase=='final':
        names=['adapt','selected_full_s1_f40','selected_reset_s1_f40','uniform_full_s1_f40','uniform_reset_s1_f40','retention_step9056','retention_step9054','retention_step9096','retention_step8827','retention_compute','main_oracle_matched','main_oracle_total']
        labels=['Adapted starting model',*LABELS,'Retention only: 9,056 updates','Retention only: 9,054 updates','Retention only: 9,096 updates','Retention only: 8,827 updates','Retention only: equal total compute','Direct acquired-information SFT: main compute','Direct acquired-information SFT: main + selection compute']
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,metric,title in zip(axes,['coverage_at_k','attempt_success'],['Theorems solved in eight attempts (%)','Successful attempts (%)']):
            for i,name in enumerate(names):
                s=d['models'][name][metric]['balanced']; lo,hi=s['context_bootstrap95']; color=COLORS[i-1] if 1<=i<=4 else '#666666'
                ax.errorbar(100*s['mean'],i,xerr=[[100*(s['mean']-lo)],[100*(hi-s['mean'])]],fmt='o',color=color,capsize=3)
            ax.set(yticks=range(len(names)),yticklabels=labels,xlabel=title); ax.invert_yaxis(); ax.grid(axis='x',alpha=.18)
        axes[1].set_yticklabels([]); fig.suptitle('Untouched final contexts · retention exposure and acquired-information controls',y=1.01,fontsize=13)
        save(fig,'controls')
    def sha(q): return hashlib.sha256(Path(q).read_bytes()).hexdigest()
    receipt={'phase':a.phase,'analysis':str(path),'analysis_sha256':sha(path),'script_sha256':sha(__file__),'outputs':{q:sha(q) for q in outputs},'uncertainty':'Paired canonical-context percentile bootstrap; fixed single online training seed, no multiplicity correction. Curves retain every frozen milestone.'}
    out=ROOT/'figures'/f'{a.phase}_factorial_plot_manifest_v1.json'; out.write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps({'receipt':str(out),'figures':len(outputs)}))

if __name__=='__main__': main()
