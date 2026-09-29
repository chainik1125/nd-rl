"""Fresh-final loss-transfer figure, from saved independent statistics."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/proof-fragments-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'reports/final_calibration_ranking_v1.json';d=json.loads(path.read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8,6.4))
    fig.subplots_adjust(bottom=.25)
    offsets={'NEGI':(-8,15),'IMPI_flat':(10,12),'NEGE':(13,-20),'ORI2':(-30,14),'ANDI':(-28,13),'DN':(10,-13),'BOTE':(5,-24),'ANDE1':(-44,-6),'ANDE2':(15,-15),'IMPI_nested':(-48,13),'ORI1':(17,-7),'IMPE':(12,10)}
    for name,v in d['families'].items():
        x,y=v['selection_utility_bits'],v['final_utility_bits'];lo,hi=v['final_context_bootstrap95'];c='#009E73' if name in d['frozen_top3'] else '#777777'
        ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='o',color=c,ecolor=c,capsize=3,markersize=6)
        ax.annotate(name,(x,y),xytext=offsets[name],textcoords='offset points',fontsize=9)
    ax.plot([-2.8,7.9],[-2.8,7.9],ls='--',color='#bbbbbb',lw=1);ax.axhline(0,color='#dddddd',lw=.8);ax.axvline(0,color='#dddddd',lw=.8)
    ax.set(xlim=(-2.8,7.8),ylim=(-3,8),xlabel='Selection loss benefit: matched control − family (bits / proof)',ylabel='Untouched final loss benefit (bits / proof)',title='Supervised family transfer effects repeat on fresh contexts')
    ax.text(.03,.96,f'Spearman ρ = {d["selection_vs_final_spearman"]:.3f}\nGreen: three families selected before confirmation',transform=ax.transAxes,va='top',fontsize=10)
    fig.text(.10,.035,'100 SFT updates per family/control arm; 768 final targets in 402 canonical contexts.\nBars: final-context bootstrap 95% intervals, conditional on one training seed.\nThis tests reference-loss transfer; verified proof completion is evaluated separately.',fontsize=10)
    outputs={}
    for ext in ['png','svg','pdf']:
        p=ROOT/'figures'/f'final_calibration_transfer_v1.{ext}';fig.savefig(p,bbox_inches='tight',dpi=200);outputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__).resolve()]},'outputs':outputs}
    (ROOT/'figures/final_calibration_transfer_manifest_v1.json').write_text(json.dumps(receipt,indent=2)+'\n');plt.close(fig)
    print(json.dumps({'figures':len(outputs)}))

if __name__=='__main__':main()
