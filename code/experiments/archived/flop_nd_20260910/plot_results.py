"""Standalone research figures from measured, hash-recorded artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import common

CONDITIONS=("relative_full","relative_reset","ppo_full","ppo_reset")
LABELS={"relative_full":"Relative · full","relative_reset":"Relative · reset",
        "ppo_full":"PPO · full","ppo_reset":"PPO · reset"}
COLORS=dict(zip(CONDITIONS,("#0072B2","#E69F00","#009E73","#CC79A7")))


def style():
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":130,"savefig.dpi":200,"font.family":"DejaVu Sans"})


def save(fig,prefix,inputs,description):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    for extension in ("png","pdf","svg"):fig.savefig(prefix.with_suffix("."+extension),bbox_inches="tight")
    common.write_json(prefix.with_suffix(".json"),{"description":description,"code_sha256":common.sha(__file__),
        "input_sha256":{str(p.resolve()):common.sha(p) for p in inputs},
        "outputs_sha256":{extension:common.sha(prefix.with_suffix("."+extension)) for extension in ("png","pdf","svg")}})
    plt.close(fig)


def curves(prefix):
    base_path=common.ROOT/"results/baseline_development_greedy_v1/result.json"
    baseline=json.loads(base_path.read_text())["metrics"];inputs=[base_path]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharex=True)
    for ax in axes:ax.grid(axis="y",alpha=.2);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
    get_metrics=lambda m:(m["all"]["any_successes"]/m["all"]["theorems"],
        sum(m["length_bin:"+b]["any_successes"] for b in ("7-9","10-12","13-20"))/288)
    base=get_metrics(baseline)
    for ax,value in zip(axes,base):ax.axhline(value,color="#444444",ls=":",lw=1.1,label="Initialization")
    for condition in CONDITIONS:
        for seed in (1,2):
            directory=common.ROOT/"results"/f"main_{condition}_seed{seed}_v1"
            updates_path=directory/"updates.jsonl"
            if not updates_path.exists():continue
            # Training may still be writing; only complete JSON lines are read.
            steps={}
            with updates_path.open() as stream:
                for line in stream:
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    steps[row["step"]]=row["cumulative_flop_proxy"]
            points=[(0,*base)]
            for step in (500,1000,2000,4000,5000):
                path=directory.parent/(directory.name+f"_dev_greedy_step{step:06d}")/"result.json"
                if path.exists() and step in steps:
                    m=json.loads(path.read_text())["metrics"];points.append((steps[step],*get_metrics(m)));inputs.append(path)
            final=directory.parent/(directory.name+"_dev_greedy")/"result.json"
            train_result=directory/"result.json"
            if final.exists() and train_result.exists():
                points.append((json.loads(train_result.read_text())["cumulative_flop_proxy"],*get_metrics(json.loads(final.read_text())["metrics"])))
                inputs.extend([final,train_result])
            if len(points)<2:continue
            inputs.append(updates_path)
            points.sort();array=np.array(points)
            for j,ax in enumerate(axes):
                ax.plot(array[:,0]/1e15,array[:,j+1],color=COLORS[condition],ls="-" if seed==1 else "--",marker="o",ms=3,
                        label=LABELS[condition]+f" · seed {seed}")
    axes[0].set_title("All 384 development theorems")
    axes[1].set_title("288 theorems with references above six lines")
    axes[0].set_ylabel("Unaided greedy verified coverage")
    for ax in axes:ax.set_xlabel("Additional declared FLOP proxy (×10¹⁵)")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,-.01),ncol=3,frameon=False)
    fig.tight_layout()
    save(fig,prefix,inputs,"Development-only fixed checkpoints; no model selection. Solid/dashed lines are separately trained seeds. FLOPs are a declared dense proxy, not hardware measurements. Initialization costs are excluded.")


def endpoints(analysis_path,prefix):
    report=json.loads(Path(analysis_path).read_text());summaries=report["summaries"]
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    panels=[("all","first","All final theorems"),("reference_over6","first","References above six lines"),
            ("all","shortest_found_pruned_over6","Verified pruned certificates above six lines")]
    for ax,(stratum,metric,title) in zip(axes,panels):
        baseline=summaries["baseline"][stratum];base=baseline[metric]/baseline["theorems"]
        ax.axhline(base,color="#444444",ls=":",label="Initialization")
        for index,condition in enumerate(CONDITIONS):
            values=[]
            for seed in (1,2):
                key=f"main_{condition}_seed{seed}_v1"
                m=summaries[key][stratum];values.append(m[metric]/m["theorems"])
            ax.bar(index,np.mean(values),color=COLORS[condition],alpha=.55,width=.65)
            ax.scatter([index-.08,index+.08],values,color=COLORS[condition],edgecolor="black",linewidth=.4,s=28,zorder=3)
        ax.set_xticks(range(4),["Relative\nfull","Relative\nreset","PPO\nfull","PPO\nreset"])
        ax.set_title(title,fontsize=10);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis="y",alpha=.2)
        ax.set_ylim(0,1 if metric=="first" else max(.03,ax.get_ylim()[1]*1.1))
    axes[0].set_ylabel("Fraction of theorems");axes[0].legend(frameon=False,loc="upper right",fontsize=8)
    fig.tight_layout()
    save(fig,prefix,[Path(analysis_path)],"Unaided greedy final evaluation at fixed512-token cap. Bars average two training seeds; dots show them separately. The right panel describes found certificates, not a lower bound on theorem difficulty.")


def effects(analysis_path,prefix):
    report=json.loads(Path(analysis_path).read_text())
    names=[("reset_with_relative","Reset effect with relative objective"),("reset_with_ppo","Reset effect with PPO"),
           ("ppo_with_full","PPO effect with full starts"),("ppo_with_reset","PPO effect with resets"),
           ("interaction","Interaction: difference of reset effects")]
    fig,ax=plt.subplots(figsize=(8,3.7));ax.axvline(0,color="#777777",lw=1)
    for index,(key,label) in enumerate(names):
        metric=report["contrasts"][key]["reference_over6"]["first"]["mean_across_seeds"]
        estimate=100*metric["estimate"];low,high=np.array(metric["ci95_context_bootstrap"])*100
        ax.hlines(index,low,high,color="#0072B2",lw=1.5)
        ax.plot([low,high],[index,index],"|",color="#0072B2",markersize=6)
        ax.plot(estimate,index,"o",color="#0072B2")
    ax.set_yticks(range(len(names)),[label for _,label in names]);ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    ax.set_xlabel("Change in verified coverage (percentage points)")
    ax.set_title("Final reference >6 stratum · paired context bootstrap")
    fig.tight_layout();save(fig,prefix,[Path(analysis_path)],"95% percentile bootstrap intervals resample canonical contexts, keeping theorem comparisons paired and averaging the two training-seed contrasts before resampling. These intervals do not estimate uncertainty across new training seeds.")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("curves","endpoints","effects"));p.add_argument("--analysis",type=Path);p.add_argument("--prefix",type=Path,required=True)
    args=p.parse_args();style()
    if args.mode=="curves":curves(args.prefix)
    elif not args.analysis:p.error("Endpoint/effect plots require --analysis")
    elif args.mode=="endpoints":endpoints(args.analysis,args.prefix)
    else:effects(args.analysis,args.prefix)
