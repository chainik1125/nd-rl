"""Final report presentation of both frozen LM evaluation profiles."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import common

conditions = [('relative_full', 'Relative\nfull', '#0072B2'),
              ('relative_reset', 'Relative\nbackward', '#E69F00'),
              ('ppo_full', 'PPO\nfull', '#009E73'),
              ('ppo_reset', 'PPO\nbackward', '#CC79A7')]
paths = [common.ROOT / f'reports/analysis_primary_final_{p}_v1.json' for p in ('greedy', 'sample4')]
docs = [json.loads(p.read_text()) for p in paths]
assert all(d['status'] == 'completed' for d in docs)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11.3, 7), sharex=True, sharey='row')
rows = []
for col, (profile, doc) in enumerate(zip(('greedy', 'sample4'), docs)):
    for row, (stratum, n, upper) in enumerate((('reference_over6', 576, .15), ('all', 768, .35))):
        ax = axes[row, col]
        base = doc['summaries']['baseline'][stratum]
        assert base['theorems'] == n
        metric = 'first' if profile == 'greedy' else 'any'
        initial = base[metric] / n
        ax.axhline(initial, color='#555555', linestyle=':', linewidth=1.5)
        ax.text(3.35, initial + .006, f'Initial: {base[metric]}/{n}', ha='right', fontsize=9, color='#444444')
        for i, (condition, label, color) in enumerate(conditions):
            vals=[]
            for seed, offset, marker in ((1, -.11, 'o'), (2, .11, 's')):
                name=f'main_{condition}_seed{seed}_v1'; a=doc['summaries'][name][stratum]
                assert a['theorems'] == n
                count=a[metric]; value=count/n; vals.append(value)
                ax.plot(i+offset, value, marker=marker, color=color, markersize=7,
                        markeredgecolor='#333333', markeredgewidth=.5, linestyle='none', zorder=3)
                ax.annotate(str(count), (i+offset, value), xytext=(-7 if seed==1 else 7,-17),
                            textcoords='offset points', ha='center', fontsize=8)
                rows.append({'profile':profile,'stratum':stratum,'endpoint':name,'successes':count,'theorems':n})
            ax.plot([i-.25,i+.25], [sum(vals)/2]*2, color=color, linewidth=2, alpha=.7)
        ax.set_ylim(0, upper);ax.set_xlim(-.45,3.45)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0));ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(4),[c[1] for c in conditions]);ax.tick_params(labelbottom=True)
        ax.set_title(('Declared primary: one greedy attempt' if row==0 and profile=='greedy' else 'One greedy attempt' if profile=='greedy' else 'Secondary: success within four sampled attempts'), fontsize=11)
        if col==0:
            ax.set_ylabel(('References ≥7 lines\n' if row==0 else 'Secondary: all final statements\n')+f'Verified coverage (n={n})')
fig.suptitle('LM training does not improve observed final coverage over initialization', fontsize=14, y=1.025)
fig.legend(handles=[Line2D([],[],color='#333333',marker='o',linestyle='none',label='Training seed 1'),
                    Line2D([],[],color='#333333',marker='s',linestyle='none',label='Training seed 2'),
                    Line2D([],[],color='#555555',linewidth=2,label='Mean of two trained seeds')],
           loc='lower center',bbox_to_anchor=(.5,-.01),ncol=3,frameon=False)
fig.text(.5,-.035,'512 response tokens per attempt. Four-sample temperature: 0.8. Numbers beside points are solved statements.',ha='center',fontsize=9)
fig.text(.5,-.06,'All conditions share the same broadly supervised short-proof LM. Reference length does not establish minimum proof length.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.045,1,1])
prefix=common.ROOT/'figures/report_lm_final_v2'
for ext in ('png','pdf','svg'):fig.savefig(prefix.with_suffix('.'+ext),bbox_inches='tight',dpi=180)
plt.close(fig)
common.write_json(prefix.with_suffix('.json'),{'status':'completed','scope':'Presentation only; all frozen main endpoints and both profiles. Top row is the outcome declared primary before training.',
 'rows':rows,'input_sha256':{str(p):common.sha(p) for p in paths},'code_sha256':common.sha(__file__),
 'outputs_sha256':{ext:common.sha(prefix.with_suffix('.'+ext)) for ext in ('png','pdf','svg')}})
print(json.dumps({'status':'completed','rows':len(rows),'output':str(prefix)}))
