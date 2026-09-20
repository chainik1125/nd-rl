"""Standalone figures from completed, hash-bound calibration analyses."""
from pathlib import Path
import hashlib
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

analysis_path = ROOT / 'reports/calibration_confirmation_models_v1.json'
family_path = ROOT / 'theory/calibration_confirmation_analysis_v1.json'
analysis = json.loads(analysis_path.read_text())
families = json.loads(family_path.read_text())
colors = {'adapt': '#777777', 'transfer': '#009E73', 'uniform_matched': '#0072B2',
          'uniform': '#56B4E9', 'surprise': '#E69F00', 'oracle_matched': '#CC79A7',
          'oracle_total': '#8C3B6C'}
names = {'adapt': 'Adapted', 'transfer': 'Transfer selection', 'surprise': 'Surprise selection',
         'uniform': 'Uniform (768 sources)', 'uniform_matched': 'Matched uniform (192)',
         'oracle_matched': 'Oracle SFT (same training)', 'oracle_total': 'Oracle SFT (total budget)'}
outputs = []


def save(fig, stem):
    for extension in ('png', 'svg', 'pdf'):
        path = OUT / f'{stem}.{extension}'
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        outputs.append(path)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7.6, 5.5))
offsets = {'NEGI': (-10, 14), 'IMPI_flat': (10, 13), 'NEGE': (14, -21), 'ORI2': (-27, 13),
           'ANDI': (-25, 14), 'DN': (0, -23), 'BOTE': (0, -26), 'ANDE1': (-44, -5),
           'ANDE2': (19, -14), 'IMPI_nested': (-48, 12), 'ORI1': (18, -5), 'IMPE': (14, 10)}
for name, item in families['families'].items():
    x, y = item['selection_utility_bits'], item['confirmation_utility_bits']
    lo, hi = item['confirmation_context_bootstrap95']
    color = colors['transfer'] if name in families['frozen_top3'] else '#777777'
    ax.errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt='o', color=color, ecolor=color,
                alpha=.85, capsize=3, ms=6, lw=1)
    ax.annotate(name, (x, y), xytext=offsets[name], textcoords='offset points',
                fontsize=8, ha='center' if name in ('NEGI', 'DN', 'BOTE') else 'left')
ax.plot([-2.8, 7.8], [-2.8, 7.8], color='#bbbbbb', ls='--', lw=1, zorder=0)
ax.axhline(0, color='#dddddd', lw=.8)
ax.axvline(0, color='#dddddd', lw=.8)
ax.set(xlabel='Selection utility: matched control − family (bits/proof)',
       ylabel='Independent confirmation utility (bits/proof)',
       title='Family replacement effects transfer across contexts',
       xlim=(-2.8, 7.6), ylim=(-3.1, 7.2))
ax.text(.03, .96, 'Spearman ρ = 0.895\nGreen: frozen top three', transform=ax.transAxes, va='top')
fig.text(.08, -.015, '100 matched SFT updates per arm; 384 confirmation proofs in 212 contexts.\n'
         'Bars: 95% canonical-context bootstrap intervals at one fixed training seed.', fontsize=9)
save(fig, 'calibration_selection_confirmation_v1')

order = ['adapt', 'transfer', 'uniform_matched', 'uniform', 'surprise', 'oracle_matched', 'oracle_total']
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.1), sharey=True)
for axis, metric, scale, title in zip(axes,
        ['whole_proof_bits', 'attempt_success', 'coverage_at_k'], [1, 100, 100],
        ['Reference loss (bits/proof)', 'Successful attempts (%)', 'Problems covered within 8 attempts (%)']):
    for i, name in enumerate(order):
        item = analysis['models'][name][metric]['balanced']
        value = item['mean'] * scale
        lo, hi = np.array(item['context_bootstrap95']) * scale
        axis.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', color=colors[name], capsize=3, ms=7)
    axis.set_xlabel(title)
    axis.grid(axis='x', color='#eeeeee')
axes[0].set_yticks(range(len(order)), [names[name] for name in order])
axes[0].invert_yaxis()
axes[0].set_xlim(0, 130)
axes[1].set_xlim(15, 45)
axes[2].set_xlim(30, 56)
fig.suptitle('Reference likelihood, average success, and coverage measure different outcomes', y=1.02, fontsize=13)
fig.text(.06, -.04, 'Same 384 confirmation problems, temperature 0.8 and 512 response tokens; reference loss at temperature 1.\n'
         'Each length stratum receives equal weight. Bars: 95% context-bootstrap intervals; seed uncertainty is not estimated.\n'
         'Oracle controls directly train cap-12 selection certificates; other mixtures directly train cap-6 suffixes.', fontsize=9)
fig.tight_layout()
save(fig, 'calibration_confirmation_metrics_v1')

fig, ax = plt.subplots(figsize=(8.5, 5.1))
bins = ['2-6', '7-9', '10-12']
selected = ['adapt', 'transfer', 'uniform_matched', 'oracle_matched']
x = np.arange(3)
for i, name in enumerate(selected):
    values = [analysis['models'][name]['coverage_at_k'][b]['mean'] * 100 for b in bins]
    shift = (i - 1.5) * .19
    bars = ax.bar(x + shift, values, width=.18, color=colors[name], label=names[name])
    for bar, b in zip(bars, bins):
        n = analysis['models'][name]['coverage_at_k'][b]['n_sources']
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(round(analysis['models'][name]['coverage_at_k'][b]['mean'] * n)), ha='center', fontsize=8)
ax.set_xticks(x, ['2–6', '7–9', '10–12'])
ax.set(xlabel='Reference proof lines, including premises (witness upper bounds)', ylabel='Covered within 8 attempts (%)',
       ylim=(0, 103), title='Most calibration coverage gains occur on short-reference problems')
ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper right')
fig.text(.10, -.03, 'Counts above bars; 128 problems per bin. Longer references often admit checked witnesses with ≤6 lines.\n'
         'These bins do not establish minimum proof difficulty. All results are development confirmation.', fontsize=9)
save(fig, 'calibration_confirmation_length_v1')

manifest = {'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (analysis_path, family_path, Path(__file__))},
            'outputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
(OUT / 'calibration_figures_manifest_v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'figures': len(outputs), 'directory': str(OUT)}))
