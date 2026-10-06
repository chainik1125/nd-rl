#!/usr/bin/env python3
"""figures/B3_judge_matrix.png: generator x judge heatmaps (rows = generator, columns = judge s0/s1/s2, siblings,
xseed) of within-theorem finished-only AUROC (top) and k=8 completion_loglik selection accuracy (bottom), per test set.
Self-judging cells (diagonal) are outlined."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, '..', 'results', 'B3_results.json')))
GENS = ('s0', 's1', 's2'); COLS = ('s0', 's1', 's2', 'siblings', 'xseed')
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ci, set_ in enumerate(('holdout250', 'textbook72')):
    for ri, (metric, lab) in enumerate((('auc', 'within-theorem AUROC, finished attempts'),
                                        ('sel8', 'k=8 selection accuracy (completion_loglik)'))):
        ax = axes[ri, ci]
        M = np.array([[R[set_]['cells'][metric][g][c] for c in COLS] for g in GENS])
        im = ax.imshow(M, cmap='Blues', vmin=M.min() - 0.02, vmax=M.max() + 0.01, aspect='auto')
        for gi in range(3):
            for cj in range(5):
                v = M[gi, cj]
                dark = (v - (M.min() - 0.02)) / (M.max() + 0.01 - (M.min() - 0.02)) > 0.6
                ax.text(cj, gi, f'{v:.3f}', ha='center', va='center', fontsize=10, color='white' if dark else '#0b0b0b')
            ax.add_patch(Rectangle((gi - 0.5, gi - 0.5), 1, 1, fill=False, ec='#e34948', lw=2.5))
        ax.axvline(2.5, color='white', lw=4)
        ax.set_xticks(range(5)); ax.set_xticklabels(['s0', 's1', 's2', 'siblings\n(other two)', 'xseed\n(all three)'], fontsize=9)
        ax.set_xlabel('judge (checkpoint whose likelihood scores the proof)', fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels([f'proofs written by {g}' for g in GENS], fontsize=9)
        tw = R[set_]['twoway'][metric]
        ax.set_title(f'{set_}: {lab}\nself effect (two-way fit) {tw["delta"]:+.3f} [{tw["lo"]:+.3f}, {tw["hi"]:+.3f}]', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
fig.suptitle('A model is the worst judge of its own proofs: the outlined diagonal (generator = judge) is the lowest cell in 5 of 6 rows', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, '..', 'figures', 'B3_judge_matrix.png'), dpi=130)
