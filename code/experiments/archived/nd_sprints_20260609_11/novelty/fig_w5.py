"""W5 figure: novelty-seeded RL. High vs low lemma-value targets (difficulty
matched). Transfer uplift, discovery, and harvest per round.

Usage: python fig_w5.py high_history.json low_history.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

high = json.load(open(sys.argv[1]))
low = json.load(open(sys.argv[2]))
out = sys.argv[3]

hr = [x['round'] for x in high]
lr = [x['round'] for x in low]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
axes[0].plot(hr, [x['transfer_passk'] for x in high], 'o-', color='tab:red', label='high lemma-value targets')
axes[0].plot(lr, [x['transfer_passk'] for x in low], 's-', color='tab:blue', label='low lemma-value targets')
axes[0].set_xlabel('hindsight-EI round'); axes[0].set_ylabel('held-out transfer pass@8')
axes[0].set_title('Transfer (common held-out set)', fontsize=11)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].plot(hr, [x['buffer_size'] for x in high], 'o-', color='tab:red', label='high')
axes[1].plot(lr, [x['buffer_size'] for x in low], 's-', color='tab:blue', label='low')
axes[1].set_xlabel('round'); axes[1].set_ylabel('cumulative theorems proven')
axes[1].set_title('Discoveries', fontsize=11)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

axes[2].plot(hr, [x.get('harvested_size', 0) for x in high], 'o-', color='tab:red', label='high')
axes[2].plot(lr, [x.get('harvested_size', 0) for x in low], 's-', color='tab:blue', label='low')
axes[2].set_xlabel('round'); axes[2].set_ylabel('cumulative harvested prefix-proofs')
axes[2].set_title('Hindsight harvest (training tokens)', fontsize=11)
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

fig.suptitle('W5: seeding RL toward interesting (high lemma-value) vs mundane targets, difficulty-matched', fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=130)
print('wrote', out)
