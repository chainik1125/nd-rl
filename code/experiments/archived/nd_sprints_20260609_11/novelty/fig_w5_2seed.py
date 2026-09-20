"""W5 two-seed figure: high vs low lemma-value transfer across both seeds,
plus harvest density. Args: hi0 lo0 hi1 lo1 out"""
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hi0, lo0, hi1, lo1 = [json.load(open(p)) for p in sys.argv[1:5]]
out = sys.argv[5]
R = lambda h: [x['round'] for x in h]
P = lambda h: [x['transfer_passk'] for x in h]
H = lambda h: [x.get('harvested_size', 0) for x in h]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].plot(R(hi0), P(hi0), 'o-', color='tab:red', label='high-novelty, seed 0')
axes[0].plot(R(lo0), P(lo0), 's-', color='tab:blue', label='low-novelty, seed 0')
axes[0].plot(R(hi1), P(hi1), 'o--', color='tab:red', alpha=0.6, label='high-novelty, seed 1')
axes[0].plot(R(lo1), P(lo1), 's--', color='tab:blue', alpha=0.6, label='low-novelty, seed 1')
axes[0].set_xlabel('hindsight-EI round'); axes[0].set_ylabel('held-out transfer pass@8 (n=500)')
axes[0].set_title('Transfer: high-novelty ends above low in both seeds', fontsize=11)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].plot(R(hi0), H(hi0), 'o-', color='tab:red', label='high, seed 0')
axes[1].plot(R(lo0), H(lo0), 's-', color='tab:blue', label='low, seed 0')
axes[1].plot(R(hi1), H(hi1), 'o--', color='tab:red', alpha=0.6, label='high, seed 1')
axes[1].plot(R(lo1), H(lo1), 's--', color='tab:blue', alpha=0.6, label='low, seed 1')
axes[1].set_xlabel('round'); axes[1].set_ylabel('cumulative harvested prefix-proofs')
axes[1].set_title('Harvest density: high-novelty ~60-70% more, both seeds', fontsize=11)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.suptitle('W5: novelty-seeded RL (difficulty-matched high vs low lemma-value), two seeds', fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=140)
print('wrote', out)
