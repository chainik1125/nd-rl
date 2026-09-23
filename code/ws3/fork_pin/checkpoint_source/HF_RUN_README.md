# Four-layer pretraining-only checkpoint

Recipe: autoresearch run 001 GPU batching, with a fresh 300-second pretraining budget and
seed 0. Four layers, hidden width 256, eight heads, absolute-reference tokenizer;
AdamW with learning rate 0.001, batch size 256, weight decay 0.1.

`pretrain.pt` contains weights **before any reinforcement learning**. No development or
holdout evaluation is run for this export. This is not an independently evaluated best
pretrained model; the recipe was selected from earlier post-RL development performance.

`source/pretrain.py` contains the exact architecture construction and training implementation.
The external fork is https://github.com/dan-pandori/nd-takehome at commit
`ab629c7d2cdf3dc50d471e302d14643ba0782749` (also pinned in `source/remote.sh`).

After downloading this run folder, install PyTorch and put the pinned fork on PYTHONPATH:

```python
import torch
from model import GPT
from tokenizer import Tokenizer

checkpoint = torch.load('pretrain.pt', map_location='cpu', weights_only=True)
tok = Tokenizer(checkpoint['tok_mode'])
model = GPT(tok.vocab_size, 4, 256, 8)
model.load_state_dict(checkpoint['state'])
model.eval()
```

The checkpoint contains no optimizer state. See `summary.json` for actual elapsed time,
step count, parameter count, device, and source hashes.
