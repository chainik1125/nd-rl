"""The run-3 compute meter (ab3/frozen/meter.py, unchanged) charges every variant exactly like the baseline.

    cd ab3/repr && $PY -m pytest -q -p no:cacheprovider tests/test_meter_variants.py      # CUDA parts need a GPU

No variant adds an nn.Linear or changes one; positions, rotations and line bookkeeping are free (not nn.Linear),
as for the baseline's RoPE. Checked per variant: FLOPs per token, no-grad / grad forward charges, eager sampling
(lib/sample.generate) charges prompt + decode rows, a captured CUDA-graph decode step of rl_repr's sampler charges
bucket rows x FLOPs per token on each replay, and a packed fine-tuning step charges 3 x rows x FLOPs per token.
(The frozen test_meter.py is also run unchanged inside a variant frozen copy: tests/run_gpu_tests.sh.)
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPR = os.path.dirname(HERE)
sys.path.insert(0, REPR)
from common_repr import FROZEN0, ORDER, TRAIN_CAP6, build_model, lib_path, read_jsonl  # noqa: E402
lib_path()
sys.path.insert(1, FROZEN0)
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import meter as M  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device visible')
FPT = 6342144      # frozen test_meter: policy_flops_per_token(99) for GPT(99, 4, 256, 8)


@pytest.fixture
def meter():
    m = M.Meter(None).install()
    yield m
    m.uninstall()


def model_of(variant, device='cpu'):
    torch.manual_seed(0)
    m, tok = build_model(variant, device)
    return m, tok


@pytest.mark.parametrize('variant', ORDER)
def test_flops_per_token_is_the_baseline(variant):
    m, _ = model_of(variant)
    per_row = sum(2 * x.in_features * x.out_features for x in m.modules() if isinstance(x, nn.Linear))
    assert per_row == M.policy_flops_per_token(99) == FPT
    assert sum(1 for x in m.modules() if isinstance(x, nn.Linear)) == 4 * 4 + 1


@pytest.mark.parametrize('variant', ORDER)
def test_forward_charges(meter, variant):
    m, _ = model_of(variant)
    x = torch.randint(1, 99, (3, 17))
    with torch.no_grad():
        m(x)
    assert meter.used() == FPT * 3 * 17
    u = meter.used()
    pos = None
    if m.pe == 'line':
        pos = torch.zeros(2, 11, 2, dtype=torch.long)
    elif m.pos_train == 'rand':
        pos = torch.arange(11).expand(2, 11) * 3
    m(torch.randint(1, 99, (2, 11)), pos=pos).float().mean().backward()
    assert meter.used() - u == 3 * FPT * 2 * 11


@cuda
@pytest.mark.parametrize('variant', ORDER)
def test_eager_generate_charges_every_row(meter, variant):
    from sample import generate
    m, tok = model_of(variant, 'cuda')
    m.eval()
    rows = []
    h = m.register_forward_pre_hook(lambda mod, args: rows.append(args[0].numel()))
    try:
        with open(os.path.join(FROZEN0, 'data', 'targets_L6-11.jsonl')) as f:
            prompts = [json.loads(next(f))['prompt'] for _ in range(5)]
        generate(m, tok, prompts, greedy=True, max_new=12, batch=4)
    finally:
        h.remove()
    assert meter.used() == FPT * sum(rows)


def _rl():
    import importlib.util
    spec = importlib.util.spec_from_file_location('rl_repr_m', os.path.join(REPR, 'rl_repr.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['rl_repr_m'] = mod
    spec.loader.exec_module(mod)
    return mod


@cuda
@pytest.mark.parametrize('variant', ORDER)
def test_graph_decode_step_replay_charges_bucket_rows(meter, variant):
    rl = _rl()
    m, tok = model_of(variant, 'cuda')
    m.eval()
    G = rl._GraphSampler(m, 128, 64, 32, 'cuda')
    with torch.no_grad():
        G.done[:64] = False
        G.keep[:64, 0] = True
        G.step(m, 64, 0, tok.pad, tok.eos)                 # warm-up (charged) + capture (free) + replay (charged)
        u = meter.used()
        G.slot.fill_(1)
        for s in range(1, 6):
            G.step(m, 64, s, tok.pad, tok.eos)             # same key span (slots < 64): replays of the one graph
    torch.cuda.synchronize()
    assert meter.used() - u == 5 * 64 * FPT
    assert meter.graph_replays >= 6


@cuda
@pytest.mark.parametrize('variant', ORDER)
def test_packed_finetune_step_charges_three_times_rows(meter, variant):
    rl = _rl()
    m, tok = model_of(variant, 'cuda')
    recs = read_jsonl(TRAIN_CAP6)[:64]
    seqs = [tok.encode_prompt(r['prompt']) + tok.encode_proof(r['proof']) for r in recs]
    masks = [[0] * len(tok.encode_prompt(r['prompt'])) + [1] * len(tok.encode_proof(r['proof'])) for r in recs]
    x, mk, pos, seg = rl.pack_rows(seqs, masks, tok, 'cuda', 1024, model=m, prng=random.Random(0))
    m.train()
    u = meter.used()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = rl.masked_ce(rl.packed_logits(m, x, pos, seg), x, mk)
    loss.backward()
    assert meter.used() - u == 3 * FPT * x.size(0) * (x.size(1) - 1)
