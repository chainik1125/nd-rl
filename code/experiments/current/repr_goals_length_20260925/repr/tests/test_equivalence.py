"""The repr pipeline reproduces the lengthexp pipeline exactly for the abs baseline (GPU; take a lock first).

    cd ab3/repr && $PY -m pytest -q -p no:cacheprovider tests/test_equivalence.py

1. repr/sampler.py == lengthexp/sampler.py: identical sampled ids (same checkpoint, prompts, seed) for arm C's
   checkpoint (lengthexp/runs/C_s101/final.pt) and the released 039 s0 pretrain.
2. rl_repr.py == rl_C.py for abs: identical graph_generate and fast_generate samples, identical packed batches
   (x, mask, positions, segments), identical encode_batch output (same RNG draws).
(The full harness smoke runs of rl_C.py and rl_repr.py on the same frozen copy are compared in
tests/run_gpu_tests.sh: same metrics and found proofs.)
"""
import importlib.util, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPR = os.path.dirname(HERE)
sys.path.insert(0, REPR)
from common_repr import FROZEN0, LE, REF039, RL_C, TRAIN_CAP6, lib_path, load_model, read_jsonl  # noqa: E402
lib_path()
import pytest  # noqa: E402
import torch  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device visible')
CKPTS = [os.path.join(LE, 'runs', 'C_s101', 'final.pt'), os.path.join(REF039.format(seed=0), 'pretrain.pt')]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prompts(n, seed):
    rs = read_jsonl(os.path.join(LE, 'testsets', 'std.jsonl'))
    return [r['prompt'] for r in random.Random(seed).sample(rs, n)]


@cuda
@pytest.mark.parametrize('ckpt', CKPTS)
def test_eval_sampler_identical_to_lengthexp(ckpt):
    old = load_mod('le_sampler', os.path.join(LE, 'sampler.py'))
    import sampler as new
    assert os.path.dirname(os.path.abspath(new.__file__)) == REPR
    model, tok, _ = load_model(ckpt, 'cuda')
    ps = [p for p in prompts(40, 0) for _ in range(4)]
    a = old.sample_ids(model, tok, ps, temperature=0.8, max_new=400, batch=64, seed=11000)
    b = new.sample_ids(model, tok, ps, temperature=0.8, max_new=400, batch=64, seed=11000)
    assert a == b


@cuda
def test_rl_samplers_and_batches_identical_to_rl_C():
    oldrl = load_mod('rl_C_eq', RL_C)
    newrl = load_mod('rl_repr_eq', os.path.join(REPR, 'rl_repr.py'))
    model, tok, _ = load_model(CKPTS[1], 'cuda')
    ps = prompts(96, 1)
    oldrl._GS.clear()
    a = oldrl.graph_generate(model, tok, ps, temperature=1.0, max_new=300, batch=96, seed=5)
    oldrl._GS.clear()
    newrl._GS.clear()
    b = newrl.graph_generate(model, tok, ps, temperature=1.0, max_new=300, batch=96, seed=5)
    newrl._GS.clear()
    assert a == b
    a = oldrl.fast_generate(model, tok, ps, temperature=1.0, max_new=300, batch=48, seed=5)
    b = newrl.fast_generate(model, tok, ps, temperature=1.0, max_new=300, batch=48, seed=5)
    assert a == b
    recs = read_jsonl(TRAIN_CAP6)[:128]
    data = [(tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])) for r in recs]
    s1, m1 = oldrl.encode_batch(data, list(range(128)), tok, random.Random(3), 24)
    s2, m2 = newrl.encode_batch(data, list(range(128)), tok, random.Random(3), 24)
    assert s1 == s2 and m1 == m2
    o = oldrl.pack_rows(s1, m1, tok, 'cuda', 1024)
    n = newrl.pack_rows(s2, m2, tok, 'cuda', 1024, model=model, prng=random.Random(9))
    assert all(torch.equal(x, y) for x, y in zip(o, n))
    with torch.no_grad():
        assert torch.equal(oldrl.packed_logits(model, o[0], o[2], o[3]), newrl.packed_logits(model, n[0], n[2], n[3]))


def test_rl_repr_config_is_arm_C():
    oldrl = load_mod('rl_C_cfg', RL_C)
    newrl = load_mod('rl_repr_cfg', os.path.join(REPR, 'rl_repr.py'))
    assert newrl.CONFIG == oldrl.CONFIG and newrl.SMOKE == oldrl.SMOKE
    assert newrl.CONFIG['max_len_train'] == 11 and newrl.CONFIG['max_proof_lines_train'] == 11


def test_rel_skips_the_index_shift():
    newrl = load_mod('rl_repr_rel', os.path.join(REPR, 'rl_repr.py'))
    from tokenizer import Tokenizer
    tk = Tokenizer('rel')
    recs = read_jsonl(TRAIN_CAP6)[:50]
    data = [(tk.encode_prompt(r['prompt']), tk.encode_proof(r['proof'])) for r in recs]
    s, m = newrl.encode_batch(data, list(range(50)), tk, random.Random(3), 24)
    assert s == [p + q for p, q in data]
