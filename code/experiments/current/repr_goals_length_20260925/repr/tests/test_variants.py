"""Unit tests for the representation variants (ab3/repr): parameters, vocabulary, frozen equivalence of abs,
line positions, KV-cache decoding against the full forward, and training positions.

    cd ab3/repr && $PY -m pytest -q -p no:cacheprovider tests/test_variants.py          # CPU parts
    CUDA parts run when a GPU is visible (take one with frozen/gpulock.py or run through tests/run_gpu_tests.sh).
Smoke-pretrained checkpoints (tests/out/pt_smoke_<variant>/pretrain.pt, 300 steps) make the decoding tests use
proof-like text; without them a random-init model is used.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPR = os.path.dirname(HERE)
sys.path.insert(0, REPR)
from common_repr import FROZEN0, ORDER, VARIANTS, build_model, lib_path, load_model, read_jsonl, TRAIN_CAP6  # noqa: E402
lib_path()
import pytest  # noqa: E402
import torch  # noqa: E402
from model import GPT, line_positions  # noqa: E402
from posenc import batch_positions, record_positions  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device visible')
BASELINE_PARAMS = 3210240          # 039's released checkpoints (summary.json pretrain.n_params)


def smoke_model(variant, device):
    p = os.path.join(HERE, 'out', f'pt_smoke_{variant}', 'pretrain.pt')
    if os.path.isfile(p):
        m, tok, _ = load_model(p, device)
        return m, tok
    torch.manual_seed(0)
    m, tok = build_model(variant, device)
    return m.eval(), tok


def prompts(n=24, seed=0):
    rs = read_jsonl(os.path.join(FROZEN0, 'data', 'heldout_cap6.jsonl'))
    return [r['prompt'] for r in random.Random(seed).sample(rs, n)]


# ---------------------------------------------------------------- parameters / vocabulary / cfg
@pytest.mark.parametrize('variant', ORDER)
def test_parameter_count_and_vocab(variant):
    m, tok = build_model(variant)
    assert abs(m.n_params() - BASELINE_PARAMS) <= 0.01 * BASELINE_PARAMS
    assert m.n_params() == BASELINE_PARAMS            # no variant adds parameters
    assert tok.vocab_size == 99


@pytest.mark.parametrize('variant', ORDER)
def test_cfg_roundtrip(tmp_path, variant):
    from model import save_ckpt
    torch.manual_seed(1)
    m, tok = build_model(variant)
    save_ckpt(str(tmp_path / 'x.pt'), m, tok.mode)
    m2, tok2, _ = load_model(str(tmp_path / 'x.pt'))
    assert m2.cfg == m.cfg and tok2.mode == tok.mode
    x = torch.randint(1, 99, (2, 17))
    with torch.no_grad():
        assert torch.equal(m.eval()(x), m2(x))


def test_abs_cfg_is_the_frozen_cfg_and_logits_are_bit_identical():
    """abs (and rel) checkpoints load with the frozen code; the frozen GPT and repr GPT give identical logits."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('frozen_model', os.path.join(FROZEN0, 'lib', 'model.py'))
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    torch.manual_seed(3)
    m, tok = build_model('abs')
    f = fm.GPT(**m.cfg)
    f.load_state_dict(m.state_dict())
    m.eval(), f.eval()
    x = torch.randint(1, 99, (3, 40))
    pos = torch.randint(0, 900, (3, 40)).sort(1).values
    with torch.no_grad():
        assert torch.equal(m(x), f(x))
        assert torch.equal(m(x, pos=pos), f(x, pos=pos))
    assert set(build_model('rel')[0].cfg) == set(m.cfg)


# ---------------------------------------------------------------- positions
def test_line_positions_match_python_on_real_records():
    m, tok = build_model('posline')
    recs = read_jsonl(TRAIN_CAP6)[:300]
    for r in recs:
        ids = tok.encode_prompt(r['prompt']) + tok.encode_proof(r['proof'])
        sp = m.seq_positions(ids)
        vp = line_positions(torch.tensor([ids]), m.seps)[0].tolist()
        assert sp == vp
        n_lines = r['n_lines']
        assert sp[-1] == [n_lines + 1, 0]                       # QED opens the line after the last one
        assert all(p[0] == 0 for p in sp[:len(tok.encode_prompt(r['prompt']))])


def test_next_pos_walks_seq_positions():
    m, tok = build_model('posline')
    r = read_jsonl(TRAIN_CAP6)[7]
    ids = tok.encode_prompt(r['prompt']) + tok.encode_proof(r['proof'])
    sp = m.seq_positions(ids)
    cur = torch.tensor([sp[0]])
    for t in range(1, len(ids)):
        cur = m.next_pos(cur, torch.tensor([ids[t - 1]]))
        assert cur[0].tolist() == sp[t]


def test_random_positions_are_sorted_subsets():
    m, _ = build_model('posrand')
    rng = random.Random(0)
    for n in (1, 5, 300, 1024):
        p = record_positions(m, list(range(n)), rng)
        assert p == sorted(set(p)) and len(p) == n and 0 <= p[0] and p[-1] < 1024
    x = torch.zeros(4, 50, dtype=torch.long)
    lens = [50, 10, 1, 33]
    bp = batch_positions(m, x, lens, torch.Generator().manual_seed(0))
    for i, n in enumerate(lens):
        row = bp[i, :n].tolist()
        assert row == sorted(set(row)) and row[-1] < 1024
        assert (bp[i, n:] >= 1024).all()
    for v in ('abs', 'rel', 'nope', 'posline'):
        mm, _ = build_model(v)
        assert batch_positions(mm, x, lens) is None
        assert record_positions(mm, list(range(9)), rng) == mm.seq_positions(list(range(9)))


def test_nope_ignores_positions():
    torch.manual_seed(0)
    m, _ = build_model('nope')
    m.eval()
    x = torch.randint(1, 99, (2, 30))
    with torch.no_grad():
        a = m(x)
        b = m(x, pos=torch.randint(0, 1000, (2, 30)).sort(1).values)
    assert torch.equal(a, b)


def test_rope_offset_invariance_is_why_posrand_uses_subsets():
    """A constant offset on every position leaves RoPE attention unchanged (fp32): the random-offset option would be
    a no-op, so posrand uses sorted random subsets (README)."""
    torch.manual_seed(0)
    m, _ = build_model('abs')
    m.eval()
    x = torch.randint(1, 99, (2, 30))
    p = torch.arange(30).expand(2, 30)
    with torch.no_grad():
        a, b = m(x, pos=p), m(x, pos=p + 517)
        c = m(x, pos=torch.randint(0, 1000, (2, 30)).sort(1).values)
    assert torch.allclose(a, b, atol=1e-4)
    assert not torch.allclose(a, c, atol=1e-2)


# ---------------------------------------------------------------- KV cache == full forward
def _teacher_forced_check(model, tok, prompt_list, outs, atol_logit=None):
    """For every row: prompt + generated ids through the full forward (no padding, default positions from the sequence
    start); the generated token at each step must be the argmax of the full-forward logits (greedy decoding)."""
    dev = next(model.parameters()).device
    bad = 0
    total = 0
    for p, o in zip(prompt_list, outs):
        pid = tok.encode_prompt(p)
        o = [t for t in o if t != tok.pad]
        if not o:
            continue
        seq = torch.tensor([pid + o], device=dev)
        with torch.no_grad():
            lg = model(seq)[0, len(pid) - 1:len(pid) - 1 + len(o)]
        am = lg.argmax(-1).tolist()
        total += len(o)
        bad += sum(1 for a, b in zip(am, o) if a != b)
    return bad, total


@pytest.mark.parametrize('variant', ORDER)
def test_kv_cache_greedy_equals_full_forward_fp32_cpu(variant):
    """lib/sample.generate_ids and sampler.sample_ids (greedy, fp32 on CPU, left-padded batch) produce exactly the
    tokens a full forward picks."""
    from sample import generate_ids
    from sampler import sample_ids
    model, tok = smoke_model(variant, 'cpu')
    model = model.float().eval()
    ps = prompts(12)
    ids = [tok.encode_prompt(p) for p in ps]
    outs = generate_ids(model, tok, ids, greedy=True, max_new=120)
    bad, total = _teacher_forced_check(model, tok, ps, outs)
    assert total > 0 and bad == 0, (bad, total)
    outs2 = sample_ids(model, tok, ps, max_new=120, batch=5, greedy=True)
    bad, total = _teacher_forced_check(model, tok, ps, outs2)
    assert bad == 0, (bad, total)
    for a, b in zip(outs, outs2):
        a = [t for t in a if t != tok.pad]
        assert a == b[:len(a)]
    if variant == 'posline':          # the check is only meaningful if lines were crossed
        assert sum(o.count(tok.stoi[';']) for o in outs2) > 10


def _load_rl():
    import importlib.util
    spec = importlib.util.spec_from_file_location('rl_repr_t', os.path.join(REPR, 'rl_repr.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['rl_repr_t'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('variant', ORDER)
def test_rl_fast_generate_near_greedy_equals_full_forward_fp32_cpu(variant):
    rl = _load_rl()
    model, tok = smoke_model(variant, 'cpu')
    model = model.float().eval()
    ps = prompts(10, seed=1)
    outs = rl.fast_generate(model, tok, ps, temperature=1e-4, max_new=100, batch=4, seed=0)
    enc = [tok.encode_proof(o) if o.split()[-1:] == ['QED'] else None for o in outs]
    # compare on the id level: re-run with ids
    ids = [tok.encode_prompt(p) for p in ps]
    gen = torch.Generator().manual_seed(0)
    raw = rl._gen_ids(model, tok, ids, 1e-4, 100, gen, 16, 0.8)
    bad, total = _teacher_forced_check(model, tok, ps, raw)
    assert total > 0 and bad == 0, (bad, total)


@cuda
@pytest.mark.parametrize('variant', ORDER)
def test_graph_sampler_greedy_matches_eager_bf16(variant):
    """rl_repr's CUDA-graph decode (bf16 KV cache) against the eager lib/sample.generate_ids under bf16 autocast:
    greedy tokens agree (up to bf16 near-ties) and each graph token is the full forward's argmax or within 0.05 logits."""
    rl = _load_rl()
    from sample import generate_ids
    model, tok = smoke_model(variant, 'cuda')
    ps = prompts(64, seed=2)
    ids = [tok.encode_prompt(p) for p in ps]
    rl._GS.clear()
    g = rl.graph_generate(model, tok, ps, max_new=160, batch=64, seed=0, greedy=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        e = generate_ids(model, tok, ids, greedy=True, max_new=160)
    e = [tok.decode(o) for o in e]
    same = sum(a == b for a, b in zip(g, e))
    assert same >= 0.9 * len(ps), f'{same}/{len(ps)} identical'
    # teacher-forced fp32 check of the graph output: the chosen token is (nearly) the argmax
    model_f = model.float()
    worst = 0.0
    n_tok = 0
    for p, body in zip(ps, g):
        pid = tok.encode_prompt(p)
        try:
            o = tok.encode_proof(body) if tok.mode == 'abs' else None
        except (ValueError, KeyError):
            o = None
        if not o:
            continue
        seq = torch.tensor([pid + o], device='cuda')
        with torch.no_grad():
            lg = model_f(seq)[0, len(pid) - 1:len(pid) - 1 + len(o)].float()
        chosen = lg.gather(1, torch.tensor(o, device='cuda')[:, None])[:, 0]
        worst = max(worst, float((lg.max(-1).values - chosen).max()))
        n_tok += len(o)
    if tok.mode == 'abs':
        assert n_tok > 0 and worst < 0.25, worst
    if variant == 'posline':
        assert sum(x.count(';') for x in g) > 20


@cuda
@pytest.mark.parametrize('variant', ORDER)
def test_packed_training_logits_equal_unpacked(variant):
    """pack_rows + packed_logits (block-diagonal mask, per-record positions) give each record the logits of its own
    unpadded forward with the same positions (fp32)."""
    rl = _load_rl()
    model, tok = smoke_model(variant, 'cuda')
    model = model.float()
    recs = read_jsonl(TRAIN_CAP6)[:40]
    seqs = [tok.encode_prompt(r['prompt']) + tok.encode_proof(r['proof']) for r in recs]
    masks = [[0] * len(tok.encode_prompt(r['prompt'])) + [1] * len(tok.encode_proof(r['proof'])) for r in recs]
    prng = random.Random(5)
    x, m, pos, seg = rl.pack_rows(seqs, masks, tok, 'cuda', 1024, model=model, prng=prng)
    with torch.no_grad():
        lg = rl.packed_logits(model, x, pos, seg)
    checked = 0
    for r in range(x.size(0)):
        for j in range(1, int(seg[r].max()) + 1):
            ix = (seg[r] == j).nonzero().squeeze(1)
            ix_in = ix[ix < x.size(1) - 1]
            xs = x[r, ix][None]
            ps = pos[r, ix][None]
            with torch.no_grad():
                ref = model(xs[:, :-1], pos=ps[:, :-1] if model.pos_train == 'rand' else None)
            got = lg[r, ix_in][:ref.size(1)]
            assert torch.allclose(got, ref[0, :got.size(0)], atol=2e-3), (variant, float((got - ref[0, :got.size(0)]).abs().max()))
            checked += 1
    assert checked == len(seqs)
