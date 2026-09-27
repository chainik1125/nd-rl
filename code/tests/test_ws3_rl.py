"""ws3/rl: advantage transforms, pass@k, prompt curriculum, EI helpers, config includes, and
CPU smoke runs of the GRPO subclass and the EI driver (plumbing only, not experiments)."""

import itertools
import json
import math
import os

import ndpath
import pytest
import torch

# isort: split
import run as ws3_run
from eval.evaluator import aggregate, pass_at_k, pass_at_k_curve
from eval.reward import ProofReward
from rl import advantages as A
from rl.curriculum import AdaptiveRepeatSampler, PromptStats
from rl.ei import select_verified, tokenize_records

HAVE = os.path.exists(ndpath.FORK_CKPT) and os.path.exists(
    os.path.join(ndpath.FORK_DATA, "heldout.jsonl")
)
PROMPT = "THM ( P > Q ) , P SEQ Q PRF"
VALID = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def t(*xs):
    return torch.tensor(xs, dtype=torch.float32)


# ------------------------------------------------------------------ pass@k


def test_pass_at_k_matches_enumeration():
    for n, c in [(8, 0), (8, 1), (8, 3), (8, 8), (6, 2)]:
        outcomes = [1] * c + [0] * (n - c)
        for k in range(1, n + 1):
            subsets = list(itertools.combinations(outcomes, k))
            brute = sum(any(s) for s in subsets) / len(subsets)
            assert pass_at_k(n, c, k) == pytest.approx(brute)
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)


def test_pass_at_k_curve_and_aggregate():
    curve = pass_at_k_curve([(4, 0), (4, 1), (4, 4)])
    assert set(curve) == {1, 2, 4}
    assert curve[1] == pytest.approx((0 + 0.25 + 1) / 3)
    assert curve[4] == pytest.approx(2 / 3)
    j = lambda ok, body="b": (body, ok, "ok", 3, 10)  # noqa: E731
    recs = [{"prompt": "a", "n_lines": 3}, {"prompt": "b", "n_lines": 3}]
    res = aggregate(
        recs, [[j(True, "x"), j(True, "x  "), j(True, "y"), j(False)], [j(False)] * 4], k=4
    )
    assert res["pass_at_k"][1] == pytest.approx(3 / 8)
    assert res["by_length"][3]["pass_at_k"][4] == pytest.approx(0.5)
    assert res["distinct_verified_mean"] == 2  # "x" and "x  " are one proof
    assert "pass_at_k" not in aggregate(recs, [[j(True)], [j(False)]], k=1)


# ------------------------------------------------------------------ advantages


def test_maxrl():
    adv = A.maxrl(t(1, 0, 0, 0, 0, 0, 0, 0), 4)
    assert adv[:4].tolist() == pytest.approx([3.0, -1.0, -1.0, -1.0], rel=1e-4)  # (r - .25) / .25
    assert adv[4:].tolist() == [0, 0, 0, 0]  # all-fail group: no signal
    easy = A.maxrl(t(1, 1, 1, 0), 4)
    assert easy.abs().max() < adv.abs().max()  # hard prompts get the bigger update


def test_w_reinforce_and_nsr():
    assert A.w_reinforce(t(1, 0, 0, 0), 4, lam=0.1).tolist() == pytest.approx([0.1, -1, -1, -1])
    assert A.w_reinforce(t(0, 0), 2, lam=0.0).tolist() == [-1, -1]


def test_virtual_positive_gives_all_fail_groups_gradient():
    adv = A.virtual_positive(t(0, 0, 0, 0), 4)
    assert adv.tolist() == pytest.approx([-0.2] * 4)
    mixed = A.virtual_positive(t(1, 0, 0, 0), 4)
    assert mixed[0] == pytest.approx(1 - 0.4) and mixed[1] == pytest.approx(-0.4)


def test_pass_at_k_advantage_matches_enumeration():
    g, k = 6, 3
    for c in range(g + 1):
        s = [1.0] * c + [0.0] * (g - c)
        adv = A.pass_at_k(torch.tensor(s), g, k=k, std_scale=False)
        rho = pass_at_k(g, c, k)
        for i in range(g):
            # expected pass@k score of k-subsets containing sample i, minus the group mean
            others = [s[j] for j in range(g) if j != i]
            subs = list(itertools.combinations(others, k - 1))
            exp = sum(max([s[i], *sub]) for sub in subs) / len(subs)
            assert adv[i].item() == pytest.approx(exp - rho, abs=1e-6)
    k1 = A.pass_at_k(t(1, 0, 0, 1), 4, k=1, std_scale=False)
    assert k1.tolist() == pytest.approx([0.5, -0.5, -0.5, 0.5])
    assert A.pass_at_k(t(0, 0, 0, 0), 4, k=2).tolist() == [0, 0, 0, 0]


def test_unlikely_ranks_successes_by_logprob():
    r = A.unlikely_rewards(t(1, 1, 0, 1), 4, t(-1.0, -5.0, -2.0, -3.0), beta=0.3)
    # successes by likelihood: idx0 (-1) most likely, idx3, idx1 (-5) least likely
    assert r[1] == pytest.approx(1.0)
    assert r[3] == pytest.approx(1 - 0.3 * 1 / 3)
    assert r[0] == pytest.approx(1 - 0.3 * 2 / 3)
    assert r[2] == 0
    adv = A.unlikely(t(1, 1, 0, 1), 4, t(-1.0, -5.0, -2.0, -3.0), beta=0.3)
    assert adv.sum().item() == pytest.approx(0, abs=1e-6)


def test_group_outcomes_and_compute():
    out = A.group_outcomes(t(0, 0, 1, 1, 1, 0), 2)
    assert out == {
        "groups/frac_all_fail": 1 / 3,
        "groups/frac_all_success": 1 / 3,
        "groups/frac_mixed": 1 / 3,
    }
    assert A.compute("maxrl", t(1, float("nan"), 0, 0), 2).tolist()[2:] == [0, 0]
    with pytest.raises(ValueError):
        A.compute("nope", t(1, 0), 2)
    with pytest.raises(ValueError):
        A.maxrl(t(1, 0, 0), 2)


# ------------------------------------------------------------------ curriculum


def _sampler(stats, lengths, **kw):
    prompts = [f"p{i}" for i in range(len(lengths))]
    return prompts, AdaptiveRepeatSampler(
        prompts, lengths, stats, mini_repeat_count=3, batch_size=2, repeat_count=2, seed=0, **kw
    )


def test_sampler_layout_matches_repeat_sampler():
    stats = PromptStats()
    _, s = _sampler(stats, [2, 3, 4, 5], weighting="uniform")
    idx = list(s)
    assert len(idx) == len(s) == 2 * 2 * 3 * 2
    block = idx[:12]  # one batch: 2 prompts x 3 repeats, the whole thing twice
    assert block[:6] == block[6:]
    assert block[0] == block[1] == block[2] and block[3] == block[4] == block[5]
    assert block[0] != block[3]


def test_p1mp_prefers_mixed_prompts_and_drops_solved():
    stats = PromptStats(window=4)
    prompts, s = _sampler(stats, [7, 7, 7, 7], weighting="p1mp", floor=0.0, drop_solved_after=2)
    stats.update(["p0", "p0"], [[0] * 8, [0] * 8])  # hopeless so far
    stats.update(["p1", "p1"], [[1, 0] * 4, [1, 0] * 4])  # mixed
    stats.update(["p2", "p2"], [[1] * 8, [1] * 8])  # solved twice
    assert s.weight(1) > s.weight(0)
    assert s.weight(2) == 0.0
    assert s.weight(3) == pytest.approx(0.25)  # unseen: prior mean 0.5


def test_skip_after_downweights_constant_groups():
    stats = PromptStats()
    _, s = _sampler(stats, [7, 7], weighting="uniform", skip_after=2, skip_keep=0.1)
    stats.update(["p0", "p0"], [[0, 0], [0, 0]])
    assert s.weight(0) == pytest.approx(0.1) and s.weight(1) == 1.0
    stats.update(["p0"], [[1, 0]])  # a mixed group resets the streak
    assert s.weight(0) == 1.0


def test_ladder_unlocks_next_length():
    stats = PromptStats()
    prompts, s = _sampler(
        stats,
        [7, 7, 8, 10],
        weighting="uniform",
        ladder={"start": 7, "threshold": 0.5, "min_groups": 2},
    )
    for _ in range(5):
        assert all(s.lengths[i] == 7 for i in s.next_batch())
    assert s.weight(2) == 0.0
    stats.update(["p0", "p1"], [[1, 1, 0], [1, 1, 0]])
    s.next_batch()
    assert s.max_len == 8 and s.weight(2) > 0 and s.weight(3) == 0.0
    assert stats.summary()["curriculum/max_length"] == 8.0


def test_fallback_stays_within_ladder_when_everything_is_dropped():
    stats = PromptStats()
    prompts, s = _sampler(
        stats,
        [7, 7, 8, 8],
        weighting="uniform",
        drop_solved_after=1,
        ladder={"start": 7, "threshold": 1.1, "min_groups": 1},  # never advances
    )
    stats.update(["p0", "p1"], [[1, 1, 1], [1, 1, 1]])  # both length-7 prompts dropped as solved
    assert s.weight(0) == s.weight(1) == 0.0
    for _ in range(5):
        assert all(s.lengths[i] == 7 for i in s.next_batch())  # never the locked length 8


def test_easy_frac():
    stats = PromptStats()
    prompts = [f"p{i}" for i in range(20)]
    lengths = [3] * 10 + [9] * 10
    s = AdaptiveRepeatSampler(
        prompts, lengths, stats, 1, 10, weighting="uniform", easy_frac=0.3, easy_max=6
    )
    batch = s.next_batch()
    assert sum(lengths[i] <= 6 for i in batch) >= 3 and len(set(batch)) == 10


# ------------------------------------------------------------------ reward, EI helpers, config


def test_unparseable_reward_variant():
    from tokenizer import Tokenizer

    ft = Tokenizer("abs")
    good = ft.encode_proof(VALID)
    wrong = ft.encode_proof("N1 Q : IMPE N9 N9 ; QED")  # parses, fails the check
    ids = [good, wrong, good[:-3]]  # the last is truncated before QED: a parse failure
    base = ProofReward()([PROMPT] * 3, [None] * 3, completion_ids=ids)
    fn = ProofReward(unparseable_reward=-0.1)
    shaped = fn([PROMPT] * 3, [None] * 3, completion_ids=ids)
    assert base == [1.0, 0.0, 0.0]
    assert shaped == [1.0, 0.0, -0.1]
    assert fn.last_stats["reward/invalid_frac"] == pytest.approx(2 / 3)


def test_select_verified_and_tokenize():
    comps = [[[1, 2, 5], [1, 2, 5], [3, 3, 3, 5], [9]], [[4]]]
    judged = [
        [
            ("A", True, "ok", 2, 3),
            ("A ", True, "ok", 2, 3),
            ("B", True, "ok", 3, 4),
            ("C", False, "x", 0, 1),
        ],
        [("D", False, "x", 0, 1)],
    ]
    allp = select_verified(["p", "q"], comps, judged, keep="all", max_per_prompt=4)
    assert [r["completion"] for r in allp] == ["A", "B"]
    short = select_verified(["p", "q"], comps, judged, keep="shortest")
    assert [r["completion"] for r in short] == ["A"]
    from hf_wrapper import build_tokenizer

    tok = build_tokenizer("abs")
    rows = tokenize_records([{"prompt": PROMPT, "completion_ids": [35, 5]}], tok)
    assert rows[0]["input_ids"][-2:] == [35, 5]
    assert rows[0]["completion_mask"] == [0] * (len(rows[0]["input_ids"]) - 2) + [1, 1]


def test_config_include_and_checks(tmp_path):
    presets = os.path.join(ndpath.WS3, "configs", "presets")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"include: [{presets}/sparse_pack.yaml, {presets}/maxrl.yaml]\n"
        "train:\n  trainer: {num_generations: 32}\n"
    )
    cfg = ws3_run.load_config(str(cfg_path))
    tr = cfg["train"]["trainer"]
    assert (
        tr["num_generations"] == 32 and tr["loss_type"] == "dr_grpo" and tr["epsilon_high"] == 0.28
    )
    assert tr["mask_truncated_completions"] is True
    assert cfg["train"]["advantage"] == {"name": "maxrl"}
    ws3_run.check_train_options(cfg)
    cfg["train"]["reward"] = {"unparseable_reward": -0.1}
    with pytest.raises(SystemExit):
        ws3_run.check_train_options(cfg)
    cfg["train"].pop("reward")
    cfg["run"]["algo"] = "rloo"
    with pytest.raises(SystemExit):
        ws3_run.check_train_options(cfg)
    ei = {"run": {"algo": "ei"}, "train": {"reward": {"unparseable_reward": -0.1}}}
    with pytest.raises(SystemExit):  # EI keeps verified proofs only; a shaped reward is meaningless
        ws3_run.check_train_options(ei)


def test_all_presets_parse():
    presets = os.path.join(ndpath.WS3, "configs", "presets")
    for fn in sorted(os.listdir(presets)):
        cfg = ws3_run.read_config(os.path.join(presets, fn))
        assert "train" in cfg, fn


# ------------------------------------------------------------------ CPU smoke (plumbing only)


def _smoke_cfg(tmp_path, extra):
    src = os.path.join(ndpath.WS3, "configs", "smoke_cpu.yaml")
    with open(src) as f:
        text = f.read()
    text = text.replace("out_dir: ../../artifacts/ws3/smoke_cpu", "out_dir: run")
    (tmp_path / "base.yaml").write_text(text)
    p = tmp_path / "config.yaml"
    p.write_text("include: [base.yaml]\n" + extra)
    return str(p)


@pytest.mark.skipif(not HAVE, reason="fork_pin data or checkpoint not fetched")
@pytest.mark.parametrize("adv", ["virtual_positive", "unlikely"])
def test_smoke_grpo_advantage_and_curriculum(tmp_path, adv):
    cfg = _smoke_cfg(
        tmp_path,
        "train:\n"
        "  prompt: {length_weights: {2: 1, 3: 1}, n_prompts: 16}\n"
        "  periodic_eval: {every: 0}\n"
        f"  advantage: {{name: {adv}}}\n"
        "  curriculum: {weighting: p1mp, skip_after: 2}\n",
    )
    ws3_run.main(["--config", cfg, "--stages", "train"])
    rows = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    roll = [r for r in rows if "reward" in r]
    assert roll and all("groups/frac_all_fail" in r and "adv/abs_mean" in r for r in roll)
    assert roll[-1]["rollout/total_completions"] == 8 * len(roll)
    assert "prompts/seen" in roll[-1] and "curriculum/allowed_prompts" in roll[-1]


@pytest.mark.skipif(not HAVE, reason="fork_pin data or checkpoint not fetched")
def test_smoke_ei(tmp_path):
    cfg = _smoke_cfg(
        tmp_path,
        "run: {algo: ei}\n"
        "train:\n"
        "  prompt: {length_weights: {2: 1}, n_prompts: 8}\n"
        "  periodic_eval: {every: 0}\n"
        "  ei: {rounds: 2, k: 4, max_new: 60, batch_size: 32, replay_frac: 0.5,\n"
        "       sft: {per_device_train_batch_size: 8}}\n",
    )
    ws3_run.main(["--config", cfg, "--stages", "train"])
    out = tmp_path / "run"
    rows = read_jsonl(out / "metrics.jsonl")
    assert [r["round"] for r in rows] == [1, 2]
    assert all(r["sft/steps"] > 0 and math.isfinite(r["sft/loss_last"]) for r in rows)
    assert rows[-1]["rollout/total_completions"] == 2 * 8 * 4
    assert (out / "final" / "config.json").exists() and (out / "sft_data" / "round2.jsonl").exists()
    assert len(read_jsonl(out / "completions.jsonl")) == 64
    cov = ws3_run.metric_coverage(rows, "ei")  # plan section 4, EI set (no periodic eval here)
    assert all(v for k, v in cov.items() if "periodic" not in k), cov
