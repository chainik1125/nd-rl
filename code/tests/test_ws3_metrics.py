"""Plan section 5 metrics beyond pass@k: avg solve rate, reliability, frontier, per-problem
comparison, valid-prefix progress, self-knowledge, known-proof log-probability, variants."""

import itertools
import math

import ndpath  # noqa: F401
import pytest
import torch

# isort: split
from data import canon_key, thm_of_prompt
from eval.evaluator import aggregate, known_proof_logprob, sequence_logprobs, textbook_avg_table
from eval.metrics import auroc, compare_problems, frontier, k_at, mean_ci, per_problem, self_knowledge
from eval.progress import valid_prefix
from eval.reward import verify_body
from eval.variants import consistency, make_variants, variant_records
from tokenizer import Tokenizer  # fork_pin

PROMPT = "THM ( P > Q ) , P SEQ Q PRF"
VALID = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"
BOX_PROMPT = "THM SEQ ( P > P ) PRF"
BOX_VALID = "N1 | P : AS ; N2 ( P > P ) : IMPI N1 N1 ; QED"


def _j(ok, n_tokens=10, body="body"):
    return (body, ok, "ok" if ok else "wrong", 3 if ok else 0, n_tokens)


# ------------------------------------------------------------------ avg solve rate and friends


def test_mean_ci_is_never_narrower_than_wilson():
    m, (lo, hi) = mean_ci([0.0] * 10)
    assert m == 0.0 and lo == 0.0 and hi > 0.2  # all-zero set still has width
    m, (lo, hi) = mean_ci([0.25, 0.75])
    assert m == 0.5 and lo < 0.5 < hi
    assert mean_ci([]) == (0.0, [0.0, 0.0])


def test_aggregate_avg_is_mean_of_per_problem_rates():
    recs = [{"prompt": "a", "n_lines": 3}, {"prompt": "b", "n_lines": 3}, {"prompt": "c", "n_lines": 4}]
    judged = [
        [_j(True), _j(True), _j(True), _j(False)],  # p = 0.75
        [_j(True), _j(False), _j(False), _j(False)],  # p = 0.25
        [_j(False)] * 4,  # p = 0
    ]
    res = aggregate(recs, judged, k=4)
    assert res["by_length"][3]["avg"] == pytest.approx(0.5)
    assert res["by_length"][3]["reliable50"] == pytest.approx(0.5)
    assert res["by_length"][3]["reliable90"] == 0.0
    assert res["by_length"][3]["solved"] == 2  # any-of-k stays as the secondary number
    assert res["overall"]["avg"] == pytest.approx(1 / 3)
    assert res["frontier_avg50"] == 3  # length 3 at 0.5, length 4 at 0
    assert res["by_length"][3]["k50"] == 1  # pass@1 already 0.5
    assert res["by_length"][4]["k50"] is None


def test_aggregate_slices_carry_expected_and_reliable_counts():
    recs = [
        {"prompt": "a", "reference_lines": 7, "slice": "dev58"},
        {"prompt": "b", "reference_lines": 3, "slice": "dev58"},
        {"prompt": "c", "slice": "train14"},
    ]
    judged = [[_j(True), _j(True)], [_j(True), _j(False)], [_j(False), _j(False)]]
    res = aggregate(recs, judged, k=2)
    assert res["by_slice"]["dev58"]["expected_solved"] == pytest.approx(1.5)
    assert res["by_slice"]["dev58"]["reliable50_count"] == 2
    assert res["by_slice"]["all"]["avg"] == pytest.approx(0.5)
    assert res["ref_len_ge7"]["expected_solved"] == pytest.approx(1.0)
    table = textbook_avg_table({"control": res})
    assert "| dev58 | 0.750 [" in table and "1.5 / 2 of 2" in table


def test_frontier_needs_every_shorter_length():
    by = {2: {"avg": 0.9}, 3: {"avg": 0.4}, 4: {"avg": 0.8}, -1: {"avg": 0.0}}
    assert frontier(by) == 2
    assert frontier({5: {"avg": 0.1}}) == 0
    assert k_at({1: 0.1, 2: 0.3, 4: 0.6}) == 4 and k_at({1: 0.1}) is None


def test_auroc_matches_pairwise_count():
    scores = [0.1, 0.4, 0.4, 0.8, 0.2, 0.9]
    labels = [0, 1, 0, 1, 0, 1]
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    brute = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)) / (len(pos) * len(neg))
    assert auroc(scores, labels) == pytest.approx(brute)
    assert auroc([1, 2], [1, 1]) is None


def test_self_knowledge_top1_picks_most_likely_sample():
    judged = [[_j(False), _j(True)], [_j(True), _j(False)]]
    logprobs = [[-5.0, -1.0], [-4.0, -2.0]]  # first problem: most likely is right; second: wrong
    sk = self_knowledge(logprobs, judged)
    assert sk["likelihood_top1"] == 0.5
    assert sk["n_verified"] == 2 and sk["n_samples"] == 4


def test_compare_problems_separates_new_capability_from_sharpening():
    rows = lambda prompt, L, oks: [{"prompt": prompt, "length": L, "ok": o} for o in oks]  # noqa: E731
    base = per_problem(rows("a", 7, [0] * 8) + rows("b", 7, [1, 0, 0, 0]) + rows("c", 8, [1] * 4)
                       + rows("d", 8, [0] * 4))
    rl = per_problem(rows("a", 7, [1, 1, 1, 0]) + rows("b", 7, [1, 1, 0, 0]) + rows("c", 8, [0] * 4)
                     + rows("d", 8, [1, 0, 0, 0]))
    cmp = compare_problems(base, rl)
    o = cmp["overall"]
    assert o["n"] == 4
    assert o["new_capability"] == 1  # a: 0 -> 0.75
    assert o["new_any"] == 2  # a and d
    assert o["sharpened"] == 1  # b: 0.25 -> 0.5
    assert o["lost"] == 1  # c: 1 -> 0
    assert cmp["by_length"][7]["new_capability"] == 1


# ------------------------------------------------------------------ valid prefix


def test_valid_prefix_on_accepted_proofs():
    for prompt, body in ((PROMPT, VALID), (BOX_PROMPT, BOX_VALID)):
        assert verify_body(prompt, body)[0]
        vp = valid_prefix(prompt, body)
        assert vp == {"n_lines": vp["n_lines"], "n_valid": vp["n_lines"], "first_error": None,
                      "goal_derived": True}


def test_valid_prefix_stops_at_first_bad_line():
    bad = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 P : IMPE N1 N2 ; N4 Q : IMPE N1 N2 ; QED"
    vp = valid_prefix(PROMPT, bad)
    assert vp["n_valid"] == 2 and vp["first_error"] == "rule check failed: IMPE (line 3)"
    # the verifier agrees on the line
    ok, reason, _ = verify_body(PROMPT, "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N1 ; QED")
    vp = valid_prefix(PROMPT, "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N1 ; QED")
    assert not ok and "(line 3)" in reason and vp["n_valid"] == 2


def test_valid_prefix_wrong_goal_truncated_junk_and_premises():
    wrong_goal = "N1 ( P > Q ) : PR ; N2 P : PR ; QED"
    vp = valid_prefix(PROMPT, wrong_goal)
    assert vp["n_valid"] == 2 and vp["first_error"] is None and not vp["goal_derived"]
    truncated = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IM"
    vp = valid_prefix(PROMPT, truncated)
    assert vp["n_valid"] == 2 and vp["first_error"] == "unparseable line"
    assert valid_prefix(PROMPT, "")["first_error"] == "empty proof"
    assert valid_prefix(PROMPT, "garbage tokens ;")["n_valid"] == 0
    swapped = "N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; QED"
    assert valid_prefix(PROMPT, swapped)["n_valid"] == 0  # premises out of declared order
    missing = "N1 ( P > Q ) : PR ; N2 Q : R N1 ; QED"
    assert valid_prefix(PROMPT, missing)["n_valid"] == 1  # premise block ended early
    forward = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N4 ; N4 P : R N2 ; QED"
    assert valid_prefix(PROMPT, forward)["n_valid"] == 2  # strict: citations must point back
    assert valid_prefix("not a prompt", VALID)["first_error"] == "malformed prompt"


def test_aggregate_reports_valid_prefix_of_failures():
    recs = [{"prompt": PROMPT, "n_lines": 4}]
    judged = [[(VALID, True, "ok", 3, 12), ("N1 ( P > Q ) : PR ; N2 P : PR ; QED", False, "x", 2, 9)]]
    res = aggregate(recs, judged, k=2)
    assert res["by_length"][4]["valid_prefix_failed"] == pytest.approx(0.5)
    assert res["by_length"][4]["proof_len_ratio"] == pytest.approx(0.75)


# ------------------------------------------------------------------ log-probabilities


class _Oracle(torch.nn.Module):
    """Logit 10 on the true next token, 0 elsewhere."""

    def __init__(self, vocab=99):
        super().__init__()
        self.vocab = vocab
        self.config = type("C", (), {"tok_mode": "abs"})()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, use_cache=False):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        logits[:, :-1].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 10.0)
        return type("O", (), {"logits": logits})()


def test_sequence_logprobs_counts_only_completion_tokens():
    ft = Tokenizer("abs")
    per_tok = 10 - math.log(math.exp(10) + 98)
    ids = [ft.encode_proof(VALID), ft.encode_proof("N1 P : PR ; QED")]
    lps = sequence_logprobs(_Oracle(), ft, [PROMPT, "THM P SEQ P PRF"], ids, batch_size=2)
    assert lps[0] == pytest.approx(per_tok * len(ids[0]), rel=1e-4)
    assert lps[1] == pytest.approx(per_tok * len(ids[1]), rel=1e-4)


def test_known_proof_logprob_takes_best_proof_per_prompt():
    ft = Tokenizer("abs")
    longer = "N1 ( P > Q ) : PR ; N2 P : PR ; N3 P : R N2 ; N4 Q : IMPE N1 N3 ; QED"
    res = known_proof_logprob(_Oracle(), ft, {PROMPT: [VALID, longer, VALID]}, {PROMPT: 3})
    assert res["n_prompts"] == 1 and res["n_proofs"] == 2  # duplicates collapse
    per_tok = 10 - math.log(math.exp(10) + 98)
    assert res["by_length"][3]["best_logprob"] == pytest.approx(per_tok * len(ft.encode_proof(VALID)), rel=1e-4)


# ------------------------------------------------------------------ variants


def test_make_variants_are_distinct_equivalent_rewrites():
    vs = make_variants(PROMPT, 5, seed=0)
    assert len(vs) == 5 and len({v for v, _ in vs}) == 5 and PROMPT not in {v for v, _ in vs}
    for v, kind in vs:
        assert kind in ("rename", "reorder", "rename+reorder")
        if kind == "rename":  # a renaming stays in the original's renaming class
            assert canon_key(thm_of_prompt(v)) == canon_key(thm_of_prompt(PROMPT))
        assert len(v.split()) == len(PROMPT.split())
    assert make_variants(PROMPT, 5, seed=0) == vs  # deterministic
    assert all(k == "rename" for _, k in make_variants(BOX_PROMPT, 3))  # no premises to reorder


def test_variant_records_drop_training_classes_and_consistency():
    rec = {"name": "t1", "prompt": PROMPT, "reference_lines": 3, "slice": "dev58"}
    recs, dropped = variant_records([rec], 3, seed=0)
    assert dropped == 0 and len(recs) == 4 and recs[0]["variant"] == "original"
    banned = {canon_key(thm_of_prompt(PROMPT))}
    recs2, dropped2 = variant_records([rec], 3, seed=0, exclude_keys=banned)
    assert all(r["variant"] in ("original", "reorder", "rename+reorder") for r in recs2)
    assert dropped2 == sum(k == "rename" for _, k in make_variants(PROMPT, 3, 0))
    judged = [[_j(True), _j(True)], [_j(True), _j(False)], [_j(True), _j(True)], [_j(False), _j(False)]]
    c = consistency(recs, judged)
    row = c["per_problem"][0]
    assert row["mean_p"] == pytest.approx(0.625) and row["min_p"] == 0.0
    assert row["consistent_reliable"] is False
    assert row["original_gap"] == pytest.approx(1.0 - 0.5)
    assert c["by_length"][3]["n_problems"] == 1
