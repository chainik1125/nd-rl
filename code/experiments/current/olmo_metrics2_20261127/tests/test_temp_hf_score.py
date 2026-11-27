import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
torch.set_num_threads(1)

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import hf_score as hs  # noqa: E402


def tiny(window=4, layers=4, dtype=torch.float32):
    from transformers import Olmo3Config, Olmo3ForCausalLM

    cfg = Olmo3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=window,
        layer_types=[
            "sliding_attention" if (i + 1) % 4 else "full_attention" for i in range(layers)
        ],
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m = Olmo3ForCausalLM(cfg).to(dtype).eval()
    return m


def direct(model, prompt, resp):
    ids = torch.tensor([prompt + resp])
    with torch.no_grad():
        lg = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    return torch.stack([lp[len(prompt) + r - 1, resp[r]] for r in range(len(resp))])


def seqs():
    g = torch.Generator().manual_seed(1)
    p = torch.randint(2, 97, (9,), generator=g).tolist()
    r = torch.randint(2, 97, (23,), generator=g).tolist()
    return p, r


@pytest.mark.parametrize("chunk", [1, 5, 8, 100])
def test_chunked_logprobs_match_direct_log_softmax(chunk):
    m = tiny()
    p, r = seqs()
    got = hs.token_logprobs(m, p, r, chunk=chunk)
    ref = direct(m, p, r)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_sliding_window_is_applied_and_reported():
    p, r = seqs()
    short = tiny(window=4)
    long = tiny(window=1000)
    long.load_state_dict(short.state_dict())
    a = hs.token_logprobs(short, p, r, 8)
    b = hs.token_logprobs(long, p, r, 8)
    assert not torch.allclose(a, b, atol=1e-6), (
        "window of 4 and of 1000 must differ on a 32-token sequence"
    )
    rep = hs.sliding_report(short)
    assert rep["sliding_window"] == 4 and rep["n_sliding_layers"] == 3 and rep["n_layers"] == 4
    assert rep["attn_implementation"] in ("sdpa", "eager", "flash_attention_2")


def test_summarize_segments_and_flags():
    lps = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    row = {
        "id": "x",
        "pid": "p",
        "group": "dpo",
        "prompt_ids": [5, 6],
        "response_ids": [7, 8, 9, 1],
        "split_at": 3,
    }
    s = hs.summarize(row, lps, eos_id=1)
    assert (
        s["total_lp"] == -10
        and s["think_lp"] == -6
        and s["answer_lp"] == -4
        and s["think_n"] == 3
        and s["answer_n"] == 1
    )
    assert s["ends_eos"] and s["finite"] and abs(s["mean_lp"] + 2.5) < 1e-12
    row["split_at"] = -1
    s = hs.summarize(row, lps, 1)
    assert s["think_n"] == 4 and s["answer_n"] == 0 and s["answer_lp"] == 0


def test_run_is_resumable_deterministic_and_selftested(tmp_path):
    m = tiny()
    p, r = seqs()
    rows = [
        {
            "id": f"s{i}",
            "pid": "p",
            "group": "dpo",
            "prompt_ids": p,
            "response_ids": r[: 10 + i],
            "split_at": 5,
        }
        for i in range(4)
    ]
    out = tmp_path / "o.jsonl"
    n = hs.run(m, rows[:2], str(out), 8, 1, selftest_n=2, log=lambda *_: None)
    assert n == 2
    n = hs.run(m, rows, str(out), 8, 1, selftest_n=2, log=lambda *_: None)
    assert n == 2  # only the two missing sequences
    got = [json.loads(ln) for ln in open(out)]
    assert [g["id"] for g in got] == ["s0", "s1", "s2", "s3"]
    assert (
        got[0]["selftest_identical"]
        and got[0]["selftest_path"] == "exact"
        and got[0]["selftest_finite"]
        and all(g["finite"] for g in got)
    )
    ref = direct(m, p, r[:10])
    assert abs(got[0]["total_lp"] - float(ref.double().sum())) < 1e-4


def test_bf16_close_to_float32():
    p, r = seqs()
    m32 = tiny()
    m16 = tiny(dtype=torch.bfloat16)
    m16.load_state_dict({k: v.to(torch.bfloat16) for k, v in m32.state_dict().items()})
    a = hs.token_logprobs(m32, p, r, 8)
    b = hs.token_logprobs(m16, p, r, 8)
    assert (a - b).abs().mean() < 0.1


def test_check_sliding_cli_fails_on_wrong_expectation(tmp_path):
    m = tiny()
    d = tmp_path / "m"
    m.save_pretrained(d)
    man = tmp_path / "m.jsonl"
    p, r = seqs()
    man.write_text(
        json.dumps(
            {
                "id": "a",
                "pid": "p",
                "group": "dpo",
                "prompt_ids": p,
                "response_ids": r,
                "split_at": -1,
            }
        )
        + "\n"
    )
    rc = hs.main(
        [
            "--model",
            str(d),
            "--manifest",
            str(man),
            "--out",
            str(tmp_path / "o.jsonl"),
            "--dtype",
            "float32",
            "--check-sliding",
            "--expect-sliding-window",
            "4096",
            "--expect-sliding-layers",
            "48",
        ]
    )
    assert rc == 4
    rc = hs.main(
        [
            "--model",
            str(d),
            "--manifest",
            str(man),
            "--out",
            str(tmp_path / "o.jsonl"),
            "--dtype",
            "float32",
            "--check-sliding",
            "--expect-sliding-window",
            "4",
            "--expect-sliding-layers",
            "3",
        ]
    )
    assert rc == 0


def test_selftest_fallback_below_and_above_tolerance(tmp_path, monkeypatch):
    """Nondeterminism of 5e-4 passes via the fallback (exit 0); 5e-3 fails both ways (exit 5)."""
    m = tiny()
    p, r = seqs()
    real = hs.token_logprobs
    for eps, want_rc, want_path in ((5e-4, 0, "fallback"), (5e-3, 5, "failed")):
        calls = {"n": 0}

        def noisy(model, prompt, resp, chunk=2048, eps=eps, calls=calls):
            out = real(model, prompt, resp, chunk)
            calls["n"] += 1
            if calls["n"] % 2 == 0:  # second pass of a selftest pair
                out = out.clone()
                out[3] += eps
            return out

        monkeypatch.setattr(hs, "token_logprobs", noisy)
        d = tmp_path / f"m{want_rc}"
        m.save_pretrained(d)
        man = tmp_path / f"m{want_rc}.jsonl"
        man.write_text(
            json.dumps(
                {
                    "id": "a",
                    "pid": "p",
                    "group": "dpo",
                    "prompt_ids": p,
                    "response_ids": r,
                    "split_at": -1,
                }
            )
            + "\n"
        )
        out = tmp_path / f"o{want_rc}.jsonl"
        rc = hs.main(
            [
                "--model",
                str(d),
                "--manifest",
                str(man),
                "--out",
                str(out),
                "--dtype",
                "float32",
                "--selftest-n",
                "1",
            ]
        )
        row = json.loads(out.read_text().splitlines()[0])
        assert rc == want_rc and row["selftest_path"] == want_path and not row["selftest_identical"]
        assert abs(row["selftest_max_abs_diff"] - eps) < 1e-6 and row["selftest_fallback_pass"] == (
            want_rc == 0
        )
    assert hs.SELFTEST_TOL == 1e-3
