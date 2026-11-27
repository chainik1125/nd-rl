import json

import numpy as np
import torch
from rank1_fixtures import make_checkpoints

# isort: split
import wpass  # noqa: E402  (rank1 dir put on sys.path by rank1_fixtures)


def run(tmp_path, signal, extra=()):
    ck = tmp_path / "models"
    ck.mkdir()
    planted = make_checkpoints(ck, signal=signal)
    out = tmp_path / "w"
    code = wpass.main(
        [
            "--models-dir",
            str(ck),
            "--out",
            str(out),
            "--layers",
            "2",
            "--device",
            "cpu",
            "--k",
            "4",
            *extra,
        ]
    )
    return code, out, planted, ck


def test_rsvd_matches_exact():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(80, 50, generator=g) + 6 * torch.outer(
        torch.randn(80, generator=g), torch.randn(50, generator=g)
    )
    U, S, V = wpass.rsvd(A, 3, iters=6, seed=0)
    ex = torch.linalg.svdvals(A)[:3]
    assert torch.allclose(S, ex, rtol=1e-3)
    assert torch.allclose(
        U @ torch.diag(S) @ V.T,
        torch.linalg.svd(A)[0][:, :3] @ torch.diag(ex) @ torch.linalg.svd(A)[2][:3],
        atol=1e-2 * float(ex[0]),
    )


def test_verdict_bands_and_weighted_median():
    assert (
        wpass.verdict(1.05)[0] == 20
        and wpass.verdict(1.1)[0] == 11
        and wpass.verdict(1.49)[0] == 11
        and wpass.verdict(1.5)[0] == 0
    )
    assert wpass.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 10.0])) == 3.0


def test_planted_signal_gives_run_band_and_files(tmp_path):
    code, out, planted, _ = run(tmp_path, 1.0)
    assert code == 0  # signal or noisy both exit 0
    band = (out / "w_band.txt").read_text().strip()
    assert band in ("signal", "noisy")
    s = json.loads((out / "w_summary.json").read_text())
    assert s["n_matrices"] == 14 and s["exact_ok"] and s["iters"] == 6
    assert s["ratio_d2300_weighted_median"] > 1.5 and band == "signal"
    assert 0 < s["energy_d2300_weighted_median"] <= 1
    assert abs(s["alpha_d2300_weighted_median"] ** -2 - s["energy_d2300_weighted_median"]) < 0.2
    assert "ratio_d100_weighted_median" not in s  # Turn 6 D6: top-1 only for d100, no null
    assert s["cos_delta500_delta2300_total"] > 0.5 and s["gamma_proj_total"] > 0
    assert set(s["dist_ratio_to_s2300"]) == {"3.17", "3.5", "4.6"}
    assert (out / "factors_d2300_k16.safetensors").exists() and (out / "w_matrices.csv").exists()
    from safetensors.torch import load_file

    f = load_file(str(out / "factors_d2300_k16.safetensors"))
    n = "model.layers.0.self_attn.q_proj.weight"
    u = f[n + "::U"][:, 0].numpy()
    assert abs(float(u @ planted[n][0])) > 0.95  # recovered planted direction


def test_pure_noise_stops(tmp_path):
    code, out, _, _ = run(tmp_path, 0.0)
    assert code == 20 and (out / "w_band.txt").read_text().strip() == "stop"


def test_exact_check_failure_exit_and_fallback(tmp_path):
    # an impossible tolerance forces the exact check to fail -> exit 2; --exact-fallback skips the check and exits by the band
    code, out, _, ck = run(tmp_path, 1.0, ["--exact-tol", "1e-12"])
    assert code == 2
    out2 = tmp_path / "w2"
    code2 = wpass.main(
        [
            "--models-dir",
            str(ck),
            "--out",
            str(out2),
            "--layers",
            "2",
            "--device",
            "cpu",
            "--k",
            "4",
            "--exact-fallback",
        ]
    )
    assert code2 == 0


def test_input_error_exit(tmp_path):
    assert (
        wpass.main(
            [
                "--models-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "o"),
                "--layers",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 3
    )
    assert (
        wpass.main(
            [
                "--models-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "o"),
                "--layers",
                "2",
                "--device",
                "cpu",
                "--iters",
                "3",
            ]
        )
        == 3
    )
