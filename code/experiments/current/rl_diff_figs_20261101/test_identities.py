"""Properties the figures rely on. Runs on CPU with two real checkpoints (ND_CKPTS)."""
import torch

import common as C
from run_sae import SAE, Edit
from run_svd import energy_kept, svds, truncated


def test_svd_endpoints():
    model, _, _ = C.load(0, 0)
    s0 = dict(model.state_dict())
    s1 = torch.load(C.ckpt_path(0, 1), map_location=C.DEVICE)["state"]
    dec = svds(s0, s1)
    assert len(dec) == 26  # Charles's count: qkv, proj, fc1, fc2 per block, plus emb and head
    zero = truncated(s0, s1, dec, 0, "base")
    assert all(torch.equal(zero[k], s0[k]) for k in s0)
    full = truncated(s0, s1, dec, 10_000, "full")
    for k in s0:
        assert torch.allclose(full[k].float(), s1[k].float(), atol=2e-5), k
    assert abs(energy_kept(dec, 10_000) - 1) < 1e-6 and energy_kept(dec, 0) == 0
    assert energy_kept(dec, 4) < energy_kept(dec, 16)


def _sae():
    torch.manual_seed(0)
    return SAE(8, 16, 2, torch.zeros(8), 2.0)


def test_identity_edit_is_noop():
    sae, x = _sae(), torch.randn(5, 3, 8)
    one, zero = torch.ones(16), torch.zeros(16)
    assert torch.allclose(Edit(sae, torch.arange(16), one, zero)(None, None, x), x)


def test_map_edit_matches_decoded_change():
    sae, x = _sae(), torch.randn(4, 8)
    S = torch.tensor([1, 3])
    a, b = 2 * torch.ones(16), 0.5 * torch.ones(16)
    y = Edit(sae, S, a, b)(None, None, x)
    z = sae.encode(x)[:, S]
    want = x + ((a[S] - 1) * z + b[S]) @ sae.dec[S] * sae.scale
    assert torch.allclose(y, want, atol=1e-6)


def test_vector_edit():
    x = torch.zeros(1, 2, 8)
    assert torch.equal(Edit(vec=torch.ones(8))(None, None, x), torch.ones(1, 2, 8))
