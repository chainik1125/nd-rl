"""CPU helper for probing long_rl_20261014 checkpoints. Usage (from any dir):

    ND_ARCH=untied ~/repos/nd-takehome/.venv/bin/python -c "
    import sys; sys.path.insert(0, '<this dir>'); import ndmodel as M
    model, tok = M.load('<ckpt.pt>')
    rows = M.token_logp(model, tok, prompt, proof)   # [(token, logp at T=1, argmax token)] for the proof tokens
    samples = M.sample(model, tok, [rec], k=8)        # per rec: [(decoded ND proof or error tag, lean_ok, raw tokens)]  (T=0.8, Lean-judged)
    "
ND_ARCH must match the checkpoint (untied | looped); set it in the environment BEFORE import.
For looped models set model.n_loops = 1/2/4/8 before calling.
Checkpoints: resume/<arm>_s<seed>/final.pt or pretrain.pt ({'state':..}), or rl_round_NNN.pt (plain state dict).
"""
import os, sys
os.environ.setdefault("ND_FORK", "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3")
sys.path.insert(0, "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014")
import torch  # noqa: E402
torch.set_num_threads(4)
import harness as H  # noqa: E402
import pretrain as P  # noqa: E402


def load(path):
    ck = torch.load(path, map_location="cpu")
    state = ck.get("state", ck)
    tok = P.make_tokenizer(P.TOK_MODE)
    model = P.LoopedGPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD, 0)
    model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
    model.load_state_dict(state)
    model.eval()
    if hasattr(model, "n_loops"):
        model.n_loops = 1
    return model, tok


@torch.no_grad()
def token_logp(model, tok, prompt, proof):
    p, q = tok.encode_prompt(prompt), tok.encode_proof(proof)
    x = torch.tensor([p + q])
    logp = torch.log_softmax(model(x[:, :-1])[0, len(p) - 1:].float(), -1)
    lp = logp.gather(1, x[0, len(p):, None])[:, 0]
    top = logp.argmax(-1)
    return [(tok.itos[t], float(l), tok.itos[int(a)]) for t, l, a in zip(q, lp, top)]  # (token, logp, argmax token)


def sample(model, tok, recs, k=8, seed=0):
    out = H.sample(model, tok, recs, k, seed, H.MAX_NEW)
    # nd is the decoded ND proof, or an error tag like "LEANPARSE ..." / "LEAN ..." when it failed; raw = model text
    return [[(nd, ok, " ".join(tok.itos[i] for i in ids)) for ids, nd, ok in r] for r in out]
