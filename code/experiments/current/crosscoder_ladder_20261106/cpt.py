"""The control arm: expert iteration's fine-tune, with nothing from RL. From the base, each round
runs `pretrain.finetune` (Muon on the matrices, AdamW on the rest, FT_RECORDS records in
token-budget batches, the same warmup and cosine) on a fresh random draw of FT_RECORDS
pretraining records instead of EI's mix of found proofs and replayed records. Checkpoints every
5 rounds and at the end, named as the EI runs' are, with a copy of the mirrored EI run's
summary.json so the ladder puts round r of this arm at round r's RL clock.

  python cpt.py --base v4_untied-ei_s0:pretrain --mirror v4_untied-ei_s0 --rounds 36 \\
      --out CL_CKPTS/cpt_s0
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import common as C
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v4_untied-ei_s0:pretrain")
    ap.add_argument(
        "--mirror",
        default="v4_untied-ei_s0",
        help="the EI run whose round count and clock this copies",
    )
    ap.add_argument(
        "--rounds", type=int, default=None, help="default: the mirrored run's EI round count"
    )
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.CKPTS / "cpt_s0"))
    ap.add_argument(
        "--records", type=int, default=None, help="override FT_RECORDS (CPU smoke only)"
    )
    a = ap.parse_args()
    if a.records:
        C.agent.FT_RECORDS = a.records
    out = Path(a.out) if a.out.startswith("/") else C.CKPTS / a.out
    out.mkdir(parents=True, exist_ok=True)
    mirror = json.loads((C.CKPTS / a.mirror / "summary.json").read_text())
    n_rounds = a.rounds or len(C.ei_rounds(mirror))
    run, stem = a.base.split(":")
    model, tok = C.load_model(C.ckpt_path(run, stem))
    train_recs = C.read(C.DATA / "pretrain.jsonl.gz")
    rng = random.Random(a.seed)
    log = []
    t0 = time.time()
    shutil.copy(C.CKPTS / a.mirror / "summary.json", out / "summary.json")
    for r in range(1, n_rounds + 1):
        if (out / f"rl_round_{r:03d}.pt").exists():  # resumable only at saved rounds
            model.load_state_dict(torch.load(out / f"rl_round_{r:03d}.pt", map_location=C.DEV))
            continue
        recs = rng.sample(train_recs, min(C.agent.FT_RECORDS, len(train_recs)))
        data = [(tok.encode_prompt(x["prompt"]), tok.encode_proof(x["proof"])) for x in recs]
        loss, steps = C.agent.finetune(model, tok, data, rng, C.DEV)
        log.append({"round": r, "loss": loss, "steps": steps, "secs": time.time() - t0})
        print(
            f"cpt round {r}: loss {loss:.4f} in {steps} steps, {time.time() - t0:.0f}s", flush=True
        )
        if r % a.every == 0:
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                out / f"rl_round_{r:03d}.pt",
            )
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "stage": "final",
            "arm": "cpt",
            "base": a.base,
            "mirror": a.mirror,
        },
        out / "final.pt",
    )
    (out / "cpt_log.json").write_text(json.dumps(log))
    print("CPT_TRAINED", flush=True)


if __name__ == "__main__":
    main()
