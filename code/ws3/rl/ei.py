"""Expert iteration (rejection-sampling fine-tuning) in TRL, the bar every RL result is compared
against (startup.md exp-05; brainstorm section B).

Each round: sample k proofs per RL prompt from the current model, keep the ones the strict
verifier accepts, and fine-tune on them with TRL's `SFTTrainer` (loss on the proof only).
`run.py` calls `train_ei(cfg)` for `run.algo: ei`, so the baseline, evaluate and report stages
are the same as for GRPO, and `final/` holds the last round's model.

    train:
      prompt: {...}                  same prompt set as GRPO (data.build_prompt_set)
      ei:
        rounds: 3
        k: 32                        samples per prompt per round
        temperature: 1.0
        max_new: 400
        batch_size: 256              generation batch
        keep: all | shortest         per prompt: every distinct verified proof, or only the shortest
        max_per_prompt: 4            cap on distinct proofs kept per prompt per round
        accumulate: false            also train on verified proofs from earlier rounds
        restart_from_base: false     ReST-EM: fine-tune the frozen checkpoint each round instead
                                     of continuing from the last round (sampling still uses the
                                     latest model)
        replay_frac: 0.0             add pre-training records (train.jsonl proofs) equal to this
                                     fraction of the verified set, to protect short lengths
                                     (Robbie's mix_rl_records)
        sft: {...}                   any SFTConfig field; overrides SFT_DEFAULTS

metrics.jsonl gets one line per round, with `rollout/total_completions` counting every sampled
proof, so EI and GRPO can be compared at the same number of samples. completions.jsonl gets
every judged sample, and sft_data/round<r>.jsonl the exact training set of each round.
"""

from __future__ import annotations

import json
import os
import random
import time

import torch

# isort: split
from data import banned_keys, build_prompt_set, key_of, load_pool, stride
from eval.evaluator import generate, judge, utc_now
from eval.reward import is_unparseable
from hf_wrapper import load_pretrained_wrapper, save_base
from metrics_callback import PeriodicEvalCallback
from tokenizer import Tokenizer  # fork_pin

SFT_DEFAULTS = dict(
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    lr_scheduler_type="constant",
    warmup_steps=0,
    max_grad_norm=1.0,
    completion_only_loss=True,
    loss_type="nll",  # TRL 1.13's default chunked_nll calls an HF backbone our wrapper lacks
    packing=False,
    max_length=1024,
    report_to="none",
    save_strategy="no",
    logging_steps=1,
    gradient_checkpointing=False,
    remove_unused_columns=True,
)
EI_DEFAULTS = dict(
    rounds=3,
    k=32,
    temperature=1.0,
    max_new=400,
    batch_size=256,
    keep="all",
    max_per_prompt=4,
    accumulate=False,
    restart_from_base=False,
    replay_frac=0.0,
)


def select_verified(prompts, comps, judged, keep="all", max_per_prompt=4):
    """Training records from one round's samples: distinct verified proofs per prompt.

    Returns a list of {prompt, completion, completion_ids}. `keep: shortest` keeps only the
    proofs with the fewest tokens for each prompt. Pure function, unit-tested.
    """
    if keep not in ("all", "shortest"):
        raise ValueError(f"keep must be all or shortest, got {keep!r}")
    out = []
    for prompt, ids_list, js in zip(prompts, comps, judged, strict=True):
        seen, cands = set(), []
        for ids, (body, ok, _reason, _n_lines, n_tok) in zip(ids_list, js, strict=True):
            norm = " ".join(body.split())
            if ok and norm not in seen:
                seen.add(norm)
                cands.append((n_tok, body, list(ids)))
        if not cands:
            continue
        if keep == "shortest":
            best = min(c[0] for c in cands)
            cands = [c for c in cands if c[0] == best]
        for _n, body, ids in cands[:max_per_prompt]:
            out.append({"prompt": prompt, "completion": body, "completion_ids": ids})
    return out


def replay_records(n, seed, ft: Tokenizer, banned):
    """n pre-training records (prompt + reference proof) from train.jsonl, skipping any renaming
    class that is banned for training."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    pool = [r for r in load_pool("train") if key_of(r) not in banned]
    picks = rng.sample(pool, min(n, len(pool)))
    return [
        {
            "prompt": r["prompt"],
            "completion": r["proof"],
            "completion_ids": ft.encode_proof(r["proof"]),
        }
        for r in picks
    ]


def tokenize_records(records, tok):
    """Pre-tokenized SFT rows: prompt ids exactly as generation sees them, then the proof ids
    (ending in QED, the end token). completion_mask marks the proof tokens for the loss."""
    rows = []
    for r in records:
        p = tok(r["prompt"])["input_ids"]
        c = list(r["completion_ids"])
        rows.append({"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c)})
    return rows


def sft_round(model, tok, records, sft_kw, out_dir, seed, device):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    kw = dict(SFT_DEFAULTS)
    kw.update(sft_kw or {})
    kw["output_dir"] = os.path.join(out_dir, "trainer")
    kw.setdefault("seed", seed)
    kw.setdefault("use_cpu", device == "cpu")
    kw.setdefault("bf16", device == "cuda")
    ds = Dataset.from_list(tokenize_records(records, tok))
    trainer = SFTTrainer(model=model, args=SFTConfig(**kw), train_dataset=ds, processing_class=tok)
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return trainer.model, {
        "sft/steps": trainer.state.global_step,
        "sft/loss_first": losses[0] if losses else None,
        "sft/loss_last": losses[-1] if losses else None,
    }


def train_ei(cfg, environment_record, write_json):
    run, tr = cfg["run"], cfg["train"]
    ei = dict(EI_DEFAULTS)
    ei.update(tr.get("ei", {}))
    out_dir = run["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(run["seed"])
    t0 = time.time()

    model, tok = load_pretrained_wrapper(device=run["device"])
    base_dir = save_base(model, tok, os.path.join(out_dir, "hf_base"))
    ft = Tokenizer(tok.tok_mode)
    p = tr["prompt"]
    prompt_recs = build_prompt_set(p["length_weights"], int(p["n_prompts"]), seed=run["seed"])
    with open(os.path.join(out_dir, "prompts.jsonl"), "w") as f:
        for r in prompt_recs:
            f.write(json.dumps(r) + "\n")
    prompts = [r["prompt"] for r in prompt_recs]
    banned = banned_keys() if ei["replay_frac"] > 0 else set()

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    comp_path = os.path.join(out_dir, "completions.jsonl")
    if os.path.exists(comp_path):
        os.remove(comp_path)
    sft_dir = os.path.join(out_dir, "sft_data")
    os.makedirs(sft_dir, exist_ok=True)
    periodic = None
    ev = tr.get("periodic_eval", {})
    if ev.get("every"):
        held = stride(load_pool("heldout"), int(ev.get("n_heldout", 100)))
        longp = stride(load_pool("rl_targets"), int(ev.get("n_long", 50)))
        periodic = PeriodicEvalCallback(
            metrics_path,
            tok,
            held + longp,
            every=1,
            max_new=int(ev.get("max_new", 400)),
            batch_size=int(ev.get("batch_size", 128)),
            dump=os.path.join(out_dir, "eval_periodic.jsonl"),
        )
        periodic._run(model, 0)
    write_json(
        os.path.join(out_dir, "run_manifest.json"),
        environment_record(
            cfg,
            {
                "algo": "ei",
                "hf_base": base_dir,
                "n_prompts": len(prompts),
                "ei": ei,
                "prompt_lengths": sorted({r["length"] for r in prompt_recs}),
            },
        ),
    )

    total, kept_all = 0, []
    for rnd in range(1, int(ei["rounds"]) + 1):
        ts = time.time()
        comps = generate(
            model,
            tok,
            prompts,
            k=int(ei["k"]),
            temperature=float(ei["temperature"]),
            max_new=int(ei["max_new"]),
            batch_size=int(ei["batch_size"]),
            seed=run["seed"] * 1000 + rnd,
        )
        gen_secs = time.time() - ts
        judged = judge(prompts, comps, tok.tok_mode)
        n = sum(len(js) for js in judged)
        total += n
        n_ok = sum(j[1] for js in judged for j in js)
        n_unp = sum(is_unparseable(j[2]) for js in judged for j in js)
        with open(comp_path, "a") as f:
            for prompt, js in zip(prompts, judged, strict=True):
                for body, ok, reason, n_lines, n_tok in js:
                    f.write(
                        json.dumps(
                            {
                                "round": rnd,
                                "prompt": prompt,
                                "completion": body,
                                "reward": float(ok),
                                "reason": reason,
                                "n_lines": n_lines,
                                "n_tokens": n_tok,
                            }
                        )
                        + "\n"
                    )
        kept = select_verified(prompts, comps, judged, ei["keep"], int(ei["max_per_prompt"]))
        kept_all = kept_all + kept if ei["accumulate"] else kept
        train_recs = list(kept_all)
        train_recs += replay_records(
            round(ei["replay_frac"] * len(kept_all)), run["seed"] * 1000 + rnd, ft, banned
        )
        with open(os.path.join(sft_dir, f"round{rnd}.jsonl"), "w") as f:
            for r in train_recs:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
        row = {
            "step": rnd,
            "round": rnd,
            "utc": utc_now(),
            "time": time.time(),
            "reward": n_ok / max(n, 1),
            "reward/invalid_frac": 1 - n_ok / max(n, 1),
            "reward/unparseable_frac": n_unp / max(n, 1),
            "ei/prompts_solved_frac": sum(any(j[1] for j in js) for js in judged) / len(prompts),
            "ei/kept_this_round": len(kept),
            "ei/train_records": len(train_recs),
            "rollout/total_completions": float(total),
            "rollout/completions_per_sec": n / max(gen_secs, 1e-9),
        }
        if train_recs:
            if ei["restart_from_base"]:
                model, _ = load_pretrained_wrapper(device=run["device"])
            model, sft_stats = sft_round(
                model, tok, train_recs, ei.get("sft"), out_dir, run["seed"] + rnd, run["device"]
            )
            row.update(sft_stats)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"EI round {rnd}: {n_ok}/{n} verified, {len(train_recs)} SFT records", flush=True)
        if periodic is not None:
            periodic._run(model, rnd)

    final_dir = os.path.join(out_dir, "final")
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    summary = {
        "utc_end": utc_now(),
        "secs": time.time() - t0,
        "rounds": int(ei["rounds"]),
        "total_completions": total,
    }
    write_json(os.path.join(out_dir, "train_summary.json"), summary)
    return summary
