"""Live same-model attempt generation and replay likelihoods.

All generated raw token likelihoods belong to the episode policy gradient.
Effective executed tokens form the next model-visible history; verifier diagnostics
remain private. The first attempt uses the original state prompt exactly.
"""

import time

import torch

PROTOCOL_VERSION = "v2_raw_truncated_effective_actions"


def history_prefix(tok, attempts):
    tokens = []
    for i, attempt in enumerate(attempts):
        tokens.extend([tok.stoi["<st>"], tok.stoi[f"n{i + 1}"]])
        tokens.extend(attempt["effective_tokens"])
        tokens.append(tok.eos)
    return tokens


def token_logprobs(base, prompt, raw, device, temperature=0.8, return_base=False):
    if not prompt or not raw or temperature <= 0:
        raise ValueError(
            "generation replay requires nonempty prompt/action and positive temperature"
        )
    ids = torch.tensor([prompt + raw], device=device)
    logits = base(ids[:, :-1])[0, len(prompt) - 1 :].float()
    targets = ids[0, len(prompt) :]
    sample_lp = torch.log_softmax(logits / temperature, -1)
    base_lp = torch.log_softmax(logits, -1)
    lp = sample_lp.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(base_lp.exp() * base_lp).sum(-1)
    if return_base:
        return lp, entropy, base_lp.gather(-1, targets[:, None]).squeeze(-1)
    return lp, entropy


@torch.no_grad()
def generate_attempt(
    controller,
    tok,
    prompt,
    previous,
    rng,
    context_limit=8192,
    attempt_bound=1024,
    max_action=256,
    max_steps=48,
    temperature=0.8,
):
    from sample import generate_ids_fast
    from state_env import Env
    from state_sample import prompt_ids

    base = controller.base
    device = next(base.parameters()).device
    env = Env(prompt, canon=True, base=rng.randint(0, 32), assign=True)
    prefix = history_prefix(tok, previous)
    raw_actions, effective_actions, actions, transition_actions = [], [], [], []
    generated = 0
    executed_calls = 0
    executed_tokens = 0
    attention_cells = 0
    sampling_seconds = 0.0
    replay_seconds = 0.0
    termination = "step_limit"
    for _step in range(max_steps):
        pids = prefix + prompt_ids(tok, env)
        capacity = min(max_action, attempt_bound - generated, context_limit - len(pids) - 16)
        if capacity < 1:
            termination = "attempt_token_limit" if generated >= attempt_bound else "context_limit"
            break
        seed = rng.randrange(2**31)
        compute = {"calls": 0, "tokens": 0, "attention_cells": 0}

        def count_forward(module, inputs, kwargs, *, compute_values=compute):
            batch, length = inputs[0].shape
            cached = kwargs.get("caches")
            keys = length + (cached[0].get("n", 0) if cached else 0)
            compute_values["calls"] += 1
            compute_values["tokens"] += batch * length
            compute_values["attention_cells"] += batch * module.cfg["n_head"] * length * keys

        sampling_started = time.monotonic()
        hook = base.register_forward_pre_hook(count_forward, with_kwargs=True)
        try:
            emitted = generate_ids_fast(
                base,
                tok,
                [pids],
                greedy=False,
                temperature=temperature,
                max_new=capacity,
                seed=seed,
                early="eos",
                compact=False,
            )[0]
        finally:
            hook.remove()
        sampling_seconds += time.monotonic() - sampling_started
        executed_calls += compute["calls"]
        executed_tokens += compute["tokens"]
        attention_cells += compute["attention_cells"]
        if tok.eos in emitted:
            emitted = emitted[: emitted.index(tok.eos) + 1]
        # Never remove sampled PAD tokens from raw emissions/log probabilities.
        replay_started = time.monotonic()
        lp, entropy, base_lp = token_logprobs(
            base, pids, emitted, device, temperature, return_base=True
        )
        atoks, ended = tok.decode_action(emitted)
        raw_actions.append(emitted)
        generated += len(emitted)
        actions.append(
            {
                "prompt_ids": pids,
                "raw_ids": emitted,
                "logprobs": lp.tolist(),
                "base_logprobs": base_lp.tolist(),
                "entropy": entropy.tolist(),
                "seed": seed,
                "temperature": temperature,
            }
        )
        replay_seconds += time.monotonic() - replay_started
        ok = False
        before_text = len(env.text)
        if ended:
            ok, _ = env.apply(atoks)
        effective = getattr(env, "_eff", atoks) if ended else atoks
        effective_ids = tok.encode_toks(effective) + [tok.eos] if ended else list(emitted)
        effective_actions.append(effective_ids)
        transition = tok.encode_toks(env.text[before_text:])
        if not ended:
            transition += emitted
        elif not ok:
            transition += tok.encode_toks(effective)
        transition_actions.append(transition)
        if not ended:
            termination = "action_truncated"
            break
        if not ok:
            termination = "grammar_failure"
            break
        if env.done:
            termination = "complete"
            break
    transcript = [token for action in transition_actions for token in action]
    transition_tokens = transcript + (
        [tok.eos] if actions and tok.eos in actions[-1]["raw_ids"] else []
    )
    candidate = {
        "candidate_id": len(previous),
        "canonical_name_base": env.base,
        "raw_actions": raw_actions,
        "raw_tokens": [t for action in raw_actions for t in action],
        "effective_actions": effective_actions,
        "effective_tokens": [t for action in effective_actions for t in action],
        "transition_actions": transition_actions,
        "transition_tokens": transition_tokens,
        "transition_text": tok.text([tok.itos[t] for t in transcript]),
        "rendered_text": tok.text([tok.itos[t] for t in transcript]),
        "raw_rendered_text": tok.text(
            [tok.itos[t] for action in raw_actions for t in action if t != tok.eos]
        ),
        "effective_text": tok.text(
            [tok.itos[t] for action in effective_actions for t in action if t != tok.eos]
        ),
        "generated_tokens": generated,
        "prompt_tokens": sum(len(a["prompt_ids"]) for a in actions),
        "logprob_sum": sum(sum(a["logprobs"]) for a in actions),
        "entropy_sum": sum(sum(a["entropy"]) for a in actions),
        "base_logprob_sum": sum(sum(a["base_logprobs"]) for a in actions),
        "sampled_pad_tokens": sum(t == tok.pad for action in raw_actions for t in action),
        "forward_calls": executed_calls,
        "sampling_wall_seconds": sampling_seconds,
        "likelihood_replay_wall_seconds": replay_seconds,
        "forward_tokens": executed_tokens,
        "attention_score_cells_per_layer": attention_cells,
        "likelihood_replay_forward_calls": len(actions),
        "likelihood_replay_forward_tokens": sum(
            len(a["prompt_ids"]) + len(a["raw_ids"]) - 1 for a in actions
        ),
    }
    candidate["mean_logprob"] = candidate["logprob_sum"] / max(1, generated)
    candidate["mean_base_logprob"] = candidate["base_logprob_sum"] / max(1, generated)
    candidate["mean_entropy"] = candidate["entropy_sum"] / max(1, generated)
    private = {
        "parse_complete": env.done,
        "termination": termination,
        "nd_proof": env.nd() if env.done else "",
        "lean_text": tok.text(env.text) if env.done else "",
    }
    return candidate, private, actions
