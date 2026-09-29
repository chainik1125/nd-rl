"""Explicit response masks, sampled behavior probabilities, and compute ledger."""
from __future__ import annotations
import contextlib
import hashlib
from dataclasses import asdict
import math
import random
import time
import torch
from torch.nn import functional as F
import common
from nd_model import ModelConfig, ProofModel, save_checkpoint
from nd_environment import Tokenizer, episode, judge, independent_replay, split_body

def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tok = Tokenizer()
    if checkpoint["tokenizer"] != tok.state_dict():
        raise ValueError("Expected pinned 43-token relative symbolic tokenizer")
    model = ProofModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), tok, checkpoint.get("metadata", {})

def amp(model, precision="fp32"):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" and next(model.parameters()).is_cuda else contextlib.nullcontext()

def collate(rows, pad_id=0, device="cpu", fixed_pad=None):
    lengths = [len(r["prompt_ids"]) + len(r["response_ids"]) for r in rows]
    width = fixed_pad or max(lengths)
    if max(lengths) > width:
        raise ValueError(f"Sequence {max(lengths)} exceeds fixed pad {width}; no truncation permitted")
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width - 1), dtype=torch.bool, device=device)
    old = torch.zeros_like(mask, dtype=torch.float32)
    for i, row in enumerate(rows):
        values = row["prompt_ids"] + row["response_ids"]
        ids[i, :len(values)] = torch.tensor(values, device=device)
        # Causal position p-1 predicts the FIRST sampled token at position p.
        start, end = len(row["prompt_ids"]) - 1, len(values) - 1
        mask[i, start:end] = True
        if "rollout_logprobs" in row:
            if len(row["rollout_logprobs"]) != len(row["response_ids"]):
                raise ValueError("Every sampled action needs its behavior probability")
            old[i, start:end] = torch.tensor(row["rollout_logprobs"], device=device)
    return {"inputs": ids[:, :-1], "targets": ids[:, 1:], "mask": mask,
            "old_logprobs": old, "lengths": lengths}

def account(ledger, model, batch, phase, backward=False):
    b, s = batch["inputs"].shape
    c = model.config
    slots = b*s
    parameters = sum(p.numel() for p in model.parameters())
    block = ledger.setdefault(phase, {"calls": 0, "padded_token_slots": 0, "actual_input_tokens": 0,
                                       "action_tokens": 0, "supplied_tokens": 0, "flop_proxy": 0})
    block["calls"] += 1
    block["padded_token_slots"] += slots
    block["actual_input_tokens"] += sum(n-1 for n in batch["lengths"])
    actions = int(batch["mask"].sum())
    block["action_tokens"] += actions
    block["supplied_tokens"] += sum(batch["lengths"]) - actions
    # Declared dense approximation, NOT hardware-measured FLOPs. Counts fixed
    # padded projection/MLP and quadratic attention work; omits softmax/LN.
    block["flop_proxy"] += (6 if backward else 2)*parameters*slots + (12 if backward else 4)*c.n_layers*b*s*s*c.d_model

def total_flop_proxy(ledger):
    if not isinstance(ledger,dict): return 0
    return sum(v if k=="flop_proxy" else total_flop_proxy(v) for k,v in ledger.items() if k=="flop_proxy" or isinstance(v,dict))

def worst_case_rl_step_flops(model, n_episodes, max_context, max_response, retention_batch, max_retention_length):
    p=sum(x.numel() for x in model.parameters()); c=model.config
    s=max_context+max_response-1
    decode_calls=max_response-1
    cache_pairs=decode_calls*max_context+decode_calls*(decode_calls+1)//2
    generation=2*p*n_episodes*(max_context+decode_calls)+4*c.n_layers*n_episodes*c.d_model*(max_context**2+cache_pairs)
    audit_and_backward=8*p*n_episodes*s+16*c.n_layers*n_episodes*s*s*c.d_model
    r=max_retention_length-1
    retention=6*p*retention_batch*r+12*c.n_layers*retention_batch*r*r*c.d_model
    return generation+audit_and_backward+retention

def logprobs(model, batch, temperature=1, precision="fp32"):
    with amp(model, precision):
        values = (model(batch["inputs"]).float() / temperature).log_softmax(-1)
    return values.gather(-1, batch["targets"][..., None]).squeeze(-1)

@torch.no_grad()
def score(model, tok, rows, batch_size=32, ledger=None, precision="fp32"):
    model.eval()
    out = []
    device = next(model.parameters()).device
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset+batch_size]
        batch = collate(chunk, tok.pad_id, device)
        chosen = logprobs(model, batch, precision=precision)
        nll = -(chosen*batch["mask"]).sum(-1)/math.log(2)
        if ledger is not None: account(ledger, model, batch, "teacher_scoring")
        for r, value in zip(chunk, nll.cpu().tolist()):
            out.append({"source_id": r["source_id"], "family": r["family"],
                        "bits": value, "tokens": len(r["response_ids"]),
                        "bits_per_token": value / len(r["response_ids"]),
                        "reset_index": r["reset_index"], "source_lines": r["source_lines"]})
    return out

def query_gradient(model,tok,rows,batch_size=16,ledger=None,precision="fp32"):
    """Gradient of equally weighted length-stratum whole-proof bits."""
    from collections import Counter
    def stratum(r):return 0 if r["source_lines"]<=6 else 1 if r["source_lines"]<=9 else 2
    counts=Counter(stratum(r) for r in rows)
    model.eval();model.zero_grad(set_to_none=True)
    for offset in range(0,len(rows),batch_size):
        chunk=rows[offset:offset+batch_size]
        batch=collate(chunk,tok.pad_id,next(model.parameters()).device)
        chosen=logprobs(model,batch,precision=precision)
        weights=torch.tensor([1/(len(counts)*counts[stratum(r)]) for r in chunk],device=chosen.device)
        loss=(-(chosen*batch["mask"]).sum(-1)*weights).sum()/math.log(2)
        loss.backward()
        if ledger is not None:account(ledger,model,batch,"query_gradient",True)
    result={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return result

@torch.no_grad()
def displacement_dot(model,base,gradient):
    original=dict(base.named_parameters())
    return sum(float((gradient[n].float()*(p.detach()-original[n].detach()).float()).sum()) for n,p in model.named_parameters())

def admissible_cuts(record):
    body = record.get("body") or record["text"].split("PRF", 1)[1].strip()
    lines = split_body(body)
    return [k for k in range(len(lines)) if not independent_replay(record["prompt"], " ".join(lines[:k]))["goal_already_available"]]

def encode_population(records, tok, mode="empty", seed=0):
    rng = random.Random(seed)
    result = []
    for r in records:
        cut = 0 if mode == "empty" else r["default_cut"] if mode == "fragment" else rng.choice(admissible_cuts(r))
        result.append({**episode(r, cut, tok), **{k:r[k] for k in ("theorem_id","context_id") if k in r}})
    return result

def optimizer_for(model, lr, weight_decay=0.0):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.95), fused=next(model.parameters()).is_cuda)

def train_sft(model, tok, primary, retention, *, steps, batch_size, fixed_pad, lr, seed,
              ledger, output, retention_fraction=0.5, precision="fp32", log_every=25,
              checkpoint_every=0, callback=None):
    rng = random.Random(seed)
    retention_rng = random.Random(seed + 1000003)
    optimizer = optimizer_for(model, lr)
    nr = round(batch_size*retention_fraction) if retention else 0
    np = batch_size-nr
    if np < 1 or not primary: raise ValueError("Empty intervention batch")
    model.eval() # deterministic dropout-free optimization/likelihoods
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}
    start = time.monotonic()
    history = []
    retention_schedule = hashlib.sha256()
    primary_schedule = hashlib.sha256()
    primary_response_tokens = retention_response_tokens = 0
    device = next(model.parameters()).device
    for step in range(1,steps+1):
        rows = [primary[rng.randrange(len(primary))] for _ in range(np)] + [retention[retention_rng.randrange(len(retention))] for _ in range(nr)]
        primary_schedule.update(common.json.dumps([r["source_id"] for r in rows[:np]]).encode())
        retention_schedule.update(common.json.dumps([r["source_id"] for r in rows[np:]]).encode())
        primary_response_tokens += sum(len(r["response_ids"]) for r in rows[:np])
        retention_response_tokens += sum(len(r["response_ids"]) for r in rows[np:])
        batch = collate(rows,tok.pad_id,device,fixed_pad)
        optimizer.zero_grad(set_to_none=True)
        chosen = logprobs(model,batch,precision=precision)
        # Proof-mean loss gives the declared equal-example mixture. Padding and
        # supplied prefix positions always have zero direct loss.
        losses = -(chosen*batch["mask"]).sum(-1)/batch["mask"].sum(-1)
        loss = losses.mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm): raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        account(ledger,model,batch,"sft_training",True)
        if callback and step == 1: callback(step)
        if step == 1 or step % log_every == 0 or step == steps:
            event = {"step":step,"loss_nats_per_token":float(loss.detach()),"grad_norm":float(norm),
                     "elapsed_seconds":time.monotonic()-start}
            history.append(event)
            print(common.json.dumps(event),flush=True)
            common.write_json(output / "progress.json",event)
        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(output / f"step_{step:06d}.pt",model,tok,step=step,format="nd-prefix-double-prf-final-v1")
    displacement = math.sqrt(sum(float(((p.detach()-initial[n]).float()**2).sum()) for n,p in model.named_parameters()))
    return {"history":history,"optimizer":"AdamW reset to identical empty state at arm start", "parameter_displacement_l2":displacement,
            "elapsed_seconds":time.monotonic()-start,"steps":steps,"fixed_pad":fixed_pad,"retention_fraction":nr/batch_size,
            "retention_schedule_sha256":retention_schedule.hexdigest(), "primary_schedule_sha256":primary_schedule.hexdigest(),
            "primary_response_tokens":primary_response_tokens,"retention_response_tokens":retention_response_tokens,
            "retention_examples":nr*steps,"primary_examples":np*steps}

@torch.inference_mode()
def sample(model,tok,episodes,*,temperature=0.8,seed=0,max_new_tokens=512,batch_size=32,precision="fp32",ledger=None):
    if temperature < 0: raise ValueError("Negative temperature")
    model.eval()
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    order=sorted(range(len(episodes)),key=lambda i:len(episodes[i]["prompt_ids"]))
    outputs=[None]*len(episodes)
    dtype=torch.bfloat16 if precision=="bf16" and device.type=="cuda" else torch.float32
    with amp(model,precision):
        for offset in range(0,len(order),batch_size):
            indices=order[offset:offset+batch_size]
            lengths=torch.tensor([len(episodes[i]["prompt_ids"]) for i in indices],device=device)
            width=int(lengths.max())
            limit=min(max_new_tokens,model.config.max_seq_len-width)
            if limit<1: raise ValueError("Context exhausted before response")
            n=len(indices)
            inputs=torch.full((n,width),tok.pad_id,dtype=torch.long,device=device)
            for j,i in enumerate(indices): inputs[j,-len(episodes[i]["prompt_ids"]):]=torch.tensor(episodes[i]["prompt_ids"],device=device)
            columns=torch.arange(width,device=device)
            valid=columns[None]>=width-lengths[:,None]
            pos=(columns[None]-width+lengths[:,None]).clamp_min(0)
            causal=torch.ones((width,width),dtype=torch.bool,device=device).tril()
            cache=model.allocate_cache(n,width+limit,dtype)
            logits=model(inputs,positions=pos,attention_mask=valid[:,None,None]&causal[None,None],cache=cache)[:,-1]
            valid_keys=torch.ones((n,width+limit),dtype=torch.bool,device=device)
            valid_keys[:,:width]=valid
            done=torch.zeros(n,dtype=torch.bool,device=device)
            stop=torch.full((n,),limit,dtype=torch.long,device=device)
            generated=torch.empty((n,limit),dtype=torch.long,device=device)
            behavior=torch.empty((n,limit),dtype=torch.float32,device=device)
            for step in range(limit):
                distribution=(logits.float()/(temperature or 1)).log_softmax(-1)
                chosen=torch.multinomial(distribution.exp(),1,generator=generator).squeeze(1) if temperature else logits.argmax(-1)
                generated[:,step]=chosen
                behavior[:,step]=distribution.gather(1,chosen[:,None]).squeeze(1) if temperature else 0
                new=(~done)&(chosen==tok.eos_id)
                stop[new]=step+1
                done|=new
                if bool(done.all()) or step+1==limit: break
                cache_start=width+step
                logits=model(chosen[:,None],positions=(lengths+step)[:,None],attention_mask=valid_keys[:,None,None,:cache_start+1],cache=cache,cache_start=cache_start)[:,-1]
            for j,i in enumerate(indices):
                used=min(int(stop[j]),step+1)
                outputs[i]={**episodes[i],"response_ids":generated[j,:used].cpu().tolist(),
                            "rollout_logprobs":behavior[j,:used].cpu().tolist(),
                            "finish_reason":"qed" if bool(done[j]) else "token_limit", "effective_max_new_tokens":limit}
            if ledger is not None:
                block=ledger.setdefault("generation",{"prefill_padded_tokens":0,"prefill_actual_tokens":0,"sampled_tokens":0,"executed_decode_slots":0,"cache_token_pair_slots":0,"flop_proxy":0})
                block["prefill_padded_tokens"]+=n*width
                block["prefill_actual_tokens"]+=int(lengths.sum())
                block["sampled_tokens"]+=sum(len(outputs[i]["response_ids"]) for i in indices)
                block["executed_decode_slots"]+=n*(step+1)
                pairs=n*(width*width+sum(width+k for k in range(1,step+1)))
                block["cache_token_pair_slots"]+=pairs
                p=sum(x.numel() for x in model.parameters());c=model.config
                block["flop_proxy"]+=2*p*n*(width+step)+4*c.n_layers*c.d_model*pairs
            del cache
    return outputs

@torch.no_grad()
def audit_behavior(model,tok,rows,*,temperature,batch_size=16,precision="fp32",tolerance=0.001,ledger=None):
    if temperature<=0: raise ValueError("RL behavior must be stochastic")
    max_error,total_error,count=0.,0.,0
    for offset in range(0,len(rows),batch_size):
        chunk=rows[offset:offset+batch_size]
        batch=collate(chunk,tok.pad_id,next(model.parameters()).device)
        teacher=logprobs(model,batch,temperature,precision)
        delta=(teacher-batch["old_logprobs"]).abs()[batch["mask"]]
        max_error=max(max_error,float(delta.max()))
        total_error+=float(delta.sum()); count+=delta.numel()
        if ledger is not None: account(ledger,model,batch,"behavior_audit")
    if max_error>tolerance: raise RuntimeError(f"Behavior/teacher likelihood error {max_error} > {tolerance}")
    return {"max_abs_error":max_error,"mean_abs_error":total_error/max(1,count),"n_action_tokens":count,"tolerance":tolerance}

def policy_update(model,tok,rows,optimizer,retention,*,temperature,max_new_tokens,microbatch=16,
                  retention_weight=.1,precision="fp32",ledger=None,clip=.2):
    # Current on-policy rows only; actual behavior denominators never replaced.
    optimizer.zero_grad(set_to_none=True)
    loss_sum=0.
    for offset in range(0,len(rows),microbatch):
        chunk=rows[offset:offset+microbatch]
        batch=collate(chunk,tok.pad_id,next(model.parameters()).device)
        chosen=logprobs(model,batch,temperature,precision)
        ratio=(chosen-batch["old_logprobs"]).clamp(-60,60).exp()
        advantages=torch.tensor([r["advantage"] for r in chunk],device=chosen.device)[:,None]
        surrogate=torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages)
        loss=-(surrogate*batch["mask"]).sum()/(len(rows)*max_new_tokens)
        loss.backward(); loss_sum+=float(loss.detach())
        if ledger is not None: account(ledger,model,batch,"rl_training",True)
    retention_loss=0.
    if retention and retention_weight:
        batch=collate(retention,tok.pad_id,next(model.parameters()).device)
        chosen=logprobs(model,batch,precision=precision)
        loss=(-(chosen*batch["mask"]).sum(-1)/batch["mask"].sum(-1)).mean()
        (retention_weight*loss).backward(); retention_loss=float(loss.detach())
        if ledger is not None: account(ledger,model,batch,"rl_retention",True)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm): raise FloatingPointError("Nonfinite RL gradient")
    optimizer.step()
    return {"policy_loss":loss_sum,"retention_loss":retention_loss,"grad_norm":float(norm)}
