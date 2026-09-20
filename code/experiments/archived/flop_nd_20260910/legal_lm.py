"""Inference-only LM policy constrained to a finite ND action trie.

This is a symbolic-interface intervention, NOT unaided token generation.
Premises, syntax, legality, and successful termination are constrained by the
frozen symbolic environment. The LM chooses tokens only at trie branches.
There is no backtracking, reference proof, optimizer, or verifier repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import torch
import common
import learning as L
from nd_data import formula_text
from nd_environment import judge, strict_complete, Tokenizer
from proof_diagnostics import compress_proof
from symbolic_general_env import Environment, Action


def encode_action(tok, action, next_line):
    # Codec uses the actual line header to encode relative citation distances;
    # this dummy prompt is never fed to the model or used by the environment.
    dummy="THM SEQ P PRF"
    text=" ".join([f"N{next_line}"]+["|"]*action.depth+[formula_text(action.formula),":",action.rule]
                  +[f"N{r}" for r in action.refs]+[";","QED"])
    return tok.encode_record(dummy+" "+text)[len(tok.encode_prompt(dummy)):-1]


def action_trie(tok, actions, next_line):
    trie={}
    for index,action in enumerate(actions):
        node=trie
        for token in encode_action(tok,action,next_line):
            if "action" in node:raise AssertionError("An action encoding is a prefix of another")
            node=node.setdefault(token,{})
        if node:raise AssertionError("Duplicate or prefix action encoding")
        node["action"]=index
    return trie


class ConstrainedState:
    def __init__(self, row, tok, max_lines=24, max_scope=4):
        self.row=row;self.tok=tok
        self.env=Environment(row["prompt"],max_scope=max_scope,max_lines=max_lines)
        self.response=[];self.trace=[];self.constraints=[];self.done=False;self.reason=None
        self.candidates=0;self.branch_tokens=0;self.forced_tokens=0
        encoded=tok.encode_record(row["prompt"]+" "+self.env.body())
        self.forced=[tok.stoi["<FINAL>"]]+encoded[len(tok.encode_prompt(row["prompt"])):-1]
        self.forced_position=0;self.actions=[];self.node=None;self.pending_qed=False

    def allowed(self):
        if self.done:return [self.tok.pad_id]
        if self.forced_position<len(self.forced):return [self.forced[self.forced_position]]
        if self.env.success():
            self.pending_qed=True
            return [self.tok.eos_id]
        if self.node is None:
            self.actions=self.env.actions()
            if not self.actions:
                self.done=True;self.reason="line_cap_or_no_legal_action"
                return [self.tok.pad_id]
            self.candidates+=len(self.actions)
            self.node=action_trie(self.tok,self.actions,len(self.env.lines)+1)
        return sorted(self.node)

    def consume(self, token, allowed):
        if self.done:return
        if token not in allowed:raise AssertionError("Decoder escaped its legal trie")
        self.response.append(token)
        self.constraints.append(allowed)
        self.branch_tokens+=len(allowed)>1;self.forced_tokens+=len(allowed)==1
        if self.forced_position<len(self.forced):
            self.forced_position+=1;return
        if self.pending_qed:
            if token!=self.tok.eos_id:raise AssertionError("Expected constrained termination")
            self.done=True;self.reason="qed";return
        self.node=self.node[token]
        if "action" in self.node:
            index=self.node["action"];action=self.actions[index]
            self.trace.append({"rule":action.rule,"formula":action.formula,"refs":action.refs,
                               "depth":action.depth,"candidate_count":len(self.actions),"selected_index":index,
                               "response_end":len(self.response)})
            self.env.step(action,check_membership=False)
            self.node=None

    def finish(self):
        episode={"prompt":self.row["prompt"],"prefix_ids":[]}
        verdict=judge(episode,self.response,self.tok)
        if self.reason=="qed" and not verdict["reward"]:
            raise AssertionError(f"Constrained proof failed both-checker gate: {verdict}")
        if self.reason!="qed" and verdict["reward"]:
            raise AssertionError("Incomplete constrained attempt received success")
        result={"source_id":self.row["source_id"],"prompt":self.row["prompt"],
                "attempt":self.row.get("attempt",0),"response_ids":self.response,
                "allowed_token_ids":self.constraints,"action_trace":self.trace,
                "finish_reason":self.reason or "token_limit", "generated_tokens":len(self.response),
                "generated_inferences":self.env.generated,"registered_premises":len(self.env.premises),
                "branch_tokens":self.branch_tokens,"forced_tokens":self.forced_tokens,
                "candidate_actions_enumerated":self.candidates,"symbolic_constraints":True,**verdict}
        if verdict["reward"]:
            compression=compress_proof(self.row["prompt"],verdict["body"])
            if not strict_complete(self.row["prompt"],compression["compressed_proof"])["ok"]:
                raise AssertionError("Compressed constrained proof invalid")
            result["compression"]=compression
        return result


@torch.no_grad()
def generate(model,tok,rows,*,max_new_tokens=1024,batch_size=16,temperature=0,seed=1,max_lines=24,max_scope=4):
    model.eval();device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    outputs=[]
    for offset in range(0,len(rows),batch_size):
        chunk=rows[offset:offset+batch_size]
        states=[ConstrainedState(row,tok,max_lines,max_scope) for row in chunk]
        prompts=[tok.encode_prompt(row["prompt"])+[tok.prf_id] for row in chunk]
        lengths=torch.tensor([len(p) for p in prompts],device=device)
        width=int(lengths.max());n=len(chunk)
        if width+max_new_tokens>model.config.max_seq_len:raise ValueError("Fixed response cap exceeds context")
        inputs=torch.full((n,width),tok.pad_id,dtype=torch.long,device=device)
        for index,prompt in enumerate(prompts):inputs[index,-len(prompt):]=torch.tensor(prompt,device=device)
        columns=torch.arange(width,device=device)
        valid=columns[None]>=width-lengths[:,None]
        positions=(columns[None]-width+lengths[:,None]).clamp_min(0)
        causal=torch.ones((width,width),dtype=torch.bool,device=device).tril()
        cache=model.allocate_cache(n,width+max_new_tokens,torch.float32)
        logits=model(inputs,positions=positions,attention_mask=valid[:,None,None]&causal[None,None],cache=cache)[:,-1]
        valid_keys=torch.ones((n,width+max_new_tokens),dtype=torch.bool,device=device)
        valid_keys[:,:width]=valid
        for step in range(max_new_tokens):
            allowed=[state.allowed() for state in states]
            masked=torch.full_like(logits,-torch.inf)
            for index,tokens in enumerate(allowed):masked[index,tokens]=logits[index,tokens]
            if temperature:
                probs=(masked.float()/temperature).softmax(-1)
                chosen=torch.multinomial(probs,1,generator=generator).squeeze(1)
            else:chosen=masked.argmax(-1)
            for state,token,tokens in zip(states,chosen.cpu().tolist(),allowed):state.consume(token,tokens)
            if all(state.done for state in states) or step+1==max_new_tokens:break
            cache_start=width+step
            logits=model(chosen[:,None],positions=(lengths+step)[:,None],
                         attention_mask=valid_keys[:,None,None,:cache_start+1],cache=cache,cache_start=cache_start)[:,-1]
        outputs.extend(state.finish() for state in states)
        del cache
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-new-tokens",type=int,default=1024)
    p.add_argument("--max-lines",type=int,default=24)
    p.add_argument("--max-scope",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--attempts",type=int,default=1)
    p.add_argument("--temperature",type=float,default=0)
    p.add_argument("--seed",type=int,default=202609101935)
    p.add_argument("--device",default="cpu")
    p.add_argument("--release",type=Path)
    args=p.parse_args();torch.set_num_threads(4)
    if args.attempts<1 or args.temperature<0 or (args.temperature==0 and args.attempts!=1):
        raise ValueError("Invalid attempt/temperature specification")
    profile={k:getattr(args,k) for k in ("max_new_tokens","max_lines","max_scope","batch_size","attempts","temperature","seed")}
    if args.data.stem=="final":
        if args.release is None:raise ValueError("Final constrained evaluation requires a frozen release")
        release=json.loads(args.release.read_text())
        if (release.get("status")!="released" or common.sha(args.checkpoint) not in release["checkpoint_sha256"].values()
                or common.sha(args.data)!=release["data_sha256"] or profile not in release["inference_profiles"]):
            raise ValueError("Final checkpoint/population/profile absent from release")
    rows=common.read_jsonl(args.data)
    if any(set(row)-{"name","source_id","prompt"} for row in rows):raise ValueError("Only theorem-only inputs allowed")
    if len({r["source_id"] for r in rows})!=len(rows):raise ValueError("Duplicate theorem identities")
    args.output.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    hashes={str(path.resolve()):common.sha(path) for path in [args.checkpoint,args.data,Path(__file__),
        Path(__file__).with_name("symbolic_general_env.py"),Path(__file__).with_name("learning.py"),
        Path(__file__).with_name("nd_environment.py"),*common.VENDOR.rglob("*.py")]}
    common.write_json(args.output/"config.json",{"args":config,"input_sha256":hashes,"pid":os.getpid(),
        "inference":"Locally normalized token trie over finite legal ND lines. Premises and successful termination are forced. No reference, backtracking, repair, or learning. Not unaided LM generation."})
    model,tok,_=L.load_model(args.checkpoint,args.device)
    started=time.monotonic();all_rows=[]
    block=max(1,args.batch_size//args.attempts)
    with (args.output/"attempts.jsonl").open("w",buffering=1) as stream:
        for offset in range(0,len(rows),block):
            episodes=[{**row,"attempt":a} for row in rows[offset:offset+block] for a in range(args.attempts)]
            results=generate(model,tok,episodes,max_new_tokens=args.max_new_tokens,batch_size=args.batch_size,
                temperature=args.temperature,seed=args.seed+offset*1009,max_lines=args.max_lines,max_scope=args.max_scope)
            for row in results:stream.write(json.dumps(row)+"\n")
            all_rows.extend(results)
            progress={"theorems_done":min(offset+block,len(rows)),"theorems_total":len(rows),
                      "attempts":len(all_rows),"successes":sum(r["reward"] for r in all_rows),
                      "elapsed_seconds":time.monotonic()-started}
            common.write_json(args.output/"progress.json",progress);print(json.dumps(progress),flush=True)
    common.write_json(args.output/"result.json",{"status":"completed",**progress,
        "any_successes":len({r["source_id"] for r in all_rows if r["reward"]}),
        "generated_tokens":sum(r["generated_tokens"] for r in all_rows),
        "branch_tokens":sum(r["branch_tokens"] for r in all_rows),"forced_tokens":sum(r["forced_tokens"] for r in all_rows),
        "failure_reasons":dict(Counter(r["finish_reason"] for r in all_rows if not r["reward"])),
        "attempts_sha256":common.sha(args.output/"attempts.jsonl"),"input_sha256":hashes,"profile":profile})


if __name__=="__main__":main()
