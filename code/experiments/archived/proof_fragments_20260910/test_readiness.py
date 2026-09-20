"""Semantic CPU checks using invented proofs and random tiny policies only."""
import copy
from pathlib import Path
import random
import unittest
import torch
import common
import learning as L
from nd_environment import Tokenizer, independent_replay, strict_complete, episode, judge
from nd_model import ProofModel, ModelConfig
from strict_verify import safe_verify
from nd_data import generate_candidate
from populations import enrich, audit
from fragment_experiment import matched_controls, training_rows, release_check
from types import SimpleNamespace

torch.set_num_threads(1)
CASES = [
 ("THM P , Q SEQ ( P & Q ) PRF", "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"),
 ("THM SEQ ( P > P ) PRF", "N1 | P : AS ; N2 | P : R N1 ; N3 ( P > P ) : IMPI N1 N2 ; QED"),
 ("THM ( P v Q ) SEQ ( Q v P ) PRF", "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED"),
 ("THM ( ~ P ) SEQ ( ~ P ) PRF", "N1 ( ~ P ) : PR ; N2 | P : AS ; N3 | F : NEGE N2 N1 ; N4 ( ~ P ) : NEGI N2 N3 ; QED"),
 ("THM ( P > Q ) , P SEQ Q PRF", "N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED"),
 ("THM ( ~ ( ~ P ) ) SEQ P PRF", "N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; QED"),
 ("THM P , ( ~ P ) SEQ Q PRF", "N1 P : PR ; N2 ( ~ P ) : PR ; N3 F : NEGE N1 N2 ; N4 Q : BOTE N3 ; QED"),
 ("THM SEQ ( P > ( Q > P ) ) PRF", "N1 | P : AS ; N2 | | Q : AS ; N3 | | P : R N1 ; N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED"),
]

def record(case=0):
    prompt,body=CASES[case]
    return {"prompt":prompt,"body":body,"family":"invented","source_id":common.digest(case)}

def tiny(tok):
    torch.manual_seed(117)
    return ProofModel(ModelConfig(vocab_size=len(tok),d_model=24,n_layers=1,n_heads=2,d_ff=48,max_seq_len=512)).eval()

class EnvironmentTests(unittest.TestCase):
    def test_all_rules_scopes_and_every_nonterminal_cut_roundtrip(self):
        tok=Tokenizer()
        for case,(prompt,body) in enumerate(CASES):
            self.assertTrue(safe_verify(prompt,body)[0],case)
            self.assertTrue(strict_complete(prompt,body)["ok"],case)
            for cut in L.admissible_cuts(record(case)):
                row=episode(record(case),cut,tok)
                self.assertEqual(judge(row,row["response_ids"],tok)["reward"],1,(case,cut))
                self.assertEqual(row["prefix_state"]["next_line"],cut+1)

    def test_inaccessible_scope_future_ref_and_wrong_rule_fail(self):
        prompt="THM SEQ ( P > P ) PRF"
        for body in (
            "N1 | P : AS ; N2 P : R N1 ; N3 ( P > P ) : IMPI N1 N2 ; QED",
            "N1 | P : AS ; N2 | P : R N3 ; N3 ( P > P ) : IMPI N1 N2 ; QED",
            "N1 | P : AS ; N2 | P : DN N1 ; N3 ( P > P ) : IMPI N1 N2 ; QED",
        ):
            self.assertFalse(strict_complete(prompt,body)["ok"])

    def test_no_terminal_or_partial_prefix_and_no_trailing_repair(self):
        with self.assertRaises(ValueError):independent_replay(CASES[0][0],"N1 P : PR")
        with self.assertRaises(ValueError):independent_replay(CASES[0][0],"QED")
        self.assertFalse(strict_complete(CASES[0][0],CASES[0][1]+" QED")["ok"])
        with self.assertRaises(ValueError):episode(record(3),1)
        tok=Tokenizer(); row=episode(record(),2,tok)
        self.assertEqual(judge(row,row["response_ids"][:-1],tok)["reward"],0)
        self.assertEqual(judge(row,[tok.eos_id],tok)["reward"],0)

    def test_original_premises_never_replaced_by_prefix_conclusion(self):
        row=episode(record(0),2)
        self.assertEqual(row["prompt"],CASES[0][0])
        self.assertEqual(row["prefix_state"]["premises_seen"],2)
        with self.assertRaises(ValueError):independent_replay(CASES[0][0],"N1 P : PR ; N2 ( P & Q ) : ANDI N1 N1 ;")

    def test_random_forward_full_agreement_and_mutations(self):
        rng=random.Random(9282); accepted=0
        for _ in range(1000):
            raw=generate_candidate(rng,rng.choice((6,10,12)))
            if raw is None:continue
            body=raw["text"].split("PRF",1)[1].strip()
            self.assertTrue(strict_complete(raw["prompt"],body)["ok"])
            accepted+=1
            mutated=body.replace("N1","N0",1)
            self.assertFalse(strict_complete(raw["prompt"],mutated)["ok"])
        self.assertGreater(accepted,30)

class LearningTests(unittest.TestCase):
    def test_control_matching_never_reuses_sources(self):
        tok=Tokenizer()
        primary=[episode(record(0),2,tok),episode(record(0),2,tok)]
        ordinary=[episode(record(1),0,tok),episode(record(7),3,tok)]
        controls,receipts=matched_controls(primary,ordinary)
        self.assertEqual(len({r["ordinary_source"] for r in receipts}),2)
        with self.assertRaises(ValueError):matched_controls(primary,ordinary[:1])

    def test_confirmation_release_preflight_blocks_before_parsing(self):
        args=SimpleNamespace(data="unopened/final_confirmation.jsonl",release=None)
        with self.assertRaises(ValueError):release_check(args)
        with self.assertRaises(ValueError):training_rows("unopened/confirmation.jsonl")

    def test_unified_generation_compute_and_one_step_upper_bound(self):
        tok=Tokenizer();model=tiny(tok);ledger={}
        episodes=[episode(record(0),2,tok)]*4
        rows=L.sample(model,tok,episodes,temperature=.8,seed=91,max_new_tokens=12,batch_size=4,ledger=ledger)
        L.audit_behavior(model,tok,rows,temperature=.8,ledger=ledger)
        for i,r in enumerate(rows):r["advantage"]=1. if i%2 else -1.
        retention=[episode(record(1),0,tok)]*2
        L.policy_update(model,tok,rows,L.optimizer_for(model,1e-4),retention,temperature=.8,max_new_tokens=12,ledger=ledger)
        bound=L.worst_case_rl_step_flops(model,4,len(episodes[0]["prompt_ids"]),12,2,len(retention[0]["prompt_ids"])+len(retention[0]["response_ids"]))
        self.assertGreater(ledger["generation"]["flop_proxy"],0)
        self.assertLessEqual(L.total_flop_proxy(ledger),bound)

    def test_causal_mask_starts_at_last_context_and_keeps_sampled_pad(self):
        tok=Tokenizer(); rows=[episode(record(0),2,tok),episode(record(7),0,tok)]
        rows[0]["response_ids"]=[tok.pad_id]+rows[0]["response_ids"]
        batch=L.collate(rows,fixed_pad=256)
        for i,r in enumerate(rows):
            start=len(r["prompt_ids"])-1
            self.assertFalse(bool(batch["mask"][i,:start].any()))
            self.assertEqual(int(batch["mask"][i].sum()),len(r["response_ids"]))
            self.assertTrue(bool(batch["mask"][i,start]))
            self.assertFalse(bool(batch["mask"][i,start+len(r["response_ids"]):].any()))
        self.assertEqual(int(batch["targets"][0,len(rows[0]["prompt_ids"])-1]),tok.pad_id)

    def test_cached_behavior_matches_teacher_with_varied_contexts(self):
        tok=Tokenizer();model=tiny(tok)
        episodes=[episode(record(0),2,tok),episode(record(7),3,tok),episode(record(1),0,tok)]
        rows=L.sample(model,tok,episodes,seed=771,temperature=.8,max_new_tokens=20,batch_size=3)
        result=L.audit_behavior(model,tok,rows,temperature=.8,tolerance=1e-5,batch_size=2)
        self.assertLess(result["max_abs_error"],1e-5)
        bad=copy.deepcopy(rows);bad[0]["rollout_logprobs"][0]+=.1
        with self.assertRaises(RuntimeError):L.audit_behavior(model,tok,bad,temperature=.8,tolerance=1e-5)

    def test_padding_and_supplied_positions_have_zero_direct_gradient(self):
        tok=Tokenizer();r=episode(record(0),2,tok)
        batch=L.collate([r],fixed_pad=128)
        logits=torch.randn(1,127,len(tok),requires_grad=True)
        chosen=logits.log_softmax(-1).gather(-1,batch["targets"][...,None]).squeeze(-1)
        loss=-(chosen*batch["mask"]).sum();loss.backward()
        self.assertEqual(float(logits.grad[~batch["mask"]].abs().sum()),0.)
        self.assertGreater(float(logits.grad[batch["mask"]].abs().sum()),0.)

    def test_one_update_has_finite_displacement_and_fixed_compute(self):
        tok=Tokenizer();rows=[episode(record(0),2,tok),episode(record(1),0,tok)]
        model=tiny(tok);ledger={}
        out=Path(__file__).parent/"tests/update_smoke";out.mkdir(parents=True,exist_ok=True)
        result=L.train_sft(model,tok,rows,rows,steps=2,batch_size=4,fixed_pad=256,lr=1e-4,
                            seed=31,ledger=ledger,output=out)
        self.assertGreater(result["parameter_displacement_l2"],0.)
        self.assertEqual(ledger["sft_training"]["padded_token_slots"],2*4*255)

if __name__=="__main__":unittest.main(verbosity=2)
