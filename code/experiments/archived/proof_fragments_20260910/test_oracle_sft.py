"""Prepare the oracle-information control using only tiny CPU fixtures."""
from pathlib import Path
import unittest
import copy
import torch
import common
import learning as L
from nd_environment import Tokenizer
from nd_model import ProofModel,ModelConfig
from oracle_sft import validate_sources,encode_with_origin,budget_plan,step_proxy,train,bind_population

FIXTURE=Path(__file__).parent/"tests/synthetic_smoke_v2"
torch.set_num_threads(1)

def populations():return {r:common.read_jsonl(FIXTURE/f"{r}.jsonl") for r in ("candidate","selection","retention")}

class OracleSourceTests(unittest.TestCase):
    def test_oracle_union_explicitly_preserves_original_roles_and_long_sources(self):
        p=populations();stats=validate_sources(p["candidate"],p["selection"],p["retention"])
        self.assertEqual(stats["selection"]["source_cap"],12)
        self.assertTrue(any(r["n_lines"]>6 for r in p["selection"]))
        encoded=encode_with_origin(p["selection"],Tokenizer())
        self.assertTrue(all(r["oracle_original_role"]=="selection" and r["reset_index"]==0 for r in encoded))
        self.assertTrue(all(r["role"]=="selection" for r in p["selection"]))

    def test_confirmation_and_textbook_paths_rejected_before_payload_read(self):
        for name in ("confirmation.jsonl","final_confirmation.jsonl","textbook_train.jsonl"):
            with self.assertRaisesRegex(ValueError,"cannot use"):bind_population("unopened/"+name,"selection")

    def test_confirmation_role_and_long_retention_are_rejected(self):
        p=populations();bad=copy.deepcopy(p["selection"]);bad[0]["role"]="confirmation"
        with self.assertRaisesRegex(ValueError,"Expected role selection"):validate_sources(p["candidate"],bad,p["retention"])
        long=copy.deepcopy(next(r for r in p["selection"] if r["n_lines"]>6));long["role"]="retention"
        with self.assertRaisesRegex(ValueError,"cap exceeded"):validate_sources(p["candidate"],p["selection"],[long])

    def test_compute_plan_is_exact_or_explicitly_floored(self):
        self.assertEqual(budget_plan("1e3",100,"exact")["derived_steps"],10)
        with self.assertRaisesRegex(ValueError,"divisible"):budget_plan(1001,100,"exact")
        plan=budget_plan(1001,100,"floor")
        self.assertEqual(plan["unused_flop_proxy"],1);self.assertEqual(plan["scheduled_training_flop_proxy"],1000)
        self.assertEqual(budget_plan("9007199254740993",1,"exact")["requested_flop_proxy"],9007199254740993)

    def test_tiny_training_records_role_and_length_exposures_and_exact_proxy(self):
        p=populations();tok=Tokenizer();torch.manual_seed(27)
        model=ProofModel(ModelConfig(vocab_size=len(tok),d_model=24,n_layers=1,n_heads=2,d_ff=48,max_seq_len=512)).eval()
        primary=encode_with_origin([p["candidate"][0],p["selection"][0]],tok)
        retention=encode_with_origin(p["retention"][:2],tok)
        output=Path(__file__).parent/"tests/oracle_tiny_training";output.mkdir(exist_ok=True,parents=True)
        ledger={};cost=step_proxy(model,4,512)
        result=train(model,tok,primary,retention,steps=2,batch_size=4,fixed_pad=512,lr=1e-4,seed=17,
                     retention_fraction=.5,precision="fp32",output=output,ledger=ledger)
        self.assertEqual(L.total_flop_proxy(ledger),2*cost)
        self.assertEqual(result["role_exposure_counts"]["retention"],4)
        self.assertEqual(sum(result["role_exposure_counts"].values()),8)
        self.assertEqual(sum(sum(v.values()) for v in result["role_reference_length_response_tokens"].values()),
                         sum(result["role_response_tokens"].values()))
        self.assertTrue(result["parameter_displacement_l2"]>0)

if __name__=="__main__":unittest.main(verbosity=2)
