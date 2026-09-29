import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import common
import learning as L
import nd_environment as E
from initial_cuts import RULE,choose_cut,state_hash,validate_map
from all_cut_pilot import contract,metadata_compatibility,group_summary

class InitialCutTests(unittest.TestCase):
    def test_max_variance_and_earlier_tie(self):
        value=choose_cut({0:8,1:4,2:4,3:7})
        self.assertEqual(value["cut"],1);self.assertTrue(value["observed_mixed_frontier"])
    def test_declared_fallbacks(self):
        self.assertEqual(choose_cut({0:0,1:8,2:8})["cut"],1)
        self.assertEqual(choose_cut({0:0,1:0,2:0})["cut"],2)
        self.assertFalse(choose_cut({0:0,1:8})["observed_mixed_frontier"])
    def test_exact_policy_sampler_and_state_map_binding(self):
        r=common.read_jsonl(Path(__file__).parent/"tests/synthetic_smoke_v2/candidate.jsonl")[0]
        tok=E.Tokenizer();cut=max(L.admissible_cuts(r));ep=E.episode(r,cut,tok)
        args=SimpleNamespace(temperature=.8,max_new_tokens=24,precision="fp32")
        value={"status":"frozen_initial_cuts","rule":RULE,**contract(args,"ck"),"counters_initialized_to_zero":True,
               "sources":{r["source_id"]:{"cut":cut,"chosen_state_sha256":state_hash(ep)}}}
        self.assertEqual(validate_map(value,[r],tok,"ck",.8,24,"fp32")[r["source_id"]],cut)
        bad=copy.deepcopy(value);bad["sources"][r["source_id"]]["chosen_state_sha256"]="wrong"
        with self.assertRaises(ValueError):validate_map(bad,[r],tok,"ck",.8,24,"fp32")
        with self.assertRaises(ValueError):validate_map(value,[r],tok,"other",.8,24,"fp32")
    def test_reuse_metadata_rejects_different_distribution_or_sampler(self):
        args=SimpleNamespace(temperature=.8,max_new_tokens=512,precision="fp32")
        current=contract(args,"ck")
        manifest={"args":{"temperature":.8,"max_new_tokens":512,"precision":"fp32","samples":8},
                  "input_sha256":{"checkpoint":"ck"},"format":E.FORMAT,"dependency_sha256":common.dependency_hashes(),
                  "code_sha256":{str(Path(L.__file__)) : common.sha(L.__file__),str(Path(E.__file__)) : common.sha(E.__file__)}}
        self.assertIsNone(metadata_compatibility(manifest,current))
        bad=copy.deepcopy(manifest);bad["args"]["temperature"]=1.
        self.assertIn("temperature",metadata_compatibility(bad,current))
        bad=copy.deepcopy(manifest);bad["code_sha256"][str(Path(L.__file__))]="changed"
        self.assertIn("learning.py",metadata_compatibility(bad,current))
    def test_loo_energy_matches_explicit_advantages(self):
        rewards=[1]*3+[0]*5
        rows=[{"reward":r,"response_ids":[1]*2,"prefix_ids":[1]} for r in rewards]
        explicit=sum((r-(sum(rewards)-r)/7)**2 for r in rewards)
        self.assertAlmostEqual(group_summary(rows)["advantage_energy"],explicit)

if __name__=="__main__":unittest.main(verbosity=2)
