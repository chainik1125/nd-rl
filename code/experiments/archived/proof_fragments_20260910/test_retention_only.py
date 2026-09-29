import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import torch
import common
import learning as L
from nd_environment import FORMAT
from retention_only import validate_records,validate_main_binding,retention_update,draw_batch,schedule_plan,step_proxy,train


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root=Path(__file__).parent/"tests"
        cls.records=common.read_jsonl(cls.root/"synthetic_smoke_v2/retention.jsonl")
    def model(self):return L.load_model(self.root/"adapt_smoke/final.pt")[:2]
    def test_exact_roles_counts_and_actual_cap_are_required(self):
        validate_records(self.records,expected_count=len(self.records))
        with self.assertRaises(ValueError):validate_records(self.records)
        for field,value in (("role","candidate"),("n_lines",7)):
            bad=copy.deepcopy(self.records);bad[0][field]=value
            with self.assertRaises(ValueError):validate_records(bad,expected_count=len(bad))
        bad=[self.records[0],self.records[0]]
        with self.assertRaises(ValueError):validate_records(bad,expected_count=2)
    def test_main_binding_rejects_schedule_or_optimizer_mismatch(self):
        args=SimpleNamespace(seed=10,lr=5e-5,batch_size=16,retention_weight=.1,precision="fp32")
        main={"args":{"command":"rl","seed":10,"lr":5e-5,"retention_batch":16,"retention_weight":.1,"precision":"fp32"},
              "input_sha256":{"checkpoint":"ck","retention":"data"},"code_sha256":{str(Path(L.__file__)):common.sha(L.__file__)},
              "dependency_sha256":common.dependency_hashes()}
        validate_main_binding(args,main,"ck","data")
        for key,value in (("seed",11),("lr",1e-4),("retention_batch",15),("retention_weight",1.),("precision","bf16")):
            bad=copy.deepcopy(main);bad["args"][key]=value
            with self.assertRaises(ValueError):validate_main_binding(args,bad,"ck","data")
        with self.assertRaises(ValueError):validate_main_binding(args,main,"other","data")
    def test_five_zero_advantage_main_updates_equal_pure_retention_exactly(self):
        main,tok=self.model();pure,_=self.model()
        opt_main=L.optimizer_for(main,5e-5);opt_pure=L.optimizer_for(pure,5e-5)
        retention=L.encode_population(self.records,tok,"empty")
        policy_records=common.read_jsonl(self.root/"synthetic_smoke_v2/candidate.jsonl")[:3]
        policy=L.encode_population(policy_records,tok,"fragment")
        rng=random.Random(17+1000003)
        for step in range(5):
            retained=draw_batch(retention,rng,16)
            batch=L.collate(policy,tok.pad_id)
            with torch.no_grad():logs=L.logprobs(main,batch,temperature=.8)
            rollout=[]
            for i,row in enumerate(policy):
                rollout.append({**row,"advantage":0.,"rollout_logprobs":logs[i][batch["mask"][i]].tolist()})
            first=L.policy_update(main,tok,rollout,opt_main,retained,temperature=.8,max_new_tokens=512,microbatch=2,retention_weight=.1)
            ledger={};second=retention_update(pure,tok,retained,opt_pure,ledger=ledger)
            self.assertEqual(first,second)
            self.assertEqual(set(ledger),{"rl_retention"})
            for p,q in zip(main.parameters(),pure.parameters()):self.assertTrue(torch.equal(p,q))
            for left,right in zip(opt_main.state.values(),opt_pure.state.values()):
                self.assertEqual(set(left),set(right))
                for key in left:self.assertTrue(torch.equal(left[key],right[key]))
    def test_schedule_checkpoints_and_compute_ceiling_are_distinct(self):
        model,tok=self.model();rows=L.encode_population(self.records,tok,"empty");seed=23
        rng=random.Random(seed+1000003);first=draw_batch(rows,rng,16)
        budget=step_proxy(model,first)+1
        plan=schedule_plan(model,rows,seed=seed,batch_size=16,steps=20,max_flop_proxy=budget,save_at_steps=[1,2])
        self.assertEqual(plan["steps"],2);self.assertEqual(plan["stop_reason"],"flop_proxy_ceiling")
        self.assertLess(plan["flop_proxy_overshoot"],plan["worst_case_one_step_flop_proxy"])
        for endpoint in plan["exposure_endpoints"].values():self.assertFalse(endpoint["compute_matched_to_main"])
        with self.assertRaises(ValueError):schedule_plan(model,rows,seed=seed,batch_size=16,steps=20,max_flop_proxy=budget,save_at_steps=[3])
        manual=random.Random(seed+1000003);digest=hashlib.sha256()
        for _ in range(2):digest.update(json.dumps([rows[manual.randrange(len(rows))]["source_id"] for _ in range(16)]).encode())
        self.assertEqual(digest.hexdigest(),plan["retention_schedule_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            ledger={};result=train(model,tok,rows,plan=plan,seed=seed,batch_size=16,lr=5e-5,retention_weight=.1,
                precision="fp32",output=Path(directory),ledger=ledger,checkpoint_metadata={"format":FORMAT},log_every=100)
            self.assertEqual(result["retention_schedule_sha256"],digest.hexdigest())
            self.assertEqual(L.total_flop_proxy(ledger),plan["actual_flop_proxy"])
            self.assertEqual(result["retention_examples"],32)
            self.assertEqual(len(result["exposure_checkpoints"]),2)
            saved=torch.load(Path(directory)/"step_000001.pt",weights_only=True)
            self.assertEqual(saved["metadata"]["endpoint_kind"],"retention_exposure_matched_only")
            self.assertFalse(saved["metadata"]["compute_matched_to_main"])


if __name__=="__main__":unittest.main(verbosity=2)
