"""Independent CPU online-curriculum tests; all proofs/checkpoints synthetic."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import common
import learning as L
import online_bank as B
import online_test as F
import train_lm_online as T
from nd_environment import Tokenizer, episode, judge

ROOT=Path(__file__).resolve().parents[1]

class IndependentOnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_ema_mixed_outcomes_and_zero_cut(self):
        s={"cut":7,"n":0,"success":0}
        rewards=[1,1,0,1,0,0,1,1]
        expected=0.
        for r in rewards:expected=.4*r+.6*expected
        self.assertIsNone(B.advance_curriculum(s,[0,2,7],rewards,mode="flop_ema"))
        self.assertAlmostEqual(s["ema"],expected)
        self.assertEqual((s["cut"],s["n"],s["success"]),(7,8,5))
        e=B.advance_curriculum(s,[0,2,7],[1]*8,mode="flop_ema")
        self.assertEqual(e["old"],7);self.assertEqual(e["new"],2)
        self.assertEqual(e["old_cut_group_outcomes_unused_after_advance"],5)
        self.assertEqual(s,{"cut":2,"n":0,"success":0,"ema":0.})
        B.advance_curriculum(s,[0,2,7],[0]*100,mode="flop_ema")
        self.assertEqual(s["cut"],2) # Failure never moves to later cut.
        zero={"cut":0,"n":0,"success":0}
        self.assertIsNone(B.advance_curriculum(zero,[0,2],[1]*20,mode="flop_ema",force_episodes=1))
        self.assertEqual(zero["cut"],0)

    def test_reference_response_is_unused_and_prefix_is_loss_masked(self):
        record,tok=F.long_record(),Tokenizer()
        bank=B.OnlineProofBank([record],tok,max_seq_len=512,max_new_tokens=128)
        bank.admit(F.simulated_attempt(record,tok),{"kind":"own_online_rollout","synthetic_fixture":True})
        ep=copy.deepcopy(bank.episodes[record["source_id"]][max(bank.cuts[record["source_id"]])])
        changed=copy.deepcopy(ep)
        changed["response_ids"]=[999999]*1000
        changed["source_lines"]=-999
        changed["prefix_state"]={"poisoned_reference_metadata":True}
        model,_=F.small_model()
        a=L.sample(model,tok,[ep],max_new_tokens=12,temperature=.8,seed=19)
        b=L.sample(model,tok,[changed],max_new_tokens=12,temperature=.8,seed=19)
        self.assertEqual(a[0]["response_ids"],b[0]["response_ids"])
        self.assertEqual(a[0]["rollout_logprobs"],b[0]["rollout_logprobs"])
        batch=L.collate(a,tok.pad_id)
        first=len(ep["prompt_ids"])-1
        self.assertFalse(bool(batch["mask"][0,:first].any()))
        self.assertEqual(int(batch["mask"].sum()),len(a[0]["response_ids"]))
        self.assertEqual(ep["prompt_ids"],ep["original_prompt_ids"]+ep["prefix_ids"]+[tok.prf_id])
        # First proof is immutable even when a different valid witness arrives.
        frozen=copy.deepcopy(bank.records)
        later=F.simulated_attempt(record,tok);later["body"]="forged later metadata"
        self.assertIsNone(bank.admit(later,{"kind":"own_online_rollout"}))
        self.assertEqual(bank.records,frozen)
        for invalid in ({"training_kind":"source"},{"reset_index":1},{"finish_reason":"token_limit"},
                        {"response_ids":[tok.eos_id]},{"body":"forged"},{"prefix_ids":[tok.prf_id]}):
            fresh=B.OnlineProofBank([record],tok,max_seq_len=512,max_new_tokens=128)
            row={**F.simulated_attempt(record,tok),**invalid}
            with self.assertRaises(ValueError):fresh.admit(row,{"kind":"own_online_rollout"})

    def test_import_step_and_actual_response_guards(self):
        for corruption in ("descriptor_before_raw","assisted_raw","corrupt_tokens","wrong_body"):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as temp:
                ck,mp,records,tok=F.main_lineage_fixture(temp)
                meta=json.loads(mp.read_text());ap=Path(meta["files"]["attempts"]["path"])
                dp=Path(meta["files"]["discoveries"]["path"])
                raw=common.read_jsonl(ap);desc=common.read_jsonl(dp)
                if corruption=="descriptor_before_raw":raw[0]["step"]=3
                elif corruption=="assisted_raw":raw[0]["reset_index"]=1
                elif corruption=="corrupt_tokens":raw[0]["response_ids"]=[tok.eos_id]
                elif corruption=="wrong_body":raw[0]["body"]="forged body"
                common.write_jsonl(ap,raw)
                B.create_import_manifest(ck,mp) # Fresh hash cannot rescue invalid provenance.
                bank=B.OnlineProofBank(records,tok,max_seq_len=512,max_new_tokens=128)
                with self.assertRaises(ValueError):B.import_main_discoveries(bank,mp,ck)

    def test_dynamic_admission_resume_and_shared_schedules(self):
        # Scripted valid trajectories test state machinery and real optimizer
        # integration. They are explicitly NOT on-policy research observations.
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);model,tok=F.small_model();ck=d/"synthetic.pt"
            F.save_checkpoint(ck,model,tok,synthetic_fixture=True)
            source=F.short_record();target=F.long_record()
            sp,tp=d/"sources.jsonl",d/"targets.jsonl"
            common.write_jsonl(sp,[source]);common.write_jsonl(tp,[{"source_id":target["source_id"],"prompt":target["prompt"]}])
            records={r["source_id"]:r for r in (source,target)}
            cfg={"checkpoint":str(ck),"data":str(sp),"retention":str(sp),"targets":str(tp),
                 "online_bank":True,"target_fraction":.5,"objective":"ppo","start":"reset","empty_fraction":0.,
                 "curriculum_mode":"flop_ema","backward_threshold":.9,"max_updates":3,"prompts_per_step":2,
                 "group_size":8,"max_new_tokens":160,"retention_batch":1,"microbatch":16,"batch_size":16,
                 "epochs":2,"checkpoint_every":1,"cpu_threads":1,"seed":32,"max_seconds":60,"max_logprob_error":1e-5}
            def scripted_sample(model,tok,episodes,**kwargs):
                rows=[]
                for ep in episodes:
                    response=episode(records[ep["source_id"]],ep["reset_index"],tok)["response_ids"]
                    rows.append({**ep,"response_ids":response,"finish_reason":"qed","effective_max_new_tokens":kwargs["max_new_tokens"]})
                batch=L.collate(rows,tok.pad_id)
                with torch.no_grad():chosen=L.logprobs(model,batch,kwargs["temperature"],kwargs["precision"])
                for i,row in enumerate(rows):row["rollout_logprobs"]=chosen[i][batch["mask"][i]].tolist()
                return rows
            def run(name,steps=3,enabled=True,start="reset",resume=False):
                out=d/name;cp=d/"config.json"
                c={**cfg,"output":str(out),"max_updates":steps,"online_bank":enabled,"start":start}
                if resume:c["resume"]=str(out/"final.pt")
                common.write_json(cp,c)
                with patch.object(L,"sample",scripted_sample),redirect_stdout(io.StringIO()):T.run(T.arguments(["--config",str(cp)]))
                return torch.load(out/"final.pt",weights_only=True)
            a=run("continuous");run("resumed",1)
            T.append_jsonl(d/"resumed/admissions.jsonl",[{"step":2,"uncommitted":True}])
            b=run("resumed",resume=True);off=run("off",enabled=False)
            for k in ("model","critic"):
                for n,v in a[k].items():torch.testing.assert_close(v,b[k][n],rtol=0,atol=0)
            sa,sb=a["training_state"],b["training_state"]
            for k in ("online_curriculum","curriculum","rngs","schedule_digest","retention_schedule_digest","ledger","generated_tokens"):
                self.assertEqual(sa[k],sb[k])
            sid=target["source_id"]
            for k in ("body","witness_id","full_start_response_ids","response_ids_sha256"):
                self.assertEqual(sa["online_bank"][sid][k],sb["online_bank"][sid][k])
            admissions=common.read_jsonl(d/"resumed/admissions.jsonl")
            self.assertEqual(len(admissions),1);self.assertEqual(admissions[0]["step"],1)
            groups=common.read_jsonl(d/"continuous/groups.jsonl")
            ogroups=common.read_jsonl(d/"off/groups.jsonl")
            self.assertEqual([(g["source_id"],g["reset_draw"]) for g in groups],[(g["source_id"],g["reset_draw"]) for g in ogroups])
            self.assertEqual(sa["retention_schedule_digest"],off["training_state"]["retention_schedule_digest"])
            targetgroups=[g for g in groups if g["training_kind"]=="target"]
            self.assertEqual(targetgroups[0]["reset_index"],0)
            self.assertNotIn("online_curriculum_update",targetgroups[0])
            self.assertTrue(targetgroups[1]["online_assisted"])
            self.assertEqual(targetgroups[1]["online_curriculum_update"]["n"],5)
            self.assertEqual(targetgroups[1]["online_curriculum_update"]["old_cut_group_outcomes_unused_after_advance"],3)
            self.assertTrue(all(g["reset_index"]==0 for g in ogroups if g["training_kind"]=="target"))
            # Full-start online bank is bookkeeping only, including new admission.
            fa=run("full_on",1,True,"full");fb=run("full_off",1,False,"full")
            for k in ("model","critic"):
                for n,v in fa[k].items():torch.testing.assert_close(v,fb[k][n],rtol=0,atol=0)

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentOnlineTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    common.write_json(ROOT/"reports/review_online.json",{"ok":result.wasSuccessful(),"tests":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),
        "hashes":{p:common.sha(ROOT/p) for p in ("code/train_lm_online.py","code/online_bank.py","code/online_test.py","code/review_online.py","code/train_lm.py","code/ppo_core.py")},
        "scope":"Synthetic CPU fixtures only; scripted valid trajectories test state/optimizer integration, not research performance. No GPU or final/model outcome access."})
    raise SystemExit(not result.wasSuccessful())
