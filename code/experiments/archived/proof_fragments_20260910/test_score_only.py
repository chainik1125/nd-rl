import unittest
import math
from pathlib import Path
import torch
import common
import learning as L
import reference_scoring as R
from score_only import validate_release,summarize

class ScoreGateTests(unittest.TestCase):
    def release(self):return {"frozen":True,"no_further_selector_changes":True,"data_sha256":["data"],"checkpoints":["a","b"]}
    def test_confirmation_requires_every_checkpoint_and_closed_selector(self):
        validate_release(self.release(),"confirmation","data",["a","b"])
        with self.assertRaises(ValueError):validate_release(self.release(),"confirmation","data",["a","c"])
        release=self.release();release["no_further_selector_changes"]=False
        with self.assertRaises(ValueError):validate_release(release,"confirmation","data",["a"])
    def test_final_closes_training_and_other_roles_rejected(self):
        with self.assertRaises(ValueError):validate_release(self.release(),"final_confirmation","data",["a"])
        release=self.release();release["no_further_training"]=True
        validate_release(release,"final_confirmation","data",["a"])
        with self.assertRaises(ValueError):validate_release(release,"selection","data",["a"])
    def test_summary_weights_length_strata_equally(self):
        rows=[{"length_bin":"2-6","family":"A","bits":2.,"tokens":2}]*2+[
            {"length_bin":"7-9","family":"B","bits":8.,"tokens":4},
            {"length_bin":"10-12","family":"B","bits":20.,"tokens":5}]
        result=summarize(rows)
        self.assertEqual(result["mean_bits"],8.)
        self.assertEqual(result["length_balanced_mean_bits"],10.)

class TemperatureScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root=Path(__file__).parent/"tests"
        cls.model,cls.tok,_=L.load_model(root/"adapt_smoke/final.pt")
        records=common.read_jsonl(root/"synthetic_smoke_v2/candidate.jsonl")[:3]
        cls.rows=L.encode_population(records,cls.tok,"fragment")
    def test_default_and_explicit_one_equal_existing_scorer(self):
        old_ledger={};new_ledger={}
        old=L.score(self.model,self.tok,self.rows,2,old_ledger)
        new=R.score(self.model,self.tok,self.rows,2,new_ledger)
        explicit=R.score(self.model,self.tok,self.rows,2,score_temperature=1.)
        self.assertEqual(new,explicit)
        self.assertEqual(old,[{k:v for k,v in row.items() if k!="score_temperature"} for row in new])
        self.assertEqual(old_ledger["teacher_scoring"],new_ledger["teacher_scoring"])
        self.assertEqual(L.total_flop_proxy(old_ledger),L.total_flop_proxy(new_ledger))
    def test_temperature_point_eight_matches_manual_action_only_logsoftmax(self):
        ledger={};scores=R.score(self.model,self.tok,self.rows,3,ledger,score_temperature=.8)
        batch=L.collate(self.rows,self.tok.pad_id)
        with torch.no_grad():distribution=(self.model(batch["inputs"]).float()/.8).log_softmax(-1)
        for i,(row,score) in enumerate(zip(self.rows,scores)):
            values=[distribution[i,len(row["prompt_ids"])-1+j,token].item() for j,token in enumerate(row["response_ids"])]
            expected=-sum(values)/math.log(2)
            self.assertAlmostEqual(expected,score["bits"],delta=3e-5)
        self.assertEqual(ledger["reference_scoring_contract"]["score_temperature"],.8)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    def test_invalid_or_mixed_temperatures_are_rejected(self):
        for value in (0,-1,float("inf"),float("nan")):
            with self.assertRaises(ValueError):R.score(self.model,self.tok,self.rows,score_temperature=value)
        ledger={};R.score(self.model,self.tok,self.rows,ledger=ledger)
        with self.assertRaises(ValueError):R.score(self.model,self.tok,self.rows,ledger=ledger,score_temperature=.8)

if __name__=="__main__":unittest.main(verbosity=2)
