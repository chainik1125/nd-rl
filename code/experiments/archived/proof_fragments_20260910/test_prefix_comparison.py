import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import common
import learning as L
import nd_environment as E
from prepare_prefix_comparison import make_pairs,full_encoding,ancestry_flags
from diagnostic_certificate_sft import validate_release


class PrefixComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).parent/"diagnostics"
        cls.mined=common.read_jsonl(root/"failed_rollout_first1000_v1/diagnostic_certificates.jsonl")
        cls.references=common.read_jsonl(root/"candidate_reference_prefixes_v1/diagnostic_certificates.jsonl")
        cls.tok=E.Tokenizer()
    def test_exact_length_distinct_disjoint_deterministic_cohorts(self):
        first=make_pairs(self.mined,self.references,self.tok,20260910)
        second=make_pairs(self.mined,self.references,self.tok,20260910)
        ids=lambda value:[(a["source_id"],b["source_id"]) for a,b in value[0]]
        self.assertEqual(ids(first),ids(second));self.assertEqual(len(first[0]),84)
        self.assertEqual(len(first[1]),1);self.assertEqual(first[1][0]["n_lines"],6)
        self.assertEqual(len({b["theorem_id"] for a,b in first[0]}),84)
        all_mined={r["theorem_id"] for r in self.mined}
        for a,b in first[0]:
            self.assertEqual(a["n_lines"],b["n_lines"]);self.assertNotIn(b["theorem_id"],all_mined)
    def test_missing_control_stratum_fails_without_duplication(self):
        references=[r for r in self.references if r["n_lines"]!=5]
        with self.assertRaises(ValueError):make_pairs(self.mined,references,self.tok,1)
    def test_full_supervision_roundtrip_and_zero_context_mask(self):
        row=next(r for r in self.mined if r["n_lines"]==5);encoded=full_encoding(row,self.tok)
        self.assertEqual(encoded["reset_index"],0);self.assertEqual(encoded["prefix_ids"],[])
        batch=L.collate([encoded],self.tok.pad_id)
        self.assertEqual(int(batch["mask"].sum()),len(encoded["response_ids"]))
        self.assertFalse(bool(batch["mask"][0,:len(encoded["prompt_ids"])-1].any()))
        self.assertTrue(E.judge(encoded,encoded["response_ids"],self.tok)["reward"])
    def test_acquired_reference_exposure_is_not_supervised_ancestry(self):
        row={"theorem_id":"x"}
        review={"intermediate_exposure_hits":{"x":[{"role":"candidate"}]},"proof_exposure_hits":{"x":[{"role":"selection"}]}}
        flags=ancestry_flags(row,review)
        self.assertFalse(flags["in_parent_supervised_ancestry_endpoint_inventory"])
        self.assertFalse(flags["in_parent_supervised_ancestry_exact_proof_inventory"])
        self.assertTrue(flags["in_acquired_candidate_or_selection_endpoint_inventory"])
        self.assertTrue(flags["in_acquired_candidate_or_selection_exact_proof_inventory"])
        review["intermediate_exposure_hits"]["x"].append({"role":"baseline"})
        self.assertTrue(ancestry_flags(row,review)["in_parent_supervised_ancestry_endpoint_inventory"])
    def test_training_requires_separate_explicit_release_and_exact_config(self):
        args=SimpleNamespace(supervision="full-certificate",steps=100,seed=1,lr=5e-5,batch_size=32,fixed_pad=512,retention_fraction=.5,precision="fp32")
        release={"frozen":True,"approved_for_diagnostic_full_certificate_sft":True,"proposal_manifest_sha256":"p",
                 "cohort_sha256":["c"],"checkpoints":["m"],"retention_sha256":"r","supervision":"full-certificate","full_source_supervision_cap":6,
                 "training_config":{k:getattr(args,k) for k in ("steps","seed","lr","batch_size","fixed_pad","retention_fraction","precision")}}
        validate_release(args,release,"p","c","m","r")
        for field in ("frozen","approved_for_diagnostic_full_certificate_sft"):
            bad=copy.deepcopy(release);bad[field]=False
            with self.assertRaises(ValueError):validate_release(args,bad,"p","c","m","r")
        bad=copy.deepcopy(release);bad["training_config"]["steps"]=101
        with self.assertRaises(ValueError):validate_release(args,bad,"p","c","m","r")
        with self.assertRaises(ValueError):validate_release(args,release,"other","c","m","r")


if __name__=="__main__":unittest.main(verbosity=2)
