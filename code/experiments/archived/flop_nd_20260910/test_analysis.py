"""Tests of analysis estimands, pairing, and raw-success integrity."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import common
from analyze_lm import paired_context_bootstrap, read_evaluation, summary_rows


class AnalysisTests(unittest.TestCase):
    def test_cluster_weighting_and_pairing(self):
        # One three-theorem context and one singleton. The target is 3/4,
        # not the unweighted context mean 1/2. Exhaustive bootstrap support
        # is {0, .75, 1}; deterministic RNG must reproduce it exactly.
        estimate=paired_context_bootstrap([1,1,1,0],["A","A","A","B"],4000,5)
        self.assertEqual(estimate["estimate"],.75)
        self.assertEqual(estimate["ci95_context_bootstrap"],[0.,1.])
        constant=paired_context_bootstrap([.5]*4,["A","A","A","B"],100,5)
        self.assertEqual(constant["ci95_context_bootstrap"],[.5,.5])
        null=paired_context_bootstrap(np.zeros(4),["A","A","A","B"],100,5)
        self.assertEqual(null["estimate"],0.)
        self.assertEqual(null["ci95_context_bootstrap"],[0.,0.])

    def write_eval(self,root,rows,n=2):
        (root/"audit.jsonl").write_text("frozen audit fixture")
        (root/"checkpoint.pt").write_text("frozen checkpoint fixture")
        inputs={str(root/name):common.sha(root/name) for name in ("audit.jsonl","checkpoint.pt")}
        common.write_jsonl(root/"attempts.jsonl",rows)
        common.write_json(root/"result.json",{"status":"completed","attempts_sha256":common.sha(root/"attempts.jsonl"),
            "input_sha256":inputs,"sampled_attempts":n,"temperature":.8,"per_theorem_token_cap":n*512})
        common.write_json(root/"config.json",{"args":{"attempts":n,"checkpoint":str(root/"checkpoint.pt"),"audit":str(root/"audit.jsonl"),
            "temperature":.8,"max_new_tokens":512,"batch_size":32,"seed":7,"precision":"fp32"},
            "input_sha256":inputs})

    def read_eval(self,root,audit):
        return read_evaluation(root,audit,common.sha(root/"audit.jsonl"))

    def row(self,attempt,reward,length=8):
        row={"source_id":"one","prompt":"theorem","attempt":attempt,"reward":reward,
             "reset_index":0,"generated_tokens":100,"independent_ok":bool(reward),"n_lines":length}
        if reward:row["compression"]={"raw_valid":True,"dependency_pruned_length":length,"compressed_length":length}
        return row

    def test_success_lengths_keep_raw_failure_and_shorter_witness_distinct(self):
        audit={"one":{"prompt":"theorem"}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.write_eval(root,[self.row(0,0),self.row(1,1)])
            result=self.read_eval(root,audit)["rows"]["one"]
            self.assertEqual((result["first"],result["any"],result["any_pruned_over6"]),(0,1,1))
            self.write_eval(root,[self.row(0,1,8),self.row(1,1,4)])
            result=self.read_eval(root,audit)["rows"]["one"]
            self.assertEqual(result["any_pruned_over6"],1)
            self.assertEqual(result["shortest_found_pruned_over6"],0)
            self.assertEqual(result["shortest_pruned"],4)
            self.assertEqual(summary_rows({"one":result},["one"])["maximum_pruned"],8)

    def test_invalid_evaluation_is_rejected(self):
        audit={"one":{"prompt":"theorem"}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.write_eval(root,[self.row(0,1),self.row(0,1)])
            with self.assertRaisesRegex(ValueError,"Missing or duplicate"):
                self.read_eval(root,audit)
            row=self.row(0,1);row["independent_ok"]=False
            self.write_eval(root,[row],1)
            with self.assertRaisesRegex(ValueError,"independent verification"):
                self.read_eval(root,audit)
            self.write_eval(root,[self.row(0,1)],1)
            with (root/"attempts.jsonl").open("a") as stream:stream.write("\n")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                self.read_eval(root,audit)
            self.write_eval(root,[self.row(0,1)],1)
            with self.assertRaisesRegex(ValueError,"analysis audit identity"):
                read_evaluation(root,audit,"altered audit identity")


if __name__=="__main__":unittest.main()
