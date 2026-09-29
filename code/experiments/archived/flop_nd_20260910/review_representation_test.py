"""Tests for parse-position token labels and weighted reference diagnostics."""
import math
import unittest
import torch
from nd_environment import Tokenizer,episode
import review_representation_likelihood as R

class RepresentationTests(unittest.TestCase):
    def test_r_atom_rule_and_uniform_hand_counts(self):
        tok=Tokenizer()
        record={"source_id":"test","prompt":"THM R SEQ R PRF","body":"N1 R : PR ; N2 R : R N1 ; QED",
                "n_lines":2,"length_bin":"2-6","family":"R","context_id":"c"}
        ep=episode(record,0,tok)
        labels,lines=R.parse_labels(ep["response_ids"],tok)
        rs=[labels[i]["coarse"] for i,t in enumerate(ep["response_ids"]) if tok.itos[t]=="R"]
        self.assertEqual(rs,["formula","formula","rule"])
        self.assertEqual([l["rule"] for l in lines],["PR","R"])
        n=len(labels)
        row=R.summarize_reference(record,ep,labels,[math.log(43)]*n,[False]*n)
        self.assertEqual(row["categories"]["coarse:formula"]["tokens"],2)
        self.assertEqual(row["categories"]["coarse:rule"]["tokens"],2)
        self.assertEqual(row["categories"]["coarse:citation"]["tokens"],3)
        self.assertAlmostEqual(row["total"]["bits_per_token"],math.log2(43))
        self.assertEqual(row["total"]["top1_accuracy"],0.)
        self.assertEqual(sum(row["categories"]["coarse:"+c]["tokens"] for c in ("formula","rule","citation","structure")),n)
    def test_scope_and_formula_operator(self):
        tok=Tokenizer();record={"prompt":"THM SEQ ( R > R ) PRF","body":"N1 | R : AS ; N2 ( R > R ) : IMPI N1 N1 ; QED"}
        labels,_=R.parse_labels(episode(record,0,tok)["response_ids"],tok)
        counts=R.Counter(l["fine"] for l in labels)
        self.assertEqual(counts["scope"],1);self.assertEqual(counts["formula_operator"],1)
        self.assertEqual(counts["formula_parenthesis"],2)
        self.assertEqual(counts["reference_distance"],2)
    def test_token_weighted_and_proof_weighted_are_distinct(self):
        rows=[]
        for i,(n,nll,correct) in enumerate(((2,2.,2),(8,32.,0))):
            stat=R.finish({"tokens":n,"nll_nats":nll,"top1_correct":correct})
            rows.append({"source_id":str(i),"context_id":str(i),"family":"f","length_bin":"7","total":stat,
                         "teacher_forced_all_top1":bool(correct),"categories":{"coarse:formula":stat}})
        a=R.aggregate(rows)["all"]
        self.assertEqual(a["total"]["nll_nats_per_token"],3.4)
        self.assertEqual(a["proof_mean_nll_nats_per_token"],2.5)
        self.assertEqual(a["total"]["top1_accuracy"],.2)
        self.assertEqual(a["proof_mean_top1_accuracy"],.5)

if __name__=="__main__":unittest.main(verbosity=2)
