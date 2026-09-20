import unittest
import nd_environment as E
from mine_failed_prefixes import local_prefixes,decode_failed_attempt,role_for_context


class FailedPrefixTests(unittest.TestCase):
    def test_retains_closed_new_goal_before_first_bad_inference(self):
        prompt="THM P SEQ R PRF"
        body="N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; N3 R : DN N2 ; N4 ( P & P ) : ANDI N1 N1 ; QED"
        found,audit=local_prefixes(prompt,body)
        self.assertEqual(len(found),1);self.assertEqual(found[0]["n_lines"],2)
        self.assertEqual(E.parse_prompt(found[0]["prompt"])[0],E.parse_prompt(prompt)[0])
        self.assertTrue(E.strict_complete(found[0]["prompt"],found[0]["body"])["ok"])
        self.assertFalse(E.strict_complete(prompt,body)["ok"])
        self.assertIsNotNone(audit["first_invalid_reason"])
    def test_open_scope_premise_and_original_goal_are_not_extracted(self):
        prompt="THM P SEQ ( P v Q ) PRF"
        body="N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; N3 | R : AS ; N4 | ( R & R ) : ANDI N3 N3 ; QED"
        found,audit=local_prefixes(prompt,body)
        self.assertEqual(found,[])
        self.assertEqual(audit["prefix_checks"]["open_scope_prefix"],2)
        self.assertEqual(audit["prefix_checks"]["merely_premise_conclusion"],1)
        self.assertEqual(audit["prefix_checks"]["original_target_conclusion"],1)
    def test_retained_certificate_cap_and_partial_boundary(self):
        prompt="THM P SEQ R PRF"
        lines=["N1 P : PR ;"]+[f"N{i} ( P v Q ) : ORI1 N1 ;" for i in range(2,9)]
        found,_=local_prefixes(prompt," ".join(lines)+" N9")
        self.assertEqual([r["n_lines"] for r in found],[2,3,4,5,6])
    def test_boundary_decoder_and_public_partition(self):
        tok=E.Tokenizer()
        self.assertEqual(decode_failed_attempt({"response_ids":[tok.eos_id]},tok)[1],"ambiguous_or_missing_FINAL")
        for bucket in range(20):
            context=f"{bucket:012x}"+"0"*52
            self.assertEqual(role_for_context(context)=="candidate",bucket<6)


if __name__=="__main__":unittest.main(verbosity=2)
