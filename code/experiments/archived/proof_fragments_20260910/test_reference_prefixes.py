import unittest
import common
from populations import identities
from mine_failed_prefixes import local_prefixes
from enumerate_reference_prefixes import canonical_prefix_key,compare


class ReferencePrefixTests(unittest.TestCase):
    def test_exact_key_is_atom_invariant_but_preserves_citation_order(self):
        first=("THM P SEQ ( P v Q ) PRF","N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED")
        renamed=("THM R SEQ ( R v S ) PRF","N1 R : PR ; N2 ( R v S ) : ORI1 N1 ; QED")
        self.assertEqual(canonical_prefix_key(*first),canonical_prefix_key(*renamed))
        p="THM P , Q SEQ ( P & Q ) PRF";b="N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"
        q="THM Q , P SEQ ( P & Q ) PRF";c="N1 Q : PR ; N2 P : PR ; N3 ( P & Q ) : ANDI N2 N1 ; QED"
        self.assertEqual(identities(p)["theorem_id"],identities(q)["theorem_id"])
        self.assertNotEqual(canonical_prefix_key(p,b),canonical_prefix_key(q,c))
    def test_overlap_uses_all_prefixes_before_endpoint_dedup(self):
        prompt="THM P SEQ ( P v Q ) PRF"
        short="N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; QED"
        long="N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; N3 ( P v Q ) : R N2 ; QED"
        def row(body):return {"prompt":prompt,"body":body,**identities(prompt),"source_id":common.digest(body),
            "canonical_prefix_id":canonical_prefix_key(prompt,body),"family":"ORI1","n_lines":body.count(";"),
            "provenance":{"parent_source_id":"parent","parent_full_source_lines":4}}
        a,b=row(long),row(short);result=compare([b],[a,b],[a])[0]
        self.assertTrue(result["endpoint_in_candidate_reference_prefixes"])
        self.assertTrue(result["exact_canonical_prefix_in_candidate_reference_prefixes"])
        self.assertFalse(result["exact_canonical_prefix_in_deduplicated_representatives"])
    def test_closed_local_goals_can_change_original_context_skeleton(self):
        prompt="THM SEQ ( ( P > P ) & ( Q > Q ) ) PRF"
        body="N1 | P : AS ; N2 ( P > P ) : IMPI N1 N1 ; N3 | Q : AS ; N4 ( Q > Q ) : IMPI N3 N3 ; N5 ( ( P > P ) & ( Q > Q ) ) : ANDI N2 N4 ; QED"
        found,audit=local_prefixes(prompt,body)
        self.assertIsNone(audit["first_invalid_reason"]);self.assertEqual(len(found),2)
        self.assertTrue(all(row["context_id"]!=identities(prompt)["context_id"] for row in found))


if __name__=="__main__":unittest.main(verbosity=2)
