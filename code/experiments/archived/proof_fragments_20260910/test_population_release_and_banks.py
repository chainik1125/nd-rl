"""CPU-only integrity checks on the previously generated tiny smoke corpus."""
import copy
from pathlib import Path
import unittest
import common
from populations import audit
from release_populations import check_record, validate_roles
from make_bank import validated_pool, select_families, uniform_match, describe

FIXTURE=Path(__file__).parent/"tests/synthetic_smoke_v2"

def tiny_populations():
    roles=("candidate","adaptation","retention","selection","confirmation")
    result={r:common.read_jsonl(FIXTURE/f"{r}.jsonl") for r in roles}
    result["final_confirmation"]=[]
    return result

def validate_tiny(populations):
    families=sorted({r["family"] for r in populations["candidate"]})
    validate_roles(populations,families,2,{"adaptation":16,"retention":8},
                   {"selection":2,"confirmation":2,"final_confirmation":0},[set(),set(),set()])

class ReleaseIntegrityTests(unittest.TestCase):
    def test_tiny_integrity_and_existing_context_partition(self):
        populations=tiny_populations();validate_tiny(populations)
        self.assertTrue(audit(populations)["ok"])

    def test_duplicate_source_is_rejected_even_if_family_count_unchanged(self):
        populations=tiny_populations();rows=populations["candidate"]
        first=rows[0];i=next(i for i,r in enumerate(rows[1:],1) if r["family"]==first["family"])
        rows[i]=copy.deepcopy(first)
        with self.assertRaisesRegex(ValueError,"Duplicate"):validate_tiny(populations)

    def test_declared_cap_and_source_hash_cannot_override_actual_certificate(self):
        row=copy.deepcopy(tiny_populations()["candidate"][0])
        row["provenance"]["full_source_lines"]+=1
        with self.assertRaisesRegex(ValueError,"line counts"):check_record(row,"candidate",6)
        row=copy.deepcopy(tiny_populations()["candidate"][0]);row["source_id"]="0"*64
        with self.assertRaisesRegex(ValueError,"certificate identity"):check_record(row,"candidate",6)

    def test_final_confirmation_role_is_explicit_and_baseline_exclusion_enforced(self):
        populations=tiny_populations()
        populations["final_confirmation"]=[copy.deepcopy(populations["confirmation"][0])]
        with self.assertRaisesRegex(ValueError,"Expected role final_confirmation"):validate_tiny(populations)
        with self.assertRaisesRegex(AssertionError,"baseline exposure"):
            audit(tiny_populations(),[FIXTURE/"selection.jsonl"])

    def test_textbook_blind_hashes_are_rechecked(self):
        populations=tiny_populations();blocked=populations["candidate"][0]["guard_ids"][0]
        with self.assertRaisesRegex(ValueError,"guard collision"):
            validate_roles(populations,sorted({r["family"] for r in populations["candidate"]}),2,
                {"adaptation":16,"retention":8},{"selection":2,"confirmation":2,"final_confirmation":0},[{blocked},set(),set()])

class BankIntegrityTests(unittest.TestCase):
    def setUp(self):self.rows=validated_pool(tiny_populations()["candidate"])

    def test_selected_and_uniform_exact_length_match_without_replacement(self):
        selected=select_families(self.rows,["ANDI","IMPI_nested","DN"],2,6,17)
        uniform=uniform_match(self.rows,selected,6,31)
        self.assertEqual(len({r["source_id"] for r in uniform}),6)
        self.assertEqual(sorted(r["n_lines"] for r in selected),sorted(r["n_lines"] for r in uniform))
        self.assertEqual(uniform,uniform_match(list(reversed(self.rows)),selected,6,31))
        self.assertEqual(describe(uniform)["unique_theorems"],6)

    def test_reference_must_be_exact_pool_subset(self):
        selected=select_families(self.rows,["ANDI","IMPI_nested","DN"],2,6,17)
        modified=copy.deepcopy(selected);modified[0]["body"]+=" "
        with self.assertRaisesRegex(ValueError,"exact subset"):uniform_match(self.rows,modified,6,31)
        with self.assertRaisesRegex(ValueError,"source uniqueness"):
            uniform_match(self.rows,[selected[0]]*6,6,31)

    def test_missing_family_and_diagnostic_source_cannot_be_silently_padded(self):
        with self.assertRaisesRegex(ValueError,"distinct-source quotas"):
            select_families(self.rows,["ANDI","IMPI_nested","ORE"],2,6,17)
        rows=copy.deepcopy(self.rows);rows[0]["diagnostic_only"]=True
        with self.assertRaisesRegex(ValueError,"Diagnostic"):validated_pool(rows)

if __name__=="__main__":unittest.main(verbosity=2)
