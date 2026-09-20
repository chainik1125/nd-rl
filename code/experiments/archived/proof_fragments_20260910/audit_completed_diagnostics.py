"""CPU-only binding, reward replay and ledger review of completed diagnostics."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import common
import initial_cuts as I
import learning as L
import nd_environment as E
import projected_loo as P
from release_populations import check_record


def completed(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    assert done["ok"] and done["result_sha256"] == common.sha(directory / "result.json")
    P.check_source_files(manifest)
    return manifest, result


def run():
    root = common.ROOT
    projected = root / "results/projected_loo_v1"
    audit = root / "results/difficulty_audit_v1"
    fit = root / "results/difficulty_fit_v1"
    pm, pr = completed(projected)
    am, ar = completed(audit)
    fm, fr = completed(fit)
    for path, digest in pm["input_sha256"].items():
        assert common.sha(path) == digest
    checkpoint = Path(pm["args"]["checkpoint"])
    assert common.sha(checkpoint) == pm["contract"]["checkpoint_sha256"]
    assert am["contract"] == fm["contract"] == pm["contract"]
    assert pr["weights_unchanged_exactly"] is True
    assert pr["policy_updates"] == pr["new_rollouts"] == ar["policy_updates"] == 0
    assert pr["ranking_changed"] is pr["final_data_accessed"] is False
    assert pr["new_total_flop_proxy"] == L.total_flop_proxy(pr["ledger"])
    assert pr["selection_gradient_sha256"] == common.sha(projected / "selection_gradient.pt")
    assert set(pr["ledger"]) == {"single_frozen_query_gradient", "original", "fresh_audit"}
    assert ar["fresh_audit_flop_proxy"] == L.total_flop_proxy(ar["ledger"])
    initial = json.loads((fit / "initial_cuts.json").read_text())
    assert common.sha(fit / "initial_cuts.json") == fr["initial_cuts_sha256"] == ar["initial_cuts_sha256"]
    assert initial["new_pilot_flop_proxy"] == L.total_flop_proxy(initial["fit_compute_ledger"])
    assert initial["total_pilot_flop_proxy"] == initial["new_pilot_flop_proxy"] + initial["reused_original_full_run_flop_proxy"]
    assert ar["total_fit_plus_audit_flop_proxy"] == initial["total_pilot_flop_proxy"] + ar["fresh_audit_flop_proxy"]
    candidate_path = pm["args"]["candidates"]
    candidates = common.read_jsonl(candidate_path)
    assert len(candidates) == 768
    for row in candidates:
        check_record(row, "candidate", 6)
    by_source = {r["source_id"]: r for r in candidates}
    tok = E.Tokenizer()
    expected = {sid: E.episode(r, 0, tok) for sid, r in by_source.items()}
    assert common.digest(tok.state_dict()) == pm["tokenizer_sha256"]
    cuts = I.validate_map(initial, candidates, tok, common.sha(checkpoint), .8, 512, "fp32")
    for sid, entry in initial["sources"].items():
        chosen = I.choose_cut(entry["successes"])
        assert all(entry[k] == v for k, v in chosen.items() if k != "successes")
        assert entry["admissible_cuts"] == L.admissible_cuts(by_source[sid])
    audit_summary = {}
    for index, condition in enumerate(("chosen", "full", "latest")):
        seed = am["args"]["seed"] + index * 1000003
        assert seed not in {initial["fit_seed"], *initial["reused_producer_seeds"]}
        rows = common.read_jsonl(audit / f"{condition}_attempts.jsonl")
        assert len(rows) == 6144
        groups = defaultdict(list)
        for row in rows:
            sid = row["source_id"]
            cut = cuts[sid] if condition == "chosen" else 0 if condition == "full" else max(L.admissible_cuts(by_source[sid]))
            state = E.episode(by_source[sid], cut, tok)
            for key in ("prompt_ids", "original_prompt_ids", "prefix_ids", "reset_index", "source_lines", "format"):
                assert row[key] == state[key]
            assert row["condition"] == condition and row["sampling_seed"] == seed
            assert row["effective_max_new_tokens"] == 512
            assert E.judge(state, row["response_ids"], tok)["reward"] == row["reward"]
            groups[sid].append(row)
        assert set(groups) == set(expected)
        assert all(sorted(r["attempt_index"] for r in group) == list(range(8)) for group in groups.values())
        counts = [sum(r["reward"] for r in group) for group in groups.values()]
        declared = ar["conditions"][condition]
        assert sum(counts) == declared["successes"]
        assert sum(0 < k < 8 for k in counts) == declared["mixed_groups"]
        assert declared["total_flop_proxy"] == L.total_flop_proxy(ar["ledger"][condition])
        assert declared["likelihood_audit"]["max_abs_error"] <= .001
        audit_summary[condition] = {"sources": len(groups), "attempts": len(rows), "successes": sum(counts),
                                    "mixed_groups": sum(0 < k < 8 for k in counts), "seed": seed}
    banks, _ = P.load_banks(pm["args"]["banks_manifest"], by_source, common.sha(candidate_path))
    projections = {}
    reused = 0
    for binding in pm["draw_bindings"]:
        kind = binding["kind"]
        rebound = P.bind_draw_directory(binding["directory"], kind, pm["contract"], common.sha(candidate_path))
        assert rebound == binding
        groups = P.load_groups(binding, expected, tok)
        estimates_path = projected / f"{kind}_source_estimates.jsonl"
        assert common.sha(estimates_path) == pr["draw_sets"][kind]["source_estimates_sha256"]
        estimates = common.read_jsonl(estimates_path)
        assert len(estimates) == 768 and {r["source_id"] for r in estimates} == set(expected)
        for row in estimates:
            group = groups[row["source_id"]]
            k = sum(x["reward"] for x in group)
            assert row["successes"] == k and row["group_size"] == 8
            assert row["D_div_512"] == row["D"] / 512
            assert row["mixed"] == row["backward_executed"] == (0 < k < 8)
            assert row["action_group_sha256"] == common.digest([x["response_ids"] for x in group])
            assert row["behavior_group_sha256"] == common.digest([x["rollout_logprobs"] for x in group])
            assert row["state_sha256"] == I.state_hash(expected[row["source_id"]])
            if k in (0, 8):
                assert row["D"] == 0
        contrasts = P.bank_contrasts({r["source_id"]: r["D"] for r in estimates}, banks)
        assert contrasts == pr["draw_sets"][kind]["family_contrasts"]
        local = pr["draw_sets"][kind]["ledger"]
        assert sum(r["mixed"] for r in estimates) == local["raw_loo_teacher_backward"]["calls"]
        assert local == pr["ledger"][kind]
        assert pr["draw_sets"][kind]["new_derivative_and_audit_flop_proxy"] == L.total_flop_proxy(local)
        projections[kind] = {"sources": 768, "mixed_groups": sum(r["mixed"] for r in estimates),
                             "algebraic_zero_groups": sum(not r["mixed"] for r in estimates)}
        reused += binding["reused_previously_charged_flop_proxy"]
    assert reused == pr["reused_previously_charged_flop_proxy"]
    plan = json.loads((fit / "plan.json").read_text())
    receipt = {"status": "passed", "audit_conditions": audit_summary, "projected_draw_sets": projections,
               "new_fit_attempts": plan["new_attempts"], "new_audit_attempts": 18432,
               "new_projected_attempts": 0, "new_fit_proxy": initial["new_pilot_flop_proxy"],
               "new_audit_proxy": ar["fresh_audit_flop_proxy"], "new_projected_proxy": pr["new_total_flop_proxy"],
               "additional_fit_audit_projection_proxy": initial["new_pilot_flop_proxy"] + ar["fresh_audit_flop_proxy"] + pr["new_total_flop_proxy"],
               "checkpoint_sha256": common.sha(checkpoint), "no_policy_updates": True,
               "weight_check_scope": "Completed projected executable asserts torch.equal against every initial parameter and clears gradients; this CPU audit checks immutable checkpoint/source/result hashes. Audit sampler has no optimizer or weight-mutating operation.",
               "gradient_numeric_recomputed": False, "gpu_work": False, "final_data_accessed": False,
               "avoid_double_counting": "Add only new_fit_proxy + new_audit_proxy + new_projected_proxy to already charged candidate diagnostics; reused subtotals are provenance receipts.",
               "source_sha256": common.sha(__file__),
               "producer_result_sha256": {str(d): common.sha(d / "result.json") for d in (fit, audit, projected)}}
    common.write_json(root / "code/tests/completed_diagnostics_independent_v1.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    run()
