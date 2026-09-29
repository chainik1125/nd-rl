"""Pure-prompt nonlearned comparisons, using isolated generic dependencies."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import common
from classical_vendor.truth_assignment import TruthAssignmentConstructor, ConstructionBudgetExceeded
from classical_vendor.engine import OrderedSearch, BudgetExceeded, proof_size, formula_size
from classical_vendor.support import CertificateOutputLimitExceeded
from nd_environment import parse_prompt, strict_complete
from nd_data import formula_text
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof

METHODS = [
    {"name": "heuristic_128", "kind": "heuristic", "budget": 128, "max_depth": 10, "seed": 0},
    {"name": "heuristic_4096", "kind": "heuristic", "budget": 4096, "max_depth": 10, "seed": 0},
    {"name": "truth_assignment_100000", "kind": "constructor", "budget": 100000, "atom_order": "alphabetical"},
]
MAX_LINES = 10000
MAX_RESPONSE_TOKENS = 1000000
BUDGETS = (512, 1024, 2048)


def truth(formula, assignment):
    kind = formula[0]
    if kind == "atom":
        return assignment[formula[1]]
    if kind == "bot":
        return False
    if kind == "not":
        return not truth(formula[1], assignment)
    a, b = truth(formula[1], assignment), truth(formula[2], assignment)
    if kind == "and":
        return a and b
    if kind == "or":
        return a or b
    if kind == "imp":
        return not a or b
    raise ValueError("Unsupported connective")


def response_tokens(body):
    """Exact pinned relative-codec length, without allocating unary token arrays."""
    lines = parse_proof_tokens(body.split())
    return 2 + sum(4 + row["depth"] + len(formula_text(row["formula"]).split())
                   + sum(row["idx"] - ref + 2 for ref in row["refs"]) for row in lines)


def code_identity():
    paths = [Path(__file__), Path(__file__).with_name("classical_run.py"), Path(__file__).with_name("nd_environment.py"),
             Path(__file__).with_name("common.py"), common.VENDOR / "nd_data.py", common.VENDOR / "nd_model.py",
             common.VENDOR / "nd_verify/verify.py", common.VENDOR / "strict_verify.py", common.VENDOR / "proof_diagnostics.py"]
    paths += sorted(Path(__file__).with_name("classical_vendor").glob("*.py"))
    return {str(p.resolve()): common.sha(p) for p in paths}


def solve(prompt, method):
    """Construct and verify from the theorem only; caller imposes hard timeout."""
    started = time.monotonic()
    result = {"method": method["name"], "method_config": method, "status": "not_started", "reward": 0,
              "operation_budget": method["budget"], "no_learned_policy": True,
              "max_certificate_lines": MAX_LINES, "max_certificate_response_tokens": MAX_RESPONSE_TOKENS,
              "reference_labels_loaded": False, "classical_dn_supported": True}
    engine = None
    try:
        premises, goal = parse_prompt(prompt)
        proof_size.cache_clear()
        formula_size.cache_clear()
        if method["kind"] == "heuristic":
            engine = OrderedSearch(method["budget"], "heuristic", method["seed"], method["max_depth"])
            body = engine.solve(premises, goal)
            result["status"] = "search_exhausted" if body is None else "constructed"
        elif method["kind"] == "constructor":
            engine = TruthAssignmentConstructor(method["budget"], method["atom_order"])
            constructed = engine.solve(prompt)
            if constructed["status"] == "countermodel":
                assignment = constructed["countermodel"]
                if not all(truth(p, assignment) for p in premises) or truth(goal, assignment):
                    raise ValueError("Constructor countermodel failed independent truth evaluation")
                result.update(status="countermodel", countermodel=assignment, countermodel_independently_checked=True)
                body = None
            else:
                body = constructed["body"]
                result["status"] = "constructed"
        else:
            raise ValueError("Unsupported fixed method")
        if body is not None:
            raw = strict_complete(prompt, body)
            if not raw["ok"]:
                result.update(status="invalid_certificate", reason=raw["reason"], raw_strict_valid=False)
            else:
                raw_tokens = response_tokens(body)
                if raw["n_lines"] > MAX_LINES or raw_tokens > MAX_RESPONSE_TOKENS:
                    raise CertificateOutputLimitExceeded("postchecked_output", MAX_RESPONSE_TOKENS, raw_tokens)
                compression = compress_proof(prompt, body)
                pruned_body = compression["compressed_proof"]
                checked = strict_complete(prompt, pruned_body)
                if not checked["ok"]:
                    raise ValueError("Pruned certificate failed independent strict checker")
                pruned_tokens = response_tokens(pruned_body)
                if checked["n_lines"] > MAX_LINES or pruned_tokens > MAX_RESPONSE_TOKENS:
                    raise CertificateOutputLimitExceeded("pruned_output", MAX_RESPONSE_TOKENS, pruned_tokens)
                result.update(status="verified", reward=1, raw_strict_valid=True, pruned_strict_valid=True,
                    raw_certificate=body, pruned_certificate=pruned_body,
                    raw_certificate_sha256=hashlib.sha256(body.encode()).hexdigest(),
                    pruned_certificate_sha256=hashlib.sha256(pruned_body.encode()).hexdigest(),
                    raw_lines=raw["n_lines"], pruned_lines=checked["n_lines"],
                    dependency_pruned_lines=compression["dependency_pruned_length"],
                    raw_response_tokens=raw_tokens, pruned_response_tokens=pruned_tokens,
                    raw_budget_eligibility={str(b): raw_tokens <= b for b in BUDGETS},
                    pruned_budget_eligibility={str(b): pruned_tokens <= b for b in BUDGETS})
    except (BudgetExceeded, ConstructionBudgetExceeded):
        result["status"] = "operation_budget_exhausted"
    except CertificateOutputLimitExceeded as error:
        result.update(status="certificate_output_censored", censor_unit=error.unit, censor_limit=error.limit,
                      attempted_output_size=error.attempted, reward=0,
                      note="No truncated certificate is counted as a success")
    except Exception as error:
        result.update(status="exception", exception_type=type(error).__name__, reason=str(error), reward=0)
    finally:
        if engine is not None:
            if method["kind"] == "heuristic":
                result.update(operations=engine.nodes, operation_components=dict(engine.counts),
                              operation_unit="subgoal visits + backward action attempts + new forward derivations")
            else:
                result.update(operations=engine.operations, operation_components=dict(engine.counts),
                              operation_unit="partial formula evaluations + decision visits + constructed proof nodes")
        result["elapsed_seconds"] = time.monotonic() - started
        proof_size.cache_clear()
        formula_size.cache_clear()
    return result
