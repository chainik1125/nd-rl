"""First-proof online replay bank and provenance-checked main-run imports.

Only strictly verified model responses can enter this bank. Externally supplied
source certificates continue to pass the unchanged cap-six loader separately.
The bank never reads an audit-reference population or replaces its first proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import common
import learning as L
from nd_environment import Tokenizer, episode, judge, strict_complete

IMPORT_VERSION = "flop-main-discovery-import-v1"
MAIN_VERSION = "flop-nd-token-ppo-v1"


def advance_curriculum(stats, positions, rewards, *, mode="window", window=16,
                       backward_threshold=.9, forward_threshold=0.,
                       ema_alpha=.4, force_episodes=20000):
    """Consume outcomes generated at the current cut and possibly move earlier.

    FLoP-style EMA decisions occur per completed episode. Once a cut advances,
    remaining outcomes in the same pre-generated group belong to the old cut
    and are not credited to the new cut. EMA mode never backs off on failure.
    """
    current = positions.index(stats["cut"])
    if mode == "window":
        stats["n"] += len(rewards)
        stats["success"] += sum(rewards)
        if stats["n"] < window:
            return None
        rate = stats["success"] / stats["n"]
        new = max(0, current - 1) if rate >= backward_threshold else (
              min(len(positions) - 1, current + 1) if rate < forward_threshold else current)
        event = {"mode": mode, "old": stats["cut"], "new": positions[new],
                 "n": stats["n"], "success": stats["success"], "rate": rate}
        stats.update(cut=positions[new], n=0, success=0)
        return event
    if mode != "flop_ema" or not 0 < ema_alpha <= 1 or force_episodes < 0:
        raise ValueError("Invalid curriculum mode, EMA alpha or forced-advance interval")
    for index, reward in enumerate(rewards):
        stats["n"] += 1
        stats["success"] += reward
        stats["ema"] = ema_alpha * reward + (1 - ema_alpha) * stats.get("ema", 0.)
        success_trigger = stats["ema"] > backward_threshold
        force_trigger = force_episodes > 0 and stats["n"] > force_episodes
        if current > 0 and (success_trigger or force_trigger):
            event = {"mode": mode, "old": stats["cut"], "new": positions[current - 1],
                     "n": stats["n"], "success": stats["success"], "ema": stats["ema"],
                     "trigger": "ema" if success_trigger else "forced_episode_count",
                     "old_cut_group_outcomes_unused_after_advance": len(rewards) - index - 1}
            stats.update(cut=positions[current - 1], n=0, success=0, ema=0.)
            return event
    return None


def stream_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class OnlineProofBank:
    def __init__(self, targets, tok, *, max_seq_len, max_new_tokens, records=None):
        self.targets = {row["source_id"]: row["prompt"] for row in targets}
        self.tok = tok
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.records = {}
        self.cuts = {}
        self.episodes = {}
        for record in (records or {}).values():
            self._restore(record)

    def _restore(self, record):
        source_id = record["source_id"]
        if source_id not in self.targets or record["prompt"] != self.targets[source_id]:
            raise ValueError("Online bank theorem is not an active theorem-only target")
        if record.get("external_supervision") is not False:
            raise ValueError("Replay bank refuses externally supervised proofs")
        provenance = record.get("provenance", {})
        if provenance.get("kind") not in ("own_online_rollout", "verified_main_checkpoint_import"):
            raise ValueError("Online proof lacks an admitted provenance kind")
        # Retain the original sampled full-start response for independent checks.
        response = record["full_start_response_ids"]
        verdict = judge({"prompt": record["prompt"], "prefix_ids": []}, response, self.tok)
        if not verdict["reward"] or verdict["body"] != record["body"]:
            raise ValueError("Archived response does not reconstruct the stored valid proof")
        if record["n_lines"] != verdict["n_lines"] or record["response_ids_sha256"] != common.digest(response):
            raise ValueError("Archived response length or token identity metadata differs")
        if record["witness_id"] != common.digest([record["prompt"], record["body"]]):
            raise ValueError("Online proof witness identity mismatch")
        available = []
        cached = {}
        for cut in L.admissible_cuts(record):
            item = episode(record, cut, self.tok)
            if len(item["prompt_ids"]) + self.max_new_tokens <= self.max_seq_len:
                item.update(training_kind="target", online_witness_id=record["witness_id"])
                available.append(cut)
                cached[cut] = item
        if 0 not in available:
            raise ValueError("Online target cannot fit the declared full-start response budget")
        self.records[source_id] = record
        self.cuts[source_id] = available
        self.episodes[source_id] = cached

    def admit(self, row, provenance):
        """Admit the first verified full-start target response, including >6 lines."""
        source_id = row["source_id"]
        if source_id in self.records:
            return None
        if row.get("training_kind") != "target" or row.get("prefix_ids", []) or row.get("reset_index", 0) != 0:
            raise ValueError("A first online proof must come from a full-start theorem-only target attempt")
        if row.get("finish_reason") != "qed":
            raise ValueError("An online proof must terminate with sampled QED")
        if source_id not in self.targets or row["prompt"] != self.targets[source_id]:
            raise ValueError("Attempt identity differs from active target")
        verdict = judge({"prompt": row["prompt"], "prefix_ids": []}, row["response_ids"], self.tok)
        if not verdict["reward"]:
            raise ValueError("Invalid sampled response cannot enter the online bank")
        if "body" in row and row["body"] != verdict["body"]:
            raise ValueError("Claimed body differs from the actual sampled tokens")
        record = {"source_id": source_id, "prompt": row["prompt"], "body": verdict["body"],
                  "n_lines": verdict["n_lines"], "witness_id": common.digest([row["prompt"], verdict["body"]]),
                  "external_supervision": False, "family": "online_discovery",
                  "full_start_response_ids": list(row["response_ids"]),
                  "response_ids_sha256": common.digest(row["response_ids"]),
                  "provenance": provenance, "independent_ok": verdict["independent_ok"]}
        self._restore(record)
        return {"source_id": source_id, "witness_id": record["witness_id"], "n_lines": record["n_lines"],
                "external_supervision": False, "provenance": provenance,
                "available_cuts": self.cuts[source_id], "initial_cut": max(self.cuts[source_id])}


def checkpoint_lineage(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("metadata", {}).get("implementation") != MAIN_VERSION:
        raise ValueError("Initial bank imports require this sprint's declared primary main-trainer checkpoint")
    state = payload["training_state"]
    run = Path(state["identity"]["config"]["output"]).resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity_sha256"] != state["identity_sha256"]:
        raise ValueError("Checkpoint and source run manifest have different learning identities")
    if common.digest(state["identity"]) != state["identity_sha256"]:
        raise ValueError("Checkpoint learning identity hash does not recompute")
    if common.digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise ValueError("Main run manifest identity hash does not recompute")
    if checkpoint.parent != run:
        # A root may copy a checkpoint for continuation. Its bytes identify the
        # source exactly; require an identical checkpoint in the declared run.
        matches = [path for path in run.glob("*.pt") if common.sha(path) == common.sha(checkpoint)]
        if not matches:
            raise ValueError("Copied checkpoint has no byte-identical checkpoint in declared main run")
    return state, run, manifest_path


def create_import_manifest(checkpoint, output):
    state, run, manifest = checkpoint_lineage(checkpoint)
    paths = {"checkpoint": Path(checkpoint).resolve(), "run_manifest": manifest,
             "discoveries": run / "discoveries.jsonl", "attempts": run / "attempts.jsonl"}
    if not paths["discoveries"].exists():
        raise ValueError("Main checkpoint run has no discovery archive; omit --initial-bank")
    result = {"version": IMPORT_VERSION, "training_identity_sha256": state["identity_sha256"],
              "checkpoint_step": state["step"],
              "files": {key: {"path": str(path), "sha256": common.sha(path)} for key, path in paths.items()},
              "admission": "First discovery per target at or before checkpoint step, matched to raw sampled full-start response and independently verified. Later rows are rejected from admission.",
              "external_supervision": False}
    common.write_json(output, result)
    return result


def import_main_discoveries(bank, manifest_path, starting_checkpoint):
    started = time.monotonic()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != IMPORT_VERSION or manifest.get("external_supervision") is not False:
        raise ValueError("Invalid initial-bank import manifest")
    files = manifest["files"]
    for key in ("checkpoint", "run_manifest", "discoveries", "attempts"):
        if common.sha(files[key]["path"]) != files[key]["sha256"]:
            raise ValueError(f"Initial-bank {key} file hash differs from frozen import manifest")
    if common.sha(starting_checkpoint) != files["checkpoint"]["sha256"]:
        raise ValueError("Initial bank must belong to the exact starting policy checkpoint lineage")
    state, run, source_manifest = checkpoint_lineage(files["checkpoint"]["path"])
    if state["identity_sha256"] != manifest["training_identity_sha256"] or state["step"] != manifest["checkpoint_step"]:
        raise ValueError("Import manifest's main checkpoint identity/step is inconsistent")
    expected_paths = {"run_manifest": source_manifest, "discoveries": run / "discoveries.jsonl", "attempts": run / "attempts.jsonl"}
    if any(Path(files[key]["path"]).resolve() != path.resolve() for key, path in expected_paths.items()):
        raise ValueError("Import files are outside the checkpoint's declared main run")
    # Discoveries are only descriptors until matched to raw sampled token rows.
    first = {}
    after_checkpoint = absent_targets = 0
    for row in stream_rows(files["discoveries"]["path"]):
        if row["step"] > state["step"]:
            after_checkpoint += 1
            continue
        if row["source_id"] not in bank.targets:
            absent_targets += 1
            continue
        if row["prompt"] != bank.targets[row["source_id"]]:
            raise ValueError("Initial discovery prompt differs from active theorem-only target")
        first.setdefault(row["source_id"], row)
    required = {(r["step"], r["source_id"], common.digest([r["prompt"], r["body"]])): r for r in first.values()}
    matched = {}
    for row in stream_rows(files["attempts"]["path"]):
        if row.get("step", 0) > state["step"] or not row.get("reward") or row.get("source_id") not in first:
            continue
        key = (row["step"], row["source_id"], common.digest([row["prompt"], row.get("body", "")]))
        if key in required and key not in matched:
            matched[key] = row
    if set(required) != set(matched):
        raise ValueError("Initial discovery has no matching sampled successful attempt before checkpoint")
    admissions = []
    for key, descriptor in required.items():
        row = matched[key]
        if row.get("training_kind") != "target" or row.get("reset_index", 0) or row.get("prefix_ids", []):
            raise ValueError("Initial discovery is not an unaided main target rollout")
        provenance = {"kind": "verified_main_checkpoint_import", "source_run_identity_sha256": state["identity_sha256"],
                      "source_checkpoint_sha256": files["checkpoint"]["sha256"], "source_checkpoint_step": state["step"],
                      "source_attempt_step": row["step"], "source_attempt_group": row.get("group"),
                      "source_attempt_index": row.get("attempt_index"),
                      "import_manifest_sha256": common.sha(manifest_path), "discoveries_sha256": files["discoveries"]["sha256"],
                      "attempts_sha256": files["attempts"]["sha256"]}
        admission = bank.admit(row, provenance)
        if admission:
            admissions.append(admission)
    return {"imported": len(admissions), "later_than_checkpoint_not_admitted": after_checkpoint,
            "outside_active_targets_not_admitted": absent_targets, "admissions": admissions,
            "manifest_sha256": common.sha(manifest_path), "elapsed_seconds": time.monotonic() - started}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = create_import_manifest(args.checkpoint, args.output)
    print(json.dumps({"output": str(args.output), "checkpoint_step": result["checkpoint_step"]}))
