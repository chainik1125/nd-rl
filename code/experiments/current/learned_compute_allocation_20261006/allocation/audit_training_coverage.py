"""Describe saved training sampling schedules without reading evaluation outcomes."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = ART / "allocation"


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    data = ART / "data/main_v1/train.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    families = {row["theorem_id"]: row["family"] for row in rows}
    schedules, results = {}, {}
    runs = [(arm, seed) for arm in ("cost", "no_cost") for seed in (101, 102, 103)]
    runs += [("fixed1", 101), ("fixed4", 101)]
    for arm, seed in runs:
        folder = ALLOC / arm / f"main_s{seed}"
        episodes_file = folder / "episodes.jsonl"
        selection_file = folder.with_name(folder.name + "_selection.json")
        selected = json.loads(selection_file.read_text())["selected_completed_updates"]
        schedule = {}
        with episodes_file.open() as stream:
            for line in stream:
                ep = json.loads(line)
                key = (ep["update"], ep["episode_slot"])
                if key in schedule or ep["theorem_id"] not in families:
                    raise ValueError("Duplicate schedule coordinate or non-training theorem")
                schedule[key] = (ep["theorem_id"], ep["episode_seed"])
        expected = {(update, slot) for update in range(200) for slot in range(8)}
        if set(schedule) != expected:
            raise ValueError("Incomplete 200-by-8 training metadata")
        full = [value[0] for value in schedule.values()]
        chosen = [value[0] for (update, _), value in schedule.items() if update < selected]
        schedules[(arm, seed)] = schedule
        results[f"{arm}:{seed}"] = {
            "episodes": len(full),
            "unique_theorems": len(set(full)),
            "repeat_episodes": len(full) - len(set(full)),
            "theorem_multiplicity_histogram": dict(Counter(Counter(full).values())),
            "family_episode_counts": dict(Counter(families[value] for value in full)),
            "selected_completed_updates": selected,
            "selected_age_episodes": len(chosen),
            "selected_age_unique_theorems": len(set(chosen)),
            "selected_age_family_episode_counts": dict(
                Counter(families[value] for value in chosen)
            ),
            "input_sha256": {
                str(path.relative_to(ROOT)): sha(path)
                for path in (
                    episodes_file,
                    folder / "config.json",
                    selection_file,
                )
            },
        }
    comparisons = {}
    pairs = [(("cost", seed), ("no_cost", seed)) for seed in (101, 102, 103)]
    pairs += [(("cost", 101), (arm, 101)) for arm in ("fixed1", "fixed4")]
    for left, right in pairs:
        mismatches = [
            key for key in sorted(schedules[left]) if schedules[left][key] != schedules[right][key]
        ]
        common_age = min(
            results[f"{left[0]}:{left[1]}"]["selected_completed_updates"],
            results[f"{right[0]}:{right[1]}"]["selected_completed_updates"],
        )
        comparisons[f"{left}-versus-{right}"] = {
            "full_schedule_identical": not mismatches,
            "full_mismatches": len(mismatches),
            "first_mismatch_coordinates": mismatches[:10],
            "common_selected_age_updates": common_age,
            "common_selected_age_schedule_identical": not any(
                k[0] < common_age for k in mismatches
            ),
        }
    output = {
        "written_utc": datetime.now(UTC).isoformat(),
        "scope": "Descriptive training-data coverage; no held-out evidence or new fitting.",
        "sampling": "1600 episodes sampled with replacement from 1600 training theorems.",
        "training_data_sha256": sha(data),
        "audit_source_sha256": sha(__file__),
        "runs": results,
        "matched_schedule_comparisons": comparisons,
        "unique_training_theorems_across_three_primary_seeds": len(
            {
                value[0]
                for (arm, _), schedule in schedules.items()
                if arm == "cost"
                for value in schedule.values()
            }
        ),
    }
    path = ALLOC / "training_theorem_coverage.json"
    with path.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(
        json.dumps(
            {
                "runs": {
                    key: {
                        field: value[field]
                        for field in (
                            "episodes",
                            "unique_theorems",
                            "selected_completed_updates",
                            "selected_age_episodes",
                            "selected_age_unique_theorems",
                        )
                    }
                    for key, value in results.items()
                },
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
