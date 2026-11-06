#!/usr/bin/env python3
"""results_10k/*.json -> report_lift.html: per-problem textbook72 solve counts for pretrain and the RL checkpoints.

  python3 passk/export_lift.py pretrain v2_final v5_round025
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tags = sys.argv[1:]
    runs = {t: {r["name"]: r for r in json.load(open(f"{HERE}/results_10k/{t}.json"))["sets"]["textbook72"]["rows"]}
            for t in tags}
    first = runs[tags[0]]
    # body is reference_lines for dev; train has no reference and carries passk.py's 20-line cap instead
    problems = [{"name": name, "split": r["split"], "ref": r["body"] if r["split"] == "dev" else None,
                 **{t: [runs[t][name]["n_ok"], runs[t][name]["n_tried"]] for t in tags}} for name, r in first.items()]
    page = open(f"{HERE}/report_lift.template.html").read().replace("/*DATA*/", json.dumps(problems))
    open(f"{HERE}/report_lift.html", "w").write(page)
    print(len(problems), "problems")


if __name__ == "__main__":
    main()
