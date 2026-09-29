"""Close this sprint's report after its contracted wall-clock deadline.

This only checks existing receipts/resources and writes reporting metadata. It
does not run training, proof search, model inference or an experiment selector.
The persistent goal status is updated separately through its supported tool.
"""
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.targets.append(value)


def private_browser_processes():
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            parts = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(x.startswith(("--user-data-dir=", "--database=")) and
               "/tmp/flop-report-browser-20260911" in x for x in parts):
            found.append({"pid": int(path.name), "program": parts[0]})
    return found


def main():
    manifest_path = ROOT / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] == "complete":
        print(json.dumps({"status": "already_complete", "completed_utc": manifest.get("completed_utc")}))
        return
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(manifest["start_utc"].replace("Z", "+00:00"))
    deadline = datetime.fromisoformat(manifest["deadline_utc"].replace("Z", "+00:00"))
    if now < deadline:
        raise SystemExit("The contracted sprint is still active; finalization is allowed at or after " + deadline.isoformat())

    assert read_json(ROOT / "reports/final_fact_verification_v1.json")["status"] == "passed"
    assert read_json(ROOT / "reports/run_inventory_v1.json")["status"] == "completed"
    assert manifest["final_lm_endpoint_count"] == 21
    assert manifest["primary_completed_training_arms"] == 8
    assert manifest["online_completed_training_arms"] == 4
    assert all(row["status"] == "completed" for row in manifest["ongoing_jobs"].values())
    assert not manifest["pending_report_commands"]
    assert manifest["new_paid_spending_usd"] == 0
    subprocess.run([sys.executable, str(ROOT / "code/audit_owned_resources.py")], check=True)
    resources = read_json(ROOT / "reports/owned_resources_v1.json")
    assert resources["status"] == "passed" and not resources["owned_research_processes"]
    jobs = resources["gpu_job_completions"]
    assert {x["job"] for x in jobs} == {"pilot_v1", "main_v1", "retention_v1", "online_v1", "final_lm_v1"}
    assert all(x["exit_code"] == 0 for x in jobs)
    browsers = private_browser_processes()
    assert not browsers, browsers

    markdown = [ROOT / "summary.md"] + sorted((ROOT / "reports").glob("*.md"))
    html = [ROOT / "reports/curriculum_walkthrough.html"]
    checks, missing, external = [], [], 0
    for document in markdown + html:
        content = document.read_text()
        if document.suffix == ".html":
            parser = Links()
            parser.feed(content)
            targets = parser.targets
        else:
            targets = re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", content)
        for target in targets:
            target = target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "data:")):
                external += 1
                continue
            if target.startswith("#"):
                continue
            target = re.sub(r":\d+$", "", unquote(target.split("#", 1)[0]))
            path = Path(target)
            path = path if path.is_absolute() else document.parent / path
            row = {"document": str(document.relative_to(ROOT)), "target": target, "exists": path.exists()}
            checks.append(row)
            if not row["exists"]:
                missing.append(row)
    assert not missing, missing
    summary_words = len((ROOT / "summary.md").read_text().split())
    assert summary_words <= 600, summary_words

    closed = datetime.now(timezone.utc)
    elapsed = (closed - start).total_seconds() / 3600
    report_path = ROOT / "reports/sprint_report.md"
    parts = report_path.read_text().split("\n\n", 2)
    assert len(parts) == 3 and parts[1].startswith("The ten-hour sprint runs")
    parts[1] = (
        f"Completed {elapsed:.2f} wall-clock hours, from {start:%Y-%m-%d %H:%M} UTC "
        f"to {closed:%Y-%m-%d %H:%M:%S} UTC. All training, final evaluations and raw-proof audits completed. "
        "Final LM evaluation ran from 02:12 to 02:56 UTC after all 21 endpoints were frozen. "
        "The final hour was reserved for reporting."
    )
    report_path.write_text("\n\n".join(parts))
    review_path = ROOT / "reports/final_document_review.md"
    with review_path.open("a") as stream:
        stream.write(
            f"\nReview closed at {closed.isoformat()}. The closing resource check found zero owned research processes "
            "and zero processes for the private report browser. All five leased GPU jobs have exit-zero completion records. "
            "The link review covers the final Markdown reports and standalone HTML illustration.\n"
        )
    link_receipt = {
        "status": "passed", "checked_at": closed.isoformat(),
        "markdown_documents": len(markdown), "html_documents": len(html),
        "local_link_checks": len(checks), "external_links_not_refetched": external,
        "missing_links": [], "checks": checks,
        "document_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in markdown + html},
        "note": "Closing report-link audit. External sources were inspected during the sprint; this check does not refetch them.",
    }
    write_json(ROOT / "reports/document_link_review_v1.json", link_receipt)
    manifest.update(
        status="complete", completed_utc=closed.isoformat(), updated_utc=closed.isoformat(),
        elapsed_hours=elapsed, remaining_hours=0,
        experiment_status="complete; all declared final measurements, audits and report deliverables finished",
        closing_resource_check_utc=resources["utc"], owned_report_browser_processes=browsers,
    )
    manifest["report_reconciliation"].update(
        status="passed", summary_words=summary_words,
        final_local_link_checks=len(checks), final_missing_links=0,
        final_document_count=len(markdown) + len(html), final_review_closed_utc=closed.isoformat(),
    )
    manifest["deliverables"] = [
        "summary.md", "reports/paper_walkthrough.md", "reports/curriculum_walkthrough.html",
        "reports/sprint_report.md", "reports/method_and_reproduction.md",
        "reports/final_document_review.md", "reports/run_inventory_v1.md",
    ]
    write_json(manifest_path, manifest)
    with (ROOT / "research_log.md").open("a") as stream:
        stream.write(
            f"\n### {closed.isoformat()} — sprint complete\n\n"
            f"Completed{elapsed:.4f}real wall-clock hours. All declared experiments, final evaluations and raw-proof audits "
            "were completed before the final writing block. The final report distinguishes the matched within-PPO curriculum "
            "contrast from the differently budgeted/data-exposed cloning comparison, and preserves both new certificates and "
            "aggregate coverage losses. The closing resource check found0owned research processes and0private report-browser processes; "
            f"all5GPU jobs exited0. Leased local GPU time{manifest['local_gpu_leased_job_wall_hours']:.6f}hours; new paid spending$0. "
            f"All{len(checks)}local report/HTML links resolve; the summary is{summary_words}words. No new training, model inference, "
            "proof-search experiment, cloud job, external message or publication was launched in the final writing block. "
            "A persistent-goal completion update is the remaining tracker operation; its outcome will be recorded separately.\n"
        )
    print(json.dumps({
        "status": "complete", "completed_utc": closed.isoformat(), "elapsed_hours": elapsed,
        "local_link_checks": len(checks), "summary_words": summary_words,
        "owned_research_processes": 0, "owned_report_browser_processes": 0,
        "leased_gpu_hours": manifest["local_gpu_leased_job_wall_hours"], "new_paid_spending_usd": 0,
    }))


if __name__ == "__main__":
    main()
