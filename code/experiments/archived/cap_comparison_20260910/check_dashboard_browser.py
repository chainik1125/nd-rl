"""Verify the stock dashboard after all six runs have been imported and audited."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7862/?project=cap-comparison-v2&smoothing=0&metric_filter=%5Eaccuracy%2F&x_axis=compute%2Frl_budget_fraction")
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "reports/dashboard_accuracy_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["complete"] and audit["records_checked"] == audit["native_database_points_checked"] == 72
    assert audit["analysis_results_sha256"] == sha(root / "reports/analysis_results.json")
    labels = sorted(record["label"] for record in audit["labels"].values())
    assert len(labels) == len(set(labels)) == 6
    screenshot = root / "reports/dashboard.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            # Live polling can prevent networkidle; wait for actual UI controls.
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
            expect(page.get_by_text("Runs (6)", exact=True)).to_be_visible(timeout=30000)
            for label in labels:
                checkbox = page.get_by_role("checkbox", name=label, exact=True)
                expect(checkbox).to_be_visible()
                expect(checkbox).to_be_checked()
            for metric in ("pretraining", "validation_synthetic", "validation_textbook"):
                expect(page.get_by_text(metric, exact=True).first).to_be_visible()
            expect(page.locator('input[aria-label="Group by"]')).to_have_value("")
            expect(page.locator('input[type="range"]')).to_have_value("0")
            expect(page.locator('input[aria-label="X-axis"]')).to_have_value("compute/rl_budget_fraction")
            query = parse_qs(urlsplit(page.url).query)
            assert query["smoothing"] == ["0"]
            assert query["metric_filter"] == ["^accuracy/"]
            assert query["x_axis"] == ["compute/rl_budget_fraction"]
            assert query["project"] == ["cap-comparison-v2"]
            page.screenshot(path=str(screenshot), full_page=True)
            assert not errors, errors
            url = page.url
        finally:
            browser.close()
    receipt = {"checked_utc": datetime.now(timezone.utc).isoformat(), "url": url,
               "source_sha256": sha(Path(__file__)), "accuracy_audit_sha256": sha(audit_path),
               "screenshot_sha256": sha(screenshot), "labels": labels, "smoothing": 0,
               "group_by": None, "x_axis": "compute/rl_budget_fraction", "browser_errors": errors,
               "scope": "Six selected, separately labeled runs; three accuracy plots; raw curves; compute axis. Native values are checked by the linked accuracy audit."}
    (root / "reports/dashboard_browser_check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
