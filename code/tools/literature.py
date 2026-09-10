# /// script
# requires-python = ">=3.11"
# dependencies = ["bibtexparser==1.4.4"]
# ///
"""Add one paper to local Zotero and read its automatically exported citation.

Run from anywhere: uv run --script code/tools/literature.py --help
Zotero and Better BibTeX must be running. No API token is needed.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import bibtexparser
from bibtexparser.customization import convert_to_unicode

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code" / ".zotero.local.json"
BIB = ROOT / "references.bib"
BASE = "http://127.0.0.1:23119"


def post(endpoint, payload, content_type="application/json"):
    data = json.dumps(payload).encode() if content_type == "application/json" else payload.encode()
    request = urllib.request.Request(
        BASE + endpoint, data=data, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode()
    return json.loads(body) if body else None


def rpc(method, *params):
    result = post(
        "/better-bibtex/json-rpc",
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)},
    )
    if "error" in result:
        raise ValueError(result["error"]["message"])
    return result["result"]


def fetch_bib(source):
    if Path(source).is_file():
        return Path(source).read_text()
    arxiv = re.fullmatch(
        r"(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
        source,
        re.I,
    )
    if arxiv:
        url = "https://arxiv.org/bibtex/" + arxiv[1]
    else:
        doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", source, flags=re.I)
        if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
            raise ValueError("Supply an arXiv ID/URL, DOI/URL, or a one-paper .bib file.")
        url = "https://doi.org/" + doi
    # curl uses the operating system's certificate store on macOS.
    return subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "40",
            "--header",
            "Accept: application/x-bibtex",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def parse_one(text):
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    entries = bibtexparser.loads(text, parser=parser).entries
    if len(entries) != 1 or not entries[0].get("title"):
        raise ValueError("Expected exactly one BibTeX entry with a title; nothing imported.")
    return entries[0]


def matches(entry, config):
    alternatives = [["title", "is", entry["title"]]]
    if entry.get("doi"):
        doi = re.sub(r"^https?://doi.org/", "", entry["doi"], flags=re.I)
        alternatives.append(["DOI", "is", doi])
    if entry.get("eprint"):
        identifier = re.sub(r"v\d+$", "", entry["eprint"])
        alternatives.extend([["extra", "contains", identifier], ["url", "contains", identifier]])
    return rpc(
        "item.search",
        [
            ["libraryID", "is", config["libraryID"]],
            ["itemType", "isNot", "attachment"],
            ["itemType", "isNot", "note"],
            ["blockStart"],
            *alternatives,
            ["blockEnd"],
        ],
    )


def wait_for_export(key):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if BIB.exists():
            entries = bibtexparser.loads(BIB.read_text()).entries
            if any(entry["ID"] == key for entry in entries):
                print(f"[@{key}]  —  {BIB.relative_to(ROOT)}")
                return
        time.sleep(1)
    raise ValueError(
        f"Paper is in Zotero as {key}, but its auto-export has not appeared in references.bib. "
        "Check Better BibTeX's Automatic Export settings; do not import a second copy."
    )


def wait_for_record(entry, config):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        found = matches(entry, config)
        if len(found) > 1:
            raise ValueError(
                "Multiple matching records after import. Inspect Zotero before retrying."
            )
        if found and found[0].get("citekey"):
            return found[0]["citekey"]
        time.sleep(1)
    raise ValueError(
        "Import returned, but no citation key is ready. Inspect Zotero before retrying."
    )


def add(source, config):
    text = fetch_bib(source)
    entry = parse_one(text)
    existing = matches(entry, config)
    if len(existing) > 1:
        raise ValueError("Multiple matching Zotero records; merge duplicates in Zotero first.")
    if existing:
        key = existing[0]["citekey"]
        collections = rpc("item.collections", [key]).get(key, [])
        if not any(c["key"] == config["key"] for c in collections):
            raise ValueError(
                f"Already in Zotero as {key}. Add that existing item to the project collection; "
                "a duplicate was not created."
            )
        print("Already present; no duplicate imported.")
    else:
        target = post("/connector/getSelectedCollection", {})
        if (target.get("libraryID"), target.get("id")) != (config["libraryID"], config["id"]):
            raise ValueError(
                "Select the natural-deduction-rl project collection in Zotero before importing. "
                "The local import API uses the selected collection. Nothing was imported."
            )
        # One import request, never automatically retry a potentially successful write.
        session = uuid.uuid4().hex
        try:
            post("/connector/import?session=" + session, text, "text/plain")
        except (TimeoutError, urllib.error.URLError):
            # Zotero can save and export before its import response completes.
            # Reconcile with the library instead of repeating the write.
            print("Import response delayed; checking whether Zotero saved the paper.")
        key = wait_for_record(entry, config)
        collections = rpc("item.collections", [key]).get(key, [])
        if not any(c["key"] == config["key"] for c in collections):
            raise ValueError(
                "Paper saved outside the project collection. Inspect Zotero before retrying."
            )
        print(f"Added: {entry['title']}")
    wait_for_export(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Register this checkout's Better BibTeX auto-export")
    setup.add_argument(
        "--collection",
        default="/1/natural-deduction-rl",
        help="BBT collection path; first component is the local library ID/name",
    )
    commands.add_parser("status", help="Show connection and local export configuration")
    add_command = commands.add_parser("add", help="Import one paper; skip existing project records")
    add_command.add_argument("source", help="arXiv ID/URL, DOI/URL, or one-paper BibTeX file")
    args = parser.parse_args()
    ready = rpc("api.ready")
    if args.command == "setup":
        config = rpc(
            "autoexport.add",
            args.collection,
            "Better BibLaTeX",
            str(BIB),
            {"exportNotes": False},
            False,
        )
        config["collection"] = args.collection
        CONFIG.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Automatic export registered: {args.collection} -> references.bib")
        return
    if not CONFIG.exists():
        raise ValueError("Run the setup command for this checkout first.")
    config = json.loads(CONFIG.read_text())
    if args.command == "status":
        print(json.dumps({**ready, **config, "bibliography_exists": BIB.exists()}, indent=2))
    else:
        add(args.source, config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, urllib.error.URLError, subprocess.CalledProcessError) as error:
        print(f"literature: {error}", file=sys.stderr)
        sys.exit(1)
