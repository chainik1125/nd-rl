"""Register a Runpod pod with the knockout_danbase supervisor (KD/sup/pods.json), for the
coordinator. The supervisor reads pods.json every pass; it writes it only to re-point a pod whose
line has ended to a displaced line waiting for a pod.

    python3 register_pod.py ssh <alias> <ip> <port>       # add the alias to ~/.ssh/config
    python3 register_pod.py add <alias> <pod id> <line> <createdAt> [--rate 0.59]
                            [--gpuh H | --budget-add USD] [--replace]
    python3 register_pod.py retire <alias>                 # after deleting it, or a dead pod
    python3 register_pod.py list

`add` pins <line> to the pod: when the line next needs a GPU, the supervisor resumes it there,
from the newest `KD/runs/<line>/pod_resume/resume_it<i>.pt` on the pod (runpod_copy_line.sh puts
it there), provided its iteration is at least the line's newest checkpoint; the line stays on the
pod; when it ends, the supervisor verifies its backup on serrano and logs one "POD READY TO DELETE:"
event. Only the coordinator can delete a pod: it deletes through the MCP, confirms with a read, then
runs `retire`. Spend counts from createdAt at `rate` $/h until `retire`. `--gpuh H`
adds H x rate x 1.3 to the pod-spend ceiling (16.90 at the start); `--budget-add` adds a sum.
`--replace` retires the line's current pod entry (a pod that died) in the same write.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PODS = C.KD / "sup" / "pods.json"
ALIAS = re.compile(r"^rp-[a-z0-9-]+$")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load() -> dict:
    return json.loads(PODS.read_text()) if PODS.exists() else {}


def save(d: dict) -> None:
    tmp = PODS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    os.replace(tmp, PODS)  # atomic: the supervisor never reads a half-written file


def ceiling(d: dict) -> float:
    return C.POD_BUDGET + sum(float(v.get("budget_add", 0)) for v in d.values())


def cmd_ssh(a: argparse.Namespace) -> None:
    if not ALIAS.match(a.alias):
        raise SystemExit("alias must look like rp-<name>")
    if not re.fullmatch(r"[0-9.]+", a.ip) or not a.port.isdigit():
        raise SystemExit("ip must be dotted digits, port digits")
    cfg = Path.home() / ".ssh" / "config"
    text = cfg.read_text() if cfg.exists() else ""
    if re.search(rf"^Host {re.escape(a.alias)}\s*$", text, re.M):
        raise SystemExit(f"{a.alias} is already in {cfg}; edit it by hand if the address changed")
    block = (f"\nHost {a.alias}\n\tHostName {a.ip}\n\tPort {a.port}\n\tUser root\n"
             "\tIdentityFile ~/.ssh/id_runpod\n\tIdentitiesOnly yes\n"
             "\tStrictHostKeyChecking accept-new\n")  # fmt: skip
    with cfg.open("a") as f:
        f.write(block if text.endswith("\n") or not text else "\n" + block)
    print(f"added {a.alias} -> root@{a.ip}:{a.port} (key ~/.ssh/id_runpod) to {cfg}")


def cmd_add(a: argparse.Namespace) -> None:
    if not ALIAS.match(a.alias):
        raise SystemExit("alias must look like rp-<name>")
    if a.line not in C.LINES:
        raise SystemExit(f"unknown line {a.line}; lines: {sorted(C.LINES)}")
    if not re.fullmatch(r"[a-z0-9]{6,40}", a.pod):
        raise SystemExit("pod id must be lower-case letters and digits")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", a.created):
        raise SystemExit("createdAt must look like 2026-11-05T04:59:57Z")
    d = load()
    if a.alias in d:
        raise SystemExit(f"{a.alias} is already registered: {d[a.alias]}")
    if any(v["pod"] == a.pod for v in d.values()):
        raise SystemExit(f"pod {a.pod} is already registered")
    cur = [k for k, v in d.items() if v["line"] == a.line and not v.get("retired")]
    if cur and not a.replace:
        raise SystemExit(f"{a.line} is already pinned to {cur}; --replace retires that entry")
    for k in cur:
        d[k]["retired"] = now()
    add = a.budget_add if a.budget_add is not None else (a.gpuh or 0.0) * a.rate * 1.3
    d[a.alias] = {"pod": a.pod, "line": a.line, "created": a.created, "rate": a.rate,
                  "budget_add": round(add, 2), "registered": now()}  # fmt: skip
    save(d)
    print(f"registered {a.alias}: {d[a.alias]}" + (f"; retired {cur}" if cur else ""))
    print(f"pod-spend ceiling now ${ceiling(d):.2f} (+${add:.2f})")


def cmd_retire(a: argparse.Namespace) -> None:
    d = load()
    if a.alias not in d:
        raise SystemExit(f"{a.alias} is not registered")
    d[a.alias]["retired"] = now()
    save(d)
    print(f"retired {a.alias}: {d[a.alias]}")


def cmd_list(_: argparse.Namespace) -> None:
    d = load()
    for k, v in d.items():
        print(k, v)
    print(f"ceiling ${ceiling(d):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ssh")
    p.add_argument("alias")
    p.add_argument("ip")
    p.add_argument("port")
    p.set_defaults(f=cmd_ssh)
    p = sp.add_parser("add")
    p.add_argument("alias")
    p.add_argument("pod")
    p.add_argument("line")
    p.add_argument("created")
    p.add_argument("--rate", type=float, default=C.POD_RATE)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpuh", type=float)
    g.add_argument("--budget-add", type=float)
    p.add_argument("--replace", action="store_true")
    p.set_defaults(f=cmd_add)
    p = sp.add_parser("retire")
    p.add_argument("alias")
    p.set_defaults(f=cmd_retire)
    sp.add_parser("list").set_defaults(f=cmd_list)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
