"""The supervisor: runs on serrano until every line has ended (Leon: "monitor the gpus and respawn
dead processes on new gpus").

    python3 sup.py                  # serrano, the acting supervisor (loop every 90 s)
    python3 sup.py --standby        # temescal: passive; takes over only if serrano is gone
    python3 sup.py --once           # one pass (testing)

State in KD/sup/state.json, events in KD/sup/events.md, the log in KD/sup/sup.log.

**Lease** (`KD/sup/lease.json` {host, utc, epoch} on both machines). The acting supervisor renews
it every pass, locally and on its peer, and copies its state to the peer
(`state.from_<host>.json`). Before every action on a line (launch, respawn, move, stop) it checks
that it still holds the lease: its own host and epoch locally, and no higher epoch on the peer
(a higher epoch there means the peer took over: it yields and exits). The standby on temescal
writes `standby_heartbeat.json` every 90 s and takes over (epoch + 1, the copied state) only when
serrano has not answered ssh for 10 min **and** its lease (serrano's heartbeat) is older than
10 min. A supervisor that starts acting first ends our segments that its state does not know
(`reap_unknown`); so does any pass where a host answers again.

Each pass, for each host: reachable? (ssh, ConnectTimeout 10) and who is on which GPU. For each
line (common.LINES):
- pending: launched (dline.py, in priority order) on a GPU with no process of anyone, ours
  included (serrano: any of its 5; temescal: at most 2 of ours; ynez: when reachable).
- running on host H, segment k: dead when H has not answered for 3 passes, when dline's process
  is gone without an end.json, when its end.json says failed, or when the GRPO log has not grown
  for 25 min (the process group is then killed). Another user's process on its GPU -> the
  segment's STOP_AT_CKPT (dline ends right after the next rolling checkpoint: end.json "moved").
  A dead or moved line is respawned as segment k+1 from its newest checkpoint with a training
  state (rolling or alarm; on H when H answers, else from its backup), on any free GPU; the lost
  iterations (last logged minus resumed) go to log.md's events (KD/sup/events.md).
- the stop rule (PREREG.md) is checked on the merged reads and metrics (merge.py): when met, the
  live segment gets STOP and the line is done.
Backups: after each new rolling checkpoint of a live segment, it is copied one way to
`KD/backup/<line>/` on another machine (the last 2 kept), with the segment's metrics, reads and
Lean dump. Collection: live remote segments are copied one way to serrano's
`KD/collected/<host>/<line>/seg<k>/` for the analysis. Nothing is ever copied into a live segment.
"""

# ruff: noqa: E501, SIM105, SIM102
from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gpu as G  # noqa: E402
import merge as M  # noqa: E402

SUP = C.KD / "sup"
STATE = SUP / "state.json"
DOWN_PASSES = 3
STALL_S = 25 * 60
PASS_S = 90
# Leon's deadline (coordinator, 2026-11-05): a secondary line not at its cap by 15:00Z ends at its
# last read, recorded as "stopped at the deadline"
SECONDARY_DEADLINE = "2026-11-05T15:00:00Z"
# Leon (coordinator, 2026-11-05 05:0xZ; replaces the cap of 4): always leave at least 2 of the
# lab's 12 GPUs free. A GPU is free if nobody has a process on it and it is not reserved; an
# unreadable host counts none of its GPUs as free. Checked before every launch, move or respawn
# (launch time only: running lines are not stopped when other users fill the lab).
MIN_FREE = 2
LAB_GPUS = {"serrano": 5, "ynez": 4, "temescal": 3}
# ... and both apply (Leon's correction, 05:0xZ): never more than 4 GPUs of ours at once, counting
# every GPU with any of our processes (other agents' too), reserved GPUs and our lines' assigned
# GPUs (an unreadable host: its last-seen ours). Launch time only, like MIN_FREE.
GLOBAL_CAP = 4
RSYNC_SSH = " ".join(G.SSH)
# The pods: KD/sup/pods.json {alias: {pod, line, created, rate, budget_add[, retired]}}, written
# by register_pod.py (the coordinator), read every pass, never written here. A pod runs only its
# pinned line; the supervisor never copies anything to a pod (the coordinator's
# runpod_copy_line.sh does, on Leon's approval). A line displaced by another user's process, with
# no lab GPU under the rules, gets one "NEEDS POD:" event and waits for a lab GPU or a pod.
PODS = SUP / "pods.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    with open(SUP / "sup.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def event(msg: str) -> None:
    """Events for log.md and the coordinator (respawns, downed hosts, stops)."""
    log("EVENT " + msg)
    with open(SUP / "events.md", "a") as f:
        f.write(f"- {now()} {msg}\n")


def sh(host: str, cmd: str, timeout: int = 60) -> str | None:
    p = G.run(host, cmd, timeout)
    return None if p is None or p.returncode != 0 else p.stdout


def local(host: str) -> bool:
    return host == G.HOST


LEASE = SUP / "lease.json"
PEER = "temescal" if G.HOST == "serrano" else "serrano"
TAKEOVER_S = 600


def read_json(host: str, path: Path) -> dict | None:
    out = sh(host, f"cat {path} 2>/dev/null; true")
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


def age(d: dict | None) -> float:
    return (
        float("inf")
        if not d
        else time.time()
        - (time.mktime(time.strptime(d["utc"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
    )


def copy(src_host: str, src: str, dst_host: str, dst: str) -> bool:
    """One file, one way, via serrano when neither end is serrano."""

    def rs(a: str, b: str) -> bool:
        cmd = ["rsync", "-rlptD", "-e", RSYNC_SSH, a, b]
        return subprocess.run(cmd, capture_output=True, timeout=1800,
                              env={"SSH_AUTH_SOCK": "", "PATH": "/usr/bin:/bin"}).returncode == 0  # fmt: skip

    def loc(h: str, p: str) -> str:
        return p if local(h) else f"{h}:{p}"

    sh(dst_host, f"mkdir -p {shlex.quote(str(Path(dst).parent))}")
    if src_host == dst_host:  # e.g. a pod's own checkpoint for its next segment
        return sh(dst_host, f"cp {shlex.quote(src)} {shlex.quote(dst)} && echo ok") is not None
    if dst_host.startswith("rp-"):  # nothing goes to a pod but runpod_copy.sh (Leon)
        return (sh(dst_host, f"test -s {shlex.quote(dst)} && echo ok") or "").strip() == "ok"
    if local(src_host) or local(dst_host):
        return rs(loc(src_host, src), loc(dst_host, dst))
    tmp = str(SUP / "relay" / Path(src).name)
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    ok = rs(f"{src_host}:{src}", tmp) and rs(tmp, f"{dst_host}:{dst}")
    Path(tmp).unlink(missing_ok=True)
    return ok


_LINE_RE = "|".join(re.escape(x) for x in C.LINES)
_PS = ("ps -u $USER -o pid=,args= | while read -r pid args; do case \"$args\" in "
       "*dline.py\\ --line*|*code.thinking.grpo*) echo \"$pid|$(readlink /proc/$pid/cwd)|$args\";; "
       "esac; done; true")  # fmt: skip


def procs(host: str) -> list[tuple[int, str, str, int]] | None:
    """This experiment's segment processes on `host`: (pid, 'dline' | 'grpo', line, seg). Only a
    python process whose command line is dline.py with one of our line tags (cwd: this folder),
    or `-m code.thinking.grpo --name <tag> --out KD/runs/<tag>/seg<k>` (cwd: our code tree); ps
    -u $USER lists only our own processes. Shells and every other process never match."""
    out = sh(host, _PS)
    if out is None:
        return None
    res = []
    dl = re.compile(
        rf"^(?:\S*/)?python3?(?:\.\d+)?\s+(?:\S*/)?dline\.py --line ({_LINE_RE}) --seg (\d+)(?: |$)"
    )
    gr = re.compile(rf"^\S*/python3?(?:\.\d+)?\s+-m code\.thinking\.grpo\s.*--name ({_LINE_RE}) "
                    rf"--out {re.escape(str(C.KD))}/runs/({_LINE_RE})/seg(\d+)(?: |$)")  # fmt: skip
    for ln in out.splitlines():
        parts = ln.split("|", 2)
        if len(parts) != 3 or not parts[0].strip().isdigit():
            continue
        pid, cwd, args = int(parts[0]), parts[1], parts[2]
        m = dl.match(args)
        if m and cwd in (str(C.HERE), "/workspace/exp"):  # a pod resolves the symlinked paths
            res.append((pid, "dline", m.group(1), int(m.group(2))))
            continue
        m = gr.match(args)
        if (
            m
            and m.group(1) == m.group(2)
            and cwd in (str(C.CODE), "/workspace/kd/snap/f35d3daf_danbase")
        ):
            res.append((pid, "grpo", m.group(1), int(m.group(3))))
    return res


def kill_seg(host: str, line: str, seg: int, ps: list | None = None) -> list[int]:
    """TERM this experiment's processes of one segment (dline, and GRPO with its process group)."""
    ps = procs(host) if ps is None else ps
    pids = [(pid, kind) for pid, kind, ln, k in ps or [] if ln == line and k == seg]
    if pids:
        sh(host, "; ".join(f"kill -TERM -- -{p}" if kind == "grpo" else f"kill -TERM {p}"
                           for p, kind in pids) + "; true")  # fmt: skip
    return [p for p, _ in pids]


class Sup:
    def __init__(self, state: Path = STATE) -> None:
        SUP.mkdir(parents=True, exist_ok=True)
        self.st = json.loads(state.read_text()) if state.exists() else {}
        for line in C.LINES:
            self.st.setdefault(line, {"status": "pending", "segs": []})
        self.st.setdefault("_hosts", {h: {"fails": 0} for h in C.HOSTS})
        self.pods_cfg: dict[str, dict] = {}
        self.read_pods()
        self.g: dict[str, dict[int, set[str]] | None] = {}
        lease = read_json(G.HOST, LEASE) or {}
        self.epoch = int(lease.get("epoch", 0)) + (0 if lease.get("host") == G.HOST else 1)
        self.yielded = False
        self.standby_warned = False

    def save(self) -> None:
        STATE.write_text(json.dumps(self.st, indent=1))

    # ---------------------------------------------------------------- the pods' registry
    def read_pods(self) -> None:
        """KD/sup/pods.json, every pass; an unreadable or invalid file keeps the last good read."""
        try:
            d = json.loads(PODS.read_text()) if PODS.exists() else {}
            ok = isinstance(d, dict) and all(
                re.fullmatch(r"rp-[a-z0-9-]+", k) and v.get("line") in C.LINES
                and v.get("pod") and v.get("created") for k, v in d.items())  # fmt: skip
        except (json.JSONDecodeError, OSError, AttributeError):
            d, ok = None, False
        if not ok:
            log(
                f"pods.json unreadable or invalid; keeping the last good read {sorted(self.pods_cfg)}"
            )
            return
        for h in d:
            self.st["_hosts"].setdefault(h, {"fails": 0})
        self.pods_cfg = d

    def hosts(self) -> tuple[str, ...]:
        """Polled: the lab, and every registered pod not yet deleted that is not retired or
        still holds a line's live segment."""
        live = {v["segs"][-1]["host"] for k, v in self.st.items() if not k.startswith("_")
                and v.get("segs") and v["status"] == "running"}  # fmt: skip
        gone = {h for h, v in self.st.get("_pods", {}).items() if v.get("deleted")}
        return (*C.HOSTS, *(h for h, v in self.pods_cfg.items()
                            if h not in gone and (not v.get("retired") or h in live)))  # fmt: skip

    def pinned(self) -> dict[str, str]:
        """line -> pod alias (a line that moved to a pod stays there)."""
        return {v["line"]: h for h, v in self.pods_cfg.items() if not v.get("retired")}

    # ---------------------------------------------------------------- the lease
    def renew(self) -> None:
        peer = read_json(PEER, LEASE) if self.up(PEER) else None
        if peer and int(peer.get("epoch", 0)) > self.epoch:
            self.yielded = True
            event(
                f"supervisor on {G.HOST} yields: {peer['host']} holds the lease (epoch "
                f"{peer['epoch']} > {self.epoch}); exiting"
            )
            return
        d = json.dumps({"host": G.HOST, "utc": now(), "epoch": self.epoch})
        LEASE.write_text(d)
        if self.up(PEER):
            sh(PEER, f"mkdir -p {SUP} && echo {shlex.quote(d)} > {LEASE}")
            copy(G.HOST, str(STATE), PEER, str(SUP / f"state.from_{G.HOST}.json"))
            if PODS.exists():  # the standby needs the pods' registry to act on a takeover
                copy(G.HOST, str(PODS), PEER, str(PODS))
            hb = read_json(PEER, SUP / "standby_heartbeat.json")
            if age(hb) > TAKEOVER_S and not self.standby_warned:
                self.standby_warned = True
                event(f"the standby on {PEER} has no heartbeat for over 10 min")
            elif age(hb) <= TAKEOVER_S:
                self.standby_warned = False

    def holds(self) -> bool:
        """Checked before every action on a line."""
        if self.yielded:
            return False
        mine = read_json(G.HOST, LEASE) or {}
        if mine.get("host") != G.HOST or int(mine.get("epoch", -1)) != self.epoch:
            self.yielded = True
            event(f"supervisor on {G.HOST} lost its lease ({mine}); exiting")
            return False
        return True

    def reap_unknown(self, hosts: list[str]) -> None:
        """End our dline segments that this state does not know as a line's live segment."""
        live = {(s["segs"][-1]["host"], k, s["segs"][-1]["seg"]) for k, s in self.st.items()
                if not k.startswith("_") and s["segs"] and s["status"] in ("running", "waiting")
                and not s["segs"][-1].get("dead")}  # fmt: skip
        for h in hosts:
            ps = procs(h)
            for line, k in sorted({(ln, k) for _, _, ln, k in ps or []}):
                if (h, line, k) not in live:
                    pids = kill_seg(h, line, k, ps)
                    event(f"ended an unknown segment: {line} seg{k} on {h} (pids {pids})")

    # ---------------------------------------------------------------- hosts
    def poll_hosts(self) -> None:
        self.read_pods()
        for h in self.hosts():
            self.g[h] = G.gpus(h)
            hs = self.st["_hosts"][h]
            if self.g[h] is not None:
                hs["ours"] = sorted(i for i, u in self.g[h].items() if G.ME in u)
            if self.g[h] is None:
                hs["fails"] += 1
                if hs["fails"] == DOWN_PASSES:
                    event(f"host {h} down ({DOWN_PASSES} passes without an answer)")
            else:
                if hs["fails"] >= DOWN_PASSES:
                    event(f"host {h} answers again")
                    if self.holds():
                        self.reap_unknown([h])
                hs["fails"] = 0

    def up(self, h: str) -> bool:
        return self.g.get(h) is not None

    def down(self, h: str) -> bool:
        return self.st["_hosts"][h]["fails"] >= DOWN_PASSES

    def ours_on(self, h: str) -> list[int]:
        return [s["segs"][-1]["gpu"] for k, s in self.st.items()
                if not k.startswith("_") and s["status"] == "running" and s["segs"][-1]["host"] == h
                and not s["segs"][-1].get("dead")]  # fmt: skip # noqa: E501

    def reserved(self) -> set[tuple[str, int]]:
        """GPUs reserved for another experiment (`KD/sup/reserved.json`, {host: [gpu, ...],
        owner, since}): never launched onto, moved onto or reclaimed, and not counted in a cap."""
        p = SUP / "reserved.json"
        try:
            d = json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:
            d = {"_bad": True}
        if d.get("_bad"):  # unreadable: reserve everything rather than risk a reserved GPU
            return {(h, i) for h in C.HOSTS for i in range(8)}
        return {(h, int(i)) for h in C.HOSTS for i in d.get(h, [])}

    def ours_used(self) -> dict[str, set[int]]:
        """Per host, the GPUs counted against the global cap: any GPU with a process of ours
        (this experiment's or another agent's), every reserved GPU, and our lines' assigned GPUs.
        An unreadable host fails closed: its last-seen ours, its reserved and assigned GPUs."""
        res = self.reserved()
        used: dict[str, set[int]] = {}
        for h in C.HOSTS:
            u = {i for hh, i in res if hh == h}
            u |= {v["segs"][-1]["gpu"] for k, v in self.st.items() if not k.startswith("_")
                  and v["status"] == "running" and v["segs"] and v["segs"][-1]["host"] == h
                  and not v["segs"][-1].get("dead")}  # fmt: skip
            g = self.g.get(h)
            if g is None:
                u |= set(self.st["_hosts"][h].get("ours", []))
            else:
                u |= {i for i, us in g.items() if G.ME in us}
            used[h] = u
        return used

    def n_used(self) -> int:
        return sum(len(v) for v in self.ours_used().values())

    def lab_free(self) -> int:
        res = self.reserved()
        n = 0
        for h in LAB_GPUS:
            g = self.g.get(h)
            if g is not None:  # an unreadable host: none of its GPUs counts as free
                n += sum(1 for i, us in g.items() if not us and (h, i) not in res)
        return n

    def free_gpu(self, line: str | None = None) -> tuple[str, int] | None:
        if line in self.pinned():  # a pod line: its own pod only, outside the lab's rules
            if not (SUP / "pods_enabled").exists():
                return None  # the master switch for every pod (created 05:21Z)
            h = self.pinned()[line]
            g = self.g.get(h)
            if g is None or self.ours_on(h):
                return None
            free = [i for i in sorted(g) if not (g[i] - G.ours(h) - {G.ME}) and not g[i]]
            return (h, free[0]) if free else None
        if self.lab_free() - 1 < MIN_FREE:
            return None  # at least 2 GPUs must stay free after the launch, move or respawn
        if self.n_used() + 1 > GLOBAL_CAP:
            return None  # and ours must stay at most 4 after it
        res = self.reserved()
        for h in ("serrano", "temescal", "ynez"):
            g = self.g.get(h)
            ours = [i for i in self.ours_on(h) if (h, i) not in res]
            if g is None or len(ours) >= C.MAX_OURS[h]:
                continue
            for i in sorted(g):
                if not g[i] and i not in self.ours_on(h) and (h, i) not in res:
                    return h, i
        return None

    def waiting_ahead(self, line: str) -> list[str]:
        """Lines waiting for a lab GPU that come before `line` (primary before secondary); a
        line pinned to a pod never waits for a lab GPU."""
        p = C.LINES[line]["prio"]
        return [k for k, v in self.st.items() if not k.startswith("_") and k != line
                and v["status"] == "waiting" and C.LINES[k]["prio"] < p
                and k not in self.pinned()]  # fmt: skip

    def preempt_candidates(self) -> list[str]:
        """Running secondary lines on a lab GPU (not reserved, host answering, not already
        moving): a waiting primary can take one's GPU. A pod line frees no lab GPU."""
        res = self.reserved()
        return [k for k, v in self.st.items() if not k.startswith("_") and v["status"] == "running"
                and C.LINES[k]["arm"] == "secondary" and v["segs"]
                and not v["segs"][-1].get("moving") and not v["segs"][-1].get("dead")
                and v["segs"][-1]["host"] in C.HOSTS
                and (v["segs"][-1]["host"], v["segs"][-1]["gpu"]) not in res
                and self.up(v["segs"][-1]["host"])]  # fmt: skip

    def preempt_helps(self) -> bool:
        """A preemption gives a waiting primary a GPU only when the lab keeps 2 free GPUs after
        the primary takes the freed one, i.e. at least MIN_FREE are free now (the cap of 4 is
        unchanged by the swap). In a lab with fewer free GPUs it would only idle the secondary."""
        return bool(self.preempt_candidates()) and self.lab_free() >= MIN_FREE

    def remaining_gpuh(self, line: str) -> float | None:
        """GPU-h left to the line's cap at its pace so far (lab GPUs): the smaller of the 15 GPU-h
        cap and the x left to Dan's round-16 x at the GPU-h per x so far."""
        try:
            v = M.line_view(line, self.st[line]["segs"])
        except Exception as e:  # an estimate only; never blocks the event
            log(f"remaining_gpuh {line}: {type(e).__name__}: {e}")
            return None
        if v["x_dan"] <= 0 or v["gpuh"] <= 0:
            return None
        return max(0.0, min(M.GPUH_CAP - v["gpuh"],
                            (v["x_cap"] - v["x_dan"]) * v["gpuh"] / v["x_dan"]))  # fmt: skip

    def needs_pod(self, line: str, seg: dict, src: tuple | None, why: str) -> None:
        """One "NEEDS POD:" event per displacement (never with the plain waiting text, which the
        coordinator's watcher filters out). The resume checkpoint is first put on serrano
        (KD/backup/<line>/seg<k>_<name>) so that runpod_copy_line.sh finds it there."""
        path = None
        if src is not None:
            sh_, it, p = src
            bd = C.KD / "backup" / line
            path = p if (sh_ == "serrano" and p.startswith(str(bd) + "/")) else \
                str(bd / f"seg{seg['seg']}_{Path(p).name}")  # fmt: skip
            if path != p and not Path(path).exists() and not copy(sh_, p, "serrano", path):
                log(f"{line}: NEEDS POD held back: the copy of {p} from {sh_} to serrano failed; "
                    "retrying next pass")  # fmt: skip
                return
        seg["needs_pod"] = now()
        rem = self.remaining_gpuh(line)
        event(
            f"NEEDS POD: {line} ({C.LINES[line]['arm']}): {why}; no lab GPU under the rules. "
            f"Resume checkpoint on serrano: {path} (it {src[1] if src else 0}); remaining about "
            f"{'unknown' if rem is None else f'{rem:.1f}'} GPU-h at its lab pace. It waits for a "
            f"lab GPU or a pod: runpod_copy_line.sh {line} <alias>, podsetup.sh <alias> {line}, "
            f"register_pod.py add <alias> <pod id> {line} <createdAt> --gpuh <remaining>"
        )

    # ---------------------------------------------------------------- segments
    def seg_dir(self, line: str, k: int) -> str:
        return str(C.KD / "runs" / line / f"seg{k}")

    def newest_ckpt(self, line: str, host: str, where: str) -> tuple[int, str] | None:
        """(iteration, path) of the newest checkpoint with a training state under `where`."""
        out = sh(host, f"ls {where}/*_rolling.pt {where}/*_alarm.pt {where}/{line}_it*[0-9].pt "
                       f"2>/dev/null; true")  # fmt: skip
        best = None
        for p in (out or "").split():
            m = re.search(r"_it(\d+)(?:_rolling|_alarm)?\.pt$", p)
            if m and not p.endswith("_snap.pt"):
                it = int(m.group(1))
                if best is None or it > best[0]:
                    best = (it, p)
        return best

    def last_logged(self, line: str, host: str, k: int) -> int | None:
        out = sh(host, f"tail -1 {self.seg_dir(line, k)}/metrics.jsonl 2>/dev/null; true")
        try:
            return int(json.loads(out)["iter"]) if out else None
        except (ValueError, KeyError):
            return None

    def deploy(self, host: str) -> bool:
        """The code, Dan's code, the bases and the pools, one way to a remote host (rsync)."""
        if local(host):
            return True
        if host.startswith("rp-"):  # a pod gets its files from the coordinator's copy only
            return (SUP / "pods_enabled").exists()
        ok = True
        for src in (C.HERE, C.KD / "snap", C.KD / "bases", C.KD / "dan_code_13e41bbd",
                    C.KD / "pools"):  # fmt: skip
            sh(host, f"mkdir -p {src.parent}")
            p = subprocess.run(["rsync", "-rlptD", "--exclude", "__pycache__", "-e", RSYNC_SSH,
                                f"{src}/", f"{host}:{src}/"], capture_output=True, timeout=1800,
                               env={"SSH_AUTH_SOCK": "", "PATH": "/usr/bin:/bin"})  # fmt: skip
            ok &= p.returncode == 0
        return ok

    def launch(self, line: str, host: str, gpu: int, resume: str | None) -> None:
        if not self.holds():
            return
        if not self.deploy(host):
            event(f"{line}: deploy to {host} failed; not launching")
            return
        s = self.st[line]
        k = len(s["segs"]) + 1
        sd = self.seg_dir(line, k)
        res = ""
        if resume:
            res = f" --resume {shlex.quote(resume)}"
        # (an assignment before the subshell is a bash syntax error: export it first)
        envp = "export DANBASE_PY=/workspace/venv/bin/python && " if host.startswith("rp-") else ""
        cmd = (f"mkdir -p {sd} {C.KD}/logs && cd {C.HERE} && {envp}"
               f"(setsid nohup timeout 72h python3 {C.HERE}/dline.py --line {line} --seg {k} --gpu {gpu}{res} "
               f"> {C.KD}/logs/{line}_seg{k}.dline.log 2>&1 < /dev/null &) && echo started")  # fmt: skip
        out = sh(host, cmd)
        s["segs"].append({"seg": k, "host": host, "gpu": gpu, "resume": resume, "start": now(),
                          "ok": bool(out)})  # fmt: skip
        s["status"] = "running"
        log(f"launch {line} seg{k} on {host} GPU {gpu} resume {resume}: {out!r}")

    def respawn(self, line: str, why: str) -> None:
        if not self.holds():
            return
        why = why.split("; behind ")[0]  # a waiting line's reason does not grow every pass
        s = self.st[line]
        seg = s["segs"][-1]
        seg["dead"] = seg.get("dead") or why  # its GPU no longer counts as ours
        h, k = seg["host"], seg["seg"]
        if self.up(h):  # an orphaned GRPO of the dead segment (dline gone) must not run on
            kill_seg(h, line, k)
        src = None
        if self.up(h):
            ck = self.newest_ckpt(line, h, f"{self.seg_dir(line, k)}/ckpts")
            for j in range(k - 1, 0, -1):  # an earlier segment's, if this one wrote none yet
                if ck is not None:
                    break
                prev = s["segs"][j - 1]
                if prev["host"] == h:
                    ck = self.newest_ckpt(line, h, f"{self.seg_dir(line, j)}/ckpts")
            if ck is not None:
                src = (h, *ck)
            last = self.last_logged(line, h, k)
        else:
            last = None
        if src is None:  # the backup
            for b in C.HOSTS:
                if b != h and self.up(b):
                    ck = self.newest_ckpt(line, b, f"{C.KD}/backup/{line}")
                    if ck is not None and (src is None or ck[0] > src[1]):
                        src = (b, *ck)
        pod = self.pinned().get(line)
        ahead = [] if pod else self.waiting_ahead(line)  # a pod line waits for no lab line
        tgt = None if ahead else self.free_gpu(line)
        if tgt is None:
            if ahead:
                why = f"{why}; behind {ahead}"
            s["status"], s["wait_resume"] = "waiting", {"why": why, "src": src, "last": last}
            displaced = seg.get("displaced") and not pod
            pod_down = pod is not None and self.down(pod)
            if (displaced or pod_down) and not seg.get("needs_pod") and not (
                    C.LINES[line]["arm"] == "primary" and self.preempt_helps()):  # fmt: skip
                # (a primary that can preempt a lab secondary gets its GPU that way instead)
                self.needs_pod(line, seg, src, seg["displaced"] if displaced else
                               f"its pod {pod} is down (a new pod: register_pod.py add ... "
                               f"--replace)")  # fmt: skip
                seg.setdefault("waiting_since", now())
                return
            msg = f"{line}: {why}; no free GPU, waiting (resume point {src})"
            if seg.get("waiting_since"):
                log(msg)  # reported once, as an event, when the wait began
            else:
                seg["waiting_since"] = now()
                event(msg)
            return
        th, tg = tgt
        resume = None
        it0 = 0
        if th.startswith("rp-") and src is not None and src[0] != th:
            # the pod has the line's checkpoint only from the coordinator's runpod_copy_line.sh
            pr = self.pod_resume(line, th)
            if pr is None or pr[0] < src[1]:
                s["status"], s["wait_resume"] = "waiting", {"why": why, "src": src, "last": last}
                msg = (f"{line}: pinned to {th}; waiting for its resume checkpoint there "
                       f"({C.KD}/runs/{line}/pod_resume/resume_it<i>.pt with i ≥ {src[1]}; "
                       f"found {pr}): runpod_copy_line.sh {line} {th}")  # fmt: skip
                if seg.get("pod_wait") == th:
                    log(msg)
                else:
                    seg["pod_wait"] = th
                    event(msg)
                return
            it0, resume = pr
        elif src is not None:
            sh_, it0, path = src
            dst = f"{self.seg_dir(line, k + 1)}/resume_it{it0}.pt"
            if not copy(sh_, path, th, dst):  # to a pod: only its own checkpoint, on the pod
                event(f"{line}: copy of {path} from {sh_} to {th} failed; retrying next pass")
                return
            resume = dst
        lost = None if last is None else last - it0
        if resume and "/pod_resume/" in resume:
            origin = "the coordinator's copy on the pod"
        elif src and src[0] != h:
            origin = f"backup on {src[0]}"
        else:
            origin = "its own host"
        self.launch(line, th, tg, resume)
        event(f"{line}: {why} (seg{k} on {h}); respawned as seg{k + 1} on {th} GPU {tg} from "
              f"it {it0} ({origin}); "
              f"lost iterations: {lost if lost is not None else 'unknown (host down)'}")  # fmt: skip
        if seg.get("needs_pod") and not th.startswith("rp-"):
            event(f"POD NOT NEEDED: {line} resumed on {th} GPU {tg} (its NEEDS POD of "
                  f"{seg['needs_pod']} is void)")  # fmt: skip

    def pod_resume(self, line: str, pod: str) -> tuple[int, str] | None:
        """(iteration, path) of the newest non-empty resume_it<i>.pt in the line's pod_resume/
        folder on its pod (runpod_copy_line.sh writes it there; rsync renames it into place
        only when complete)."""
        d = f"{C.KD}/runs/{line}/pod_resume"
        out = sh(pod, f'for f in {d}/resume_it*.pt; do test -s "$f" && echo "$f"; done; true')
        best = None
        for p in (out or "").split():
            m = re.search(r"/resume_it(\d+)\.pt$", p)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), p)
        return best

    def check_running(self, line: str) -> None:
        s = self.st[line]
        seg = s["segs"][-1]
        h, k, sd = seg["host"], seg["seg"], self.seg_dir(line, seg["seg"])
        if self.down(h):
            self.respawn(line, f"host {h} down")
            return
        if not self.up(h):
            return  # not yet 3 passes
        out = sh(h, f"cat {sd}/end.json 2>/dev/null; echo ===; cat {sd}/status.json 2>/dev/null; "
                    f"echo ===; stat -c %Y {C.KD}/logs/{line}_seg{k}.log 2>/dev/null; echo ===; "
                    f"test -f {sd}/STOP_AT_CKPT && stat -c %Y {sd}/STOP_AT_CKPT; true")  # fmt: skip # noqa: E501
        ps = procs(h)
        if out is None or ps is None:
            return
        end, status, mtime, sac = (x.strip() for x in (out.split("===") + ["", "", ""])[:4])
        if sac and not seg.get("moving"):  # the segment's own GPU check asked for a move
            seg["moving"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(sac)))
            seg["displaced"] = (f"displaced by another user's process on {h} GPU {seg['gpu']} "
                                "(the segment's own check)")  # fmt: skip
        alive = (
            "alive"
            if any(kind == "dline" and ln == line and kk == k for _, kind, ln, kk in ps)
            else ""
        )
        if end:
            how = json.loads(end)["how"]
            seg["end"] = json.loads(end)["utc"]
            self.collect(line)  # the segment's last files (end.json, last reads) to serrano
            if how in ("done", "stopped"):
                s["status"] = "done"
                event(f"{line}: ended ({how}) on {h} seg{k}")
            else:  # moved / failed
                self.respawn(line, f"segment ended: {how}")
            return
        if alive != "alive":
            self.respawn(line, "process gone (killed or OOM)")
            return
        if mtime and time.time() - int(mtime) > STALL_S:
            kill_seg(h, line, k, ps)
            self.respawn(line, "no progress for 25 min")
            return
        if (h, seg["gpu"]) in self.reserved() and not seg.get("moving") and self.holds():
            sh(h, f"touch {sd}/STOP_AT_CKPT")
            seg["moving"] = now()
            event(f"{line}: its GPU ({h} GPU {seg['gpu']}) is reserved for another experiment; "
                  "availability stop at the next checkpoint")  # fmt: skip
            return
        pf = seg.get("preempted_for")
        if pf and self.st.get(pf, {}).get("status") != "waiting" and self.holds():
            # the primary it made room for no longer waits: the preemption is cancelled
            sh(h, f"rm -f {sd}/STOP_AT_CKPT")
            seg.pop("moving", None)
            seg.pop("preempted_for")
            event(f"{line}: preemption cancelled ({pf} no longer waits)")
            return
        g0 = (self.g.get(h) or {}).get(seg["gpu"], set())
        if (seg.get("moving") and not seg.get("preempted_for") and not seg.get("cap_stop")
                and (h, seg["gpu"]) not in self.reserved() and not (g0 - G.ours(h))):  # fmt: skip
            # the other user's process has gone: a move that is no longer needed is cancelled
            sh(h, f"rm -f {sd}/STOP_AT_CKPT")
            event(f"{line}: move cancelled; {h} GPU {seg['gpu']} has no other user's process now")
            seg.pop("moving")
            seg.pop("displaced", None)
            return
        if seg.get("moving") and time.time() - M._utc(seg["moving"]) > 8 * 60:
            # a segment started before 00:3xZ waits for a *rolling* checkpoint only; an alarmed
            # iteration writes an alarm checkpoint instead: end it once a newer one exists
            ck = self.newest_ckpt(line, h, f"{sd}/ckpts")
            t_ck = sh(h, f"stat -c %Y {ck[1]} 2>/dev/null; true") if ck else None
            if t_ck and t_ck.strip() and int(t_ck.strip()) > M._utc(seg["moving"]) and self.holds():
                kill_seg(h, line, k, ps)
                self.respawn(line, f"moved at its it {ck[0]} checkpoint")
                return
        g = self.g.get(h) or {}
        if g.get(seg["gpu"], set()) - G.ours(h):
            if not seg.get("moving") and self.holds():
                sh(h, f"touch {sd}/STOP_AT_CKPT")
                seg["moving"] = now()
                users = sorted(g[seg["gpu"]] - G.ours(h))
                seg["displaced"] = (f"displaced by another user's process on {h} GPU "
                                    f"{seg['gpu']} ({users})")  # fmt: skip
                event(f"{line}: another user's process on {h} GPU {seg['gpu']} "
                      f"({users}); moving at the next checkpoint")  # fmt: skip

    # ---------------------------------------------------------------- copies
    def backup(self, line: str) -> None:
        seg = self.st[line]["segs"][-1]
        h, k = seg["host"], seg["seg"]
        if not self.up(h):
            return
        b = "serrano" if h != "serrano" else ("temescal" if self.up("temescal") else "ynez")
        if not self.up(b):
            return
        ck = self.newest_ckpt(line, h, f"{self.seg_dir(line, k)}/ckpts")
        bd = f"{C.KD}/backup/{line}"
        if ck is not None:
            name = f"seg{k}_{Path(ck[1]).name}"
            have = sh(b, f"ls {bd}/{name} 2>/dev/null; true")
            if not have and copy(h, ck[1], b, f"{bd}/{name}"):
                sh(b, f"cd {bd} && ls -t *_rolling.pt *_alarm.pt 2>/dev/null | tail -n +3 | "
                      f"xargs -r rm -f")  # fmt: skip
                log(f"backup {line}: {name} -> {b}")
        for f in ("metrics.jsonl", "lean_dump.jsonl", "reads/"):
            copy(h, f"{self.seg_dir(line, k)}/{f}", b, f"{bd}/seg{k}/{f}")

    def collect(self, line: str) -> None:
        for seg in self.st[line]["segs"]:
            h, k = seg["host"], seg["seg"]
            if local(h) or not self.up(h):
                continue
            src = f"{h}:{self.seg_dir(line, k)}/"
            dst = C.KD / "collected" / h / line / f"seg{k}"
            dst.mkdir(parents=True, exist_ok=True)
            env = {"SSH_AUTH_SOCK": "", "PATH": "/usr/bin:/bin"}
            subprocess.run(["rsync", "-rlptD", "--exclude", "ckpts/", "--exclude", "*.pt", "-e",
                            RSYNC_SSH, src, str(dst)], capture_output=True, timeout=1800, env=env)  # fmt: skip
            (C.KD / "collected" / h / "logs").mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-rlptD", "-e", RSYNC_SSH, f"{h}:{C.KD}/logs/{line}_seg{k}.log",
                            str(C.KD / "collected" / h / "logs") + "/"], capture_output=True,
                           timeout=1800, env=env)  # fmt: skip

    # ---------------------------------------------------------------- the loop
    def stop_rule(self, line: str) -> None:
        r = M.line_view(line, self.st[line]["segs"])
        c = r["cross_dan"]
        if c is not None and not self.st[line].get("crossed"):
            self.st[line]["crossed"] = c
            event(
                f"{line}: FIRST CROSSING of Dan's bar: round {c['round']} (it {c['it']}), "
                f"x_dan {c['x_dan']:.3f}, x_ours {c['x_ours']:.4f}, cumulative {c['cum']}"
            )
        why = M.stop_reason(line, r)
        if why and self.holds():
            seg = self.st[line]["segs"][-1]
            sh(seg["host"], f"touch {self.seg_dir(line, seg['seg'])}/STOP")
            self.st[line]["stop_reason"] = why
            event(f"{line}: stop rule met ({why}); STOP written")

    def one_pass(self) -> None:
        self.poll_hosts()
        self.renew()
        if self.yielded:
            return
        late = time.time() >= M._utc(SECONDARY_DEADLINE)
        for line in sorted(C.LINES, key=lambda x: C.LINES[x]["prio"]):
            s = self.st[line]
            try:
                if (late and C.LINES[line]["arm"] == "secondary" and s["status"] != "done"
                        and line not in self.pinned()):  # pod lines run to their cap (Leon)  # fmt: skip
                    if s["status"] == "running" and s["segs"] and self.holds():
                        seg = s["segs"][-1]
                        sh(seg["host"], f"touch {self.seg_dir(line, seg['seg'])}/STOP")
                        seg["end"] = now()
                        self.collect(line)
                    s["status"], s["stop_reason"] = "done", "stopped at the deadline (15:00Z)"
                    event(f"{line}: stopped at the deadline (15:00Z)")
                    self.save()
                    continue
                if s["status"] == "running":
                    self.check_running(line)
                    if s["status"] == "running":
                        self.backup(line)
                        self.collect(line)
                        if "stop_reason" not in s:
                            self.stop_rule(line)
                elif s["status"] == "waiting":
                    w = s.pop("wait_resume")
                    s["status"] = "running"
                    self.respawn(line, w["why"])
                elif s["status"] == "pending":
                    tgt = (self.free_gpu(line) if line in self.pinned()
                           else None if self.waiting_ahead(line) else self.free_gpu(line))  # fmt: skip
                    if tgt is not None:
                        self.launch(line, *tgt, None)
                        event(f"{line}: launched on {tgt[0]} GPU {tgt[1]}")
            except Exception as e:  # one line's trouble must not stop the others
                log(f"ERROR {line}: {type(e).__name__}: {e}")
            self.save()
        try:
            self.preempt()
        except Exception as e:
            log(f"ERROR preempt / cap: {type(e).__name__}: {e}")
        try:
            self.pods()
        except Exception as e:
            log(f"ERROR pods: {type(e).__name__}: {e}")
        self.save()
        live = [k for k, v in self.st.items() if not k.startswith("_") and v["status"] != "done"]
        log(
            f"pass: hosts {[h for h in self.hosts() if self.up(h)]}; not done: {live}; "
            f"pod spend ${self.pod_spend():.2f}"
        )

    # ---------------------------------------------------------------- the pods
    def rest(self, method: str, pod: str) -> tuple[int, dict]:
        """Runpod REST with the key read from ~/.runpod_env and passed to curl on stdin only."""
        sh_ = ('set -a; . "$HOME/.runpod_env"; set +a; printf \'header = "Authorization: Bearer %s"\\n\' '
               f'"$RUNPOD_API_KEY" | curl -sS --max-time 60 -o /dev/stdout -w "\\n%{{http_code}}" '
               f"-X {method} -K - https://rest.runpod.io/v1/pods/{pod}")  # fmt: skip
        p = subprocess.run(["bash", "-c", sh_], capture_output=True, text=True, timeout=120)
        body, _, code = p.stdout.rpartition("\n")
        try:
            d = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            d = {"_raw_len": len(body)}
        return (int(code) if code.strip().isdigit() else -1), d

    def pod_spend(self) -> float:
        """Every registered pod, from its createdAt at its rate until its deletion here (or its
        retirement in pods.json: the coordinator then owns its deletion)."""
        t = time.time()
        tot = 0.0
        for h, v in self.pods_cfg.items():
            end = self.st.get("_pods", {}).get(h, {}).get("deleted") or v.get("retired")
            t1 = M._utc(end) if end else t
            tot += max(0.0, t1 - M._utc(v["created"])) / 3600 * float(v.get("rate", C.POD_RATE))
        return tot

    def ceiling(self) -> float:
        return C.POD_BUDGET + sum(float(v.get("budget_add", 0)) for v in self.pods_cfg.values())

    def pods(self) -> None:
        pd = self.st.setdefault("_pods", {})
        spend, ceil = self.pod_spend(), self.ceiling()
        if spend >= ceil and float(pd.get("budget_warned_at", 0)) < ceil:
            pd["budget_warned"], pd["budget_warned_at"] = now(), ceil
            event(f"pod spend ${spend:.2f} has reached the ceiling ${ceil:.2f}; "
                  "nothing is stopped (the coordinator asks Leon)")  # fmt: skip
        for h, v in self.pods_cfg.items():
            line, ps_ = v["line"], pd.setdefault(h, {})
            if ps_.get("deleted") or v.get("retired"):
                continue
            seg = self.st[line]["segs"][-1] if self.st[line]["segs"] else None
            if (self.st[line]["status"] == "running" and seg and seg["host"] != h
                    and not seg.get("dead") and ps_.get("idle_warned") != seg["seg"]):  # fmt: skip
                ps_["idle_warned"] = seg["seg"]
                event(f"pod {h} ({v['pod']}) is registered for {line}, which runs on "
                      f"{seg['host']} GPU {seg['gpu']} (seg{seg['seg']}); the pod idles until the "
                      "line's next move (delete it, or keep it for the line)")  # fmt: skip
            if self.st[line]["status"] != "done" or not self.holds():
                continue
            # the line has ended: final copies to serrano, verified, then the pod is deleted
            if seg is None or seg["host"] != h:
                if not ps_.get("verify_warned"):
                    ps_["verify_warned"] = now()
                    event(f"{line} ended on {seg and seg['host']}, not on its pod {h} "
                          f"({v['pod']}); the pod is not deleted here (the coordinator decides)")  # fmt: skip
                continue
            if self.up(h):
                self.backup(line)
                self.collect(line)
            bd = C.KD / "backup" / line
            ck = sorted(bd.glob("*_rolling.pt")) + sorted(bd.glob("*_alarm.pt"))
            ok = (bool(ck) and (bd / f"seg{seg['seg']}" / "metrics.jsonl").exists()
                  and any((bd / f"seg{seg['seg']}" / "reads").glob("it*.json")))  # fmt: skip
            if not ok:
                if not ps_.get("verify_warned"):
                    ps_["verify_warned"] = now()
                    event(f"{line} ended but its backup on serrano is incomplete; {h} is kept")
                continue
            code, _ = self.rest("DELETE", v["pod"])
            code2, d2 = self.rest("GET", v["pod"])
            gone = code2 == 404 or d2.get("desiredStatus") in ("TERMINATED",)
            if code in (200, 202, 204) and gone:
                ps_["deleted"] = now()
                event(f"pod {v['pod']} ({h}, {line}) deleted after its line ended and its backup "
                      f"was verified (DELETE {code}, GET {code2}); pod spend now "
                      f"${self.pod_spend():.2f}")  # fmt: skip
            elif not ps_.get("delete_warned"):
                ps_["delete_warned"] = now()
                event(f"pod {v['pod']} ({h}): delete not confirmed (DELETE {code}, GET {code2}); "
                      "retrying every pass")  # fmt: skip

    def preempt(self) -> None:
        """A primary line waits and no GPU is free: the lowest-priority running secondary line
        (not on a reserved GPU) ends at its next checkpoint and waits behind it (coordinator,
        2026-11-05)."""
        waiting = [k for k, v in self.st.items() if not k.startswith("_")
                   and v["status"] == "waiting" and C.LINES[k]["arm"] == "primary"
                   and k not in self.pinned()]  # fmt: skip
        if not waiting or self.free_gpu() is not None or not self.preempt_helps():
            return
        if any(v["segs"] and v["segs"][-1].get("preempted_for") for k, v in self.st.items()
               if not k.startswith("_") and v["status"] == "running"):  # fmt: skip
            return  # one preemption in flight
        cand = self.preempt_candidates()  # lab GPUs only: a pod line frees none
        if not cand or not self.holds():
            return
        k = max(cand, key=lambda x: C.LINES[x]["prio"])
        seg = self.st[k]["segs"][-1]
        sh(seg["host"], f"touch {self.seg_dir(k, seg['seg'])}/STOP_AT_CKPT")
        seg["moving"], seg["preempted_for"] = now(), waiting[0]
        pd = (
            self.st[waiting[0]]["segs"][-1].get("displaced")
            if self.st[waiting[0]]["segs"]
            else None
        )
        if pd:  # the chain started with another user's process: this line may need a pod
            seg["displaced"] = f"preempted for {waiting[0]}, which was {pd}"
        event(
            f"{k}: preempted for the primary {waiting[0]} (no free GPU); ends at its next "
            f"checkpoint on {seg['host']} GPU {seg['gpu']} and waits behind the primary lines"
        )

    def loop(self) -> None:
        self.poll_hosts()
        self.renew()
        if not self.yielded and self.holds():
            self.reap_unknown([h for h in self.hosts() if self.up(h)])
        while any(v["status"] != "done" for k, v in self.st.items() if not k.startswith("_")):
            if self.yielded:
                return
            t = time.time()
            self.one_pass()
            time.sleep(max(5.0, PASS_S - (time.time() - t)))
        event("every line has ended; supervisor exits")


def standby() -> None:
    """Passive on temescal: a heartbeat every 90 s; takes over only if serrano is gone."""
    since = None
    while True:
        (SUP / "standby_heartbeat.json").write_text(json.dumps({"utc": now(), "host": G.HOST}))
        ok = sh("serrano", "echo ok", timeout=30)
        since = None if ok else (since or time.time())
        lease = read_json(G.HOST, LEASE)
        if lease and lease.get("host") == G.HOST:
            return  # this host already holds the lease (an earlier takeover): act
        if since is not None and time.time() - since >= TAKEOVER_S and age(lease) >= TAKEOVER_S:
            src = SUP / "state.from_serrano.json"
            if src.exists():
                STATE.write_text(src.read_text())
            event(
                f"standby on {G.HOST} takes over: serrano unreachable for "
                f"{(time.time() - since) / 60:.0f} min, its lease {age(lease) / 60:.0f} min old"
            )
            return
        time.sleep(PASS_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--standby", action="store_true")
    a = ap.parse_args()
    SUP.mkdir(parents=True, exist_ok=True)
    lock = open(SUP / "sup.lock", "w")  # noqa: SIM115 (held for the process's life)
    try:  # one supervisor per machine: a second one exits at once
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another sup.py holds KD/sup/sup.lock on this machine") from None
    if a.standby:
        standby()  # returns when this host is to act
    s = Sup()
    if a.once:
        s.one_pass()
    else:
        s.loop()


if __name__ == "__main__":
    main()
