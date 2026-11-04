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
RSYNC_SSH = " ".join(G.SSH)


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
        cmd = ["rsync", "-a", "-e", RSYNC_SSH, a, b]
        return subprocess.run(cmd, capture_output=True, timeout=1800,
                              env={"SSH_AUTH_SOCK": "", "PATH": "/usr/bin:/bin"}).returncode == 0  # fmt: skip

    def loc(h: str, p: str) -> str:
        return p if local(h) else f"{h}:{p}"

    sh(dst_host, f"mkdir -p {shlex.quote(str(Path(dst).parent))}")
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
        if m and cwd == str(C.HERE):
            res.append((pid, "dline", m.group(1), int(m.group(2))))
            continue
        m = gr.match(args)
        if m and m.group(1) == m.group(2) and cwd == str(C.CODE):
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
        self.g: dict[str, dict[int, set[str]] | None] = {}
        lease = read_json(G.HOST, LEASE) or {}
        self.epoch = int(lease.get("epoch", 0)) + (0 if lease.get("host") == G.HOST else 1)
        self.yielded = False
        self.standby_warned = False

    def save(self) -> None:
        STATE.write_text(json.dumps(self.st, indent=1))

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
        for h in C.HOSTS:
            self.g[h] = G.gpus(h)
            hs = self.st["_hosts"][h]
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

    def free_gpu(self) -> tuple[str, int] | None:
        for h in ("serrano", "temescal", "ynez"):
            g = self.g.get(h)
            if g is None or len(self.ours_on(h)) >= C.MAX_OURS[h]:
                continue
            for i in sorted(g):
                if not g[i] and i not in self.ours_on(h):  # nobody's process at all
                    return h, i
        return None

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
        ok = True
        for src in (C.HERE, C.KD / "snap", C.KD / "bases", C.KD / "dan_code_13e41bbd",
                    C.KD / "pools"):  # fmt: skip
            sh(host, f"mkdir -p {src.parent}")
            p = subprocess.run(["rsync", "-a", "--exclude", "__pycache__", "-e", RSYNC_SSH,
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
        cmd = (f"mkdir -p {sd} {C.KD}/logs && cd {C.HERE} && "
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
        tgt = self.free_gpu()
        if tgt is None:
            s["status"], s["wait_resume"] = "waiting", {"why": why, "src": src, "last": last}
            event(f"{line}: {why}; no free GPU, waiting (resume point {src})")
            return
        th, tg = tgt
        resume = None
        it0 = 0
        if src is not None:
            sh_, it0, path = src
            dst = f"{self.seg_dir(line, k + 1)}/resume_it{it0}.pt"
            if not copy(sh_, path, th, dst):
                event(f"{line}: copy of {path} from {sh_} to {th} failed; retrying next pass")
                return
            resume = dst
        lost = None if last is None else last - it0
        self.launch(line, th, tg, resume)
        event(f"{line}: {why} (seg{k} on {h}); respawned as seg{k + 1} on {th} GPU {tg} from "
              f"it {it0} ({'backup on ' + src[0] if src and src[0] != h else 'its own host'}); "
              f"lost iterations: {lost if lost is not None else 'unknown (host down)'}")  # fmt: skip

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
                    f"echo ===; stat -c %Y {C.KD}/logs/{line}_seg{k}.log 2>/dev/null; true")  # fmt: skip # noqa: E501
        ps = procs(h)
        if out is None or ps is None:
            return
        end, status, mtime = (x.strip() for x in (out.split("===") + ["", ""])[:3])
        alive = (
            "alive"
            if any(kind == "dline" and ln == line and kk == k for _, kind, ln, kk in ps)
            else ""
        )
        if end:
            how = json.loads(end)["how"]
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
        g = self.g.get(h) or {}
        if g.get(seg["gpu"], set()) - {G.ME}:
            if not seg.get("moving") and self.holds():
                sh(h, f"touch {sd}/STOP_AT_CKPT")
                seg["moving"] = now()
                event(f"{line}: another user's process on {h} GPU {seg['gpu']} "
                      f"({sorted(g[seg['gpu']] - {G.ME})}); moving at the next checkpoint")  # fmt: skip

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
            subprocess.run(["rsync", "-a", "--exclude", "ckpts/", "-e", RSYNC_SSH, src, str(dst)],
                           capture_output=True, timeout=1800,
                           env={"SSH_AUTH_SOCK": "", "PATH": "/usr/bin:/bin"})  # fmt: skip

    # ---------------------------------------------------------------- the loop
    def stop_rule(self, line: str) -> None:
        r = M.line_view(line, self.st[line]["segs"])
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
        for line in sorted(C.LINES, key=lambda x: C.LINES[x]["prio"]):
            s = self.st[line]
            try:
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
                    tgt = self.free_gpu()
                    if tgt is not None:
                        self.launch(line, *tgt, None)
                        event(f"{line}: launched on {tgt[0]} GPU {tgt[1]}")
            except Exception as e:  # one line's trouble must not stop the others
                log(f"ERROR {line}: {type(e).__name__}: {e}")
            self.save()
        live = [k for k, v in self.st.items() if not k.startswith("_") and v["status"] != "done"]
        log(f"pass: hosts {[h for h in C.HOSTS if self.up(h)]}; not done: {live}")

    def loop(self) -> None:
        self.poll_hosts()
        self.renew()
        if not self.yielded and self.holds():
            self.reap_unknown([h for h in C.HOSTS if self.up(h)])
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
    if a.standby:
        standby()  # returns when this host is to act
    s = Sup()
    if a.once:
        s.one_pass()
    else:
        s.loop()


if __name__ == "__main__":
    main()
