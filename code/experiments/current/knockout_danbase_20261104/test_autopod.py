# ruff: noqa
# pyright: basic, reportAttributeAccessIssue=false, reportOptionalSubscript=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportIndexIssue=false, reportCallIssue=false
"""Run: KD=<an empty scratch dir> python3 test_autopod.py. Mocked dummy-state tests of the pods.json / NEEDS POD supervisor (no ssh, no Runpod calls)."""

import json, os, subprocess, sys, time

W = "/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104"
sys.path.insert(0, W)
import sup

KD = sup.C.KD
assert "kdtest" in str(KD), KD
(KD / "sup").mkdir(parents=True, exist_ok=True)
(sup.SUP / "pods_enabled").write_text("test")
FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def whole_ckpt(path, host):
    """A small real zip, as torch.save writes, and the pod reporting the same size."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.pkl", "x" * 100)
    STAT[host] = str(path.stat().st_size)


# ---------------- mocks
cmds = []
POD_RESUME = {}  # alias -> list of paths the pod "has"
END = {}  # (host, line, seg) -> end.json text
ALIVE = set()  # (host, line, seg) with a live dline
STAT = {}  # host -> the size stat reports for its newest checkpoint


def fake_sh(h, c, timeout=60):
    cmds.append((h, c))
    if "/pod_resume/resume_it*.pt" in c:
        return "\n".join(POD_RESUME.get(h, [])) + "\n"
    if c.startswith("cat ") and "end.json" in c:
        for (hh, ln, k), e in END.items():
            if hh == h and f"/runs/{ln}/seg{k}/end.json" in c:
                return e + "\n===\n{}\n===\n" + str(int(time.time())) + "\n===\n"
        return "\n===\n{}\n===\n" + str(int(time.time())) + "\n===\n"
    if "dline.py --line" in c and "setsid" in c:
        return "started\n"
    if c.startswith("stat -c %s "):
        return STAT.get(h, "") + "\n"
    return ""


sup.sh = fake_sh


def fake_procs(h):
    return [(1, "dline", ln, k) for (hh, ln, k) in ALIVE if hh == h]


sup.procs = fake_procs
sup.kill_seg = lambda h, line, seg, ps=None: []
copies = []


def fake_copy(sh_, src, dh, dst):
    copies.append((sh_, src, dh, dst))
    return True


sup.copy = fake_copy


def new_sup(state):
    sup.STATE.write_text(json.dumps(state))
    s = sup.Sup()
    s.holds = lambda: True
    s.renew = lambda: None
    s.backup = lambda line: None
    s.collect = lambda line: None
    s.stop_rule = lambda line: None
    s.newest_ckpt = lambda line, h, where: CK.get((line, h, "backup" in where))
    s.last_logged = lambda line, h, k: LAST.get(line)
    s.remaining_gpuh = lambda line: 6.4
    s.deploy = lambda h: True  # never rsync from a test
    return s


CK, LAST = {}, {}
seg = lambda h, g, k=1, **kw: {"seg": k, "host": h, "gpu": g, "start": "2026-11-05T00:00:00Z", **kw}
FULL = {
    "serrano": {i: {"bob"} for i in range(5)},
    "ynez": {i: {"bob"} for i in range(4)},
    "temescal": {0: {"bob"}, 1: {"bob"}, 2: {"bob"}},
}


def events():
    return (sup.SUP / "events.md").read_text() if (sup.SUP / "events.md").exists() else ""


def reset_events():
    (sup.SUP / "events.md").write_text("")


def poll(s, g):
    s.read_pods()
    s.g = {**{h: None for h in s.hosts()}, **g}
    for h in s.hosts():
        s.st["_hosts"].setdefault(h, {"fails": 0})


def register(*args, env_home=None):
    env = {**os.environ, "KD": str(KD)}
    if env_home:
        env["HOME"] = env_home
    return subprocess.run(
        [sys.executable, f"{W}/register_pod.py", *args], capture_output=True, text=True, env=env
    )


# ---------------- 0. register the two existing pods; validation
(sup.SUP / "pods.json").unlink(missing_ok=True)
r = register("add", "rp-s0", "vsr5wcacsoq4g8", "dkoour_s0", "2026-11-05T04:59:57Z")
check("register rp-s0", r.returncode == 0, r.stderr)
r = register("add", "rp-s1", "in1myc2jl5fz75", "dkoour_s1", "2026-11-05T04:59:58Z")
check("register rp-s1", r.returncode == 0, r.stderr)
check(
    "bad alias refused",
    register("add", "pod9", "abcdef12", "dko_s1", "2026-11-05T06:00:00Z").returncode != 0,
)
check(
    "unknown line refused",
    register("add", "rp-x", "abcdef12", "nope", "2026-11-05T06:00:00Z").returncode != 0,
)
check(
    "second pod for a pinned line refused without --replace",
    register("add", "rp-x", "abcdef12", "dkoour_s0", "2026-11-05T06:00:00Z").returncode != 0,
)
home = KD / "home"
(home / ".ssh").mkdir(parents=True, exist_ok=True)
(home / ".ssh/config").write_text("Host serrano\n\tuser x\n")
r = register("ssh", "rp-dko-s1", "1.2.3.4", "22001", env_home=str(home))
cfg = (home / ".ssh/config").read_text()
check(
    "ssh alias block added",
    r.returncode == 0
    and "Host rp-dko-s1\n\tHostName 1.2.3.4\n\tPort 22001" in cfg
    and "accept-new" in cfg,
    r.stderr + cfg,
)
check(
    "ssh alias not added twice",
    register("ssh", "rp-dko-s1", "1.2.3.4", "22001", env_home=str(home)).returncode != 0,
)

# ---------------- 1. displacement on a full lab -> one NEEDS POD (primary, nothing to preempt)
reset_events()
cmds.clear()
copies.clear()
st = {
    "dko_s1": {"status": "running", "segs": [seg("temescal", 2, 3)]},
    "dkoour_s0": {"status": "running", "segs": [seg("rp-s0", 0, 2)]},
    "dkoour_s1": {"status": "running", "segs": [seg("rp-s1", 0, 2)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
ALIVE |= {("temescal", "dko_s1", 3), ("rp-s0", "dkoour_s0", 2), ("rp-s1", "dkoour_s1", 2)}
s = new_sup(st)
g = {
    **FULL,
    "temescal": {0: {"bob"}, 1: {"bob"}, 2: {"cowifornia", "alice"}},
    "rp-s0": {0: {"root"}},
    "rp-s1": {0: {"root"}},
}
poll(s, g)
check(
    "pods.json read: both pods pinned",
    s.pinned() == {"dkoour_s0": "rp-s0", "dkoour_s1": "rp-s1"},
    s.pinned(),
)
check("pods polled", set(s.hosts()) >= {"rp-s0", "rp-s1"})
s.check_running("dko_s1")
sg = s.st["dko_s1"]["segs"][-1]
check(
    "other user -> STOP_AT_CKPT and displaced",
    sg.get("moving") and "alice" in (sg.get("displaced") or ""),
    sg,
)
END[("temescal", "dko_s1", 3)] = json.dumps({"how": "moved", "utc": "2026-11-05T06:00:00Z"})
CK[("dko_s1", "temescal", False)] = (480, f"{KD}/runs/dko_s1/seg3/ckpts/dko_s1_it480_rolling.pt")
LAST["dko_s1"] = 480
s.check_running("dko_s1")
ev = events()
check(
    "NEEDS POD event",
    "NEEDS POD: dko_s1 (primary)" in ev and "it 480" in ev and "6.4 GPU-h" in ev,
    ev,
)
check("no plain waiting text in it", "no free GPU, waiting" not in ev, ev)
check(
    "checkpoint put on serrano first",
    copies
    and copies[-1][2] == "serrano"
    and copies[-1][3].endswith("backup/dko_s1/seg3_dko_s1_it480_rolling.pt"),
    copies,
)
check("line waits", s.st["dko_s1"]["status"] == "waiting")
# next pass: no second event
reset_events()
w = s.st["dko_s1"].pop("wait_resume")
s.st["dko_s1"]["status"] = "running"
s.respawn("dko_s1", w["why"])
check(
    "no second NEEDS POD",
    "NEEDS POD" not in events() and s.st["dko_s1"]["status"] == "waiting",
    events(),
)

# ---------------- 2. coordinator registers a pod for dko_s1; checkpoint missing, then stale, then good
reset_events()
r = register("add", "rp-dko-s1", "q1w2e3r4t5", "dko_s1", "2026-11-05T06:10:00Z", "--gpuh", "6.4")
check(
    "register rp-dko-s1 with budget",
    r.returncode == 0 and "+$4.91" in r.stdout,
    r.stdout + r.stderr,
)
g["rp-dko-s1"] = {0: set()}
poll(s, g)
check("dko_s1 now pinned", s.pinned().get("dko_s1") == "rp-dko-s1")
w = s.st["dko_s1"].pop("wait_resume")
s.st["dko_s1"]["status"] = "running"
s.respawn("dko_s1", w["why"])
ev = events()
check(
    "missing pod checkpoint -> waits, one event",
    "waiting for its resume checkpoint there" in ev and s.st["dko_s1"]["status"] == "waiting",
    ev,
)
POD_RESUME["rp-dko-s1"] = [f"{KD}/runs/dko_s1/pod_resume/resume_it475.pt"]
reset_events()
w = s.st["dko_s1"].pop("wait_resume")
s.st["dko_s1"]["status"] = "running"
s.respawn("dko_s1", w["why"])
check(
    "stale pod checkpoint (475 < 480) -> still waits, no repeat event",
    s.st["dko_s1"]["status"] == "waiting" and events() == "",
    events(),
)
POD_RESUME["rp-dko-s1"].append(f"{KD}/runs/dko_s1/pod_resume/resume_it480.pt")
cmds.clear()
w = s.st["dko_s1"].pop("wait_resume")
s.st["dko_s1"]["status"] = "running"
s.respawn("dko_s1", w["why"])
last = s.st["dko_s1"]["segs"][-1]
check(
    "resumed on the pod from pod_resume it 480",
    last["host"] == "rp-dko-s1"
    and last["resume"].endswith("pod_resume/resume_it480.pt")
    and last["seg"] == len(s.st["dko_s1"]["segs"]),
    last,
)
lc = [c for h, c in cmds if "setsid" in c][-1]
check(
    "pod launch exports DANBASE_PY and parses",
    "export DANBASE_PY=/workspace/venv/bin/python && (" in lc
    and subprocess.run(["bash", "-n", "-c", lc]).returncode == 0,
    lc,
)
check("no copy to the pod", not any(c[2].startswith("rp-") for c in copies), copies)
check(
    "respawn event names the coordinator's copy",
    "the coordinator's copy on the pod" in events(),
    events(),
)
check("no POD NOT NEEDED for a pod resume", "POD NOT NEEDED" not in events())

# ---------------- 3a. full lab: a displaced primary gets NEEDS POD at once (a preemption could not help)
reset_events()
copies.clear()
st = {
    "dko_s0": {"status": "running", "segs": [seg("serrano", 1, 1)]},
    "dkoour_s2": {"status": "running", "segs": [seg("ynez", 3, 1)]},
    "dkoour_s0": {"status": "running", "segs": [seg("rp-s0", 0, 1)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
ALIVE |= {("serrano", "dko_s0", 1), ("ynez", "dkoour_s2", 1)}
s = new_sup(st)
g3 = {
    **{h: dict(v) for h, v in FULL.items()},
    "rp-s0": {0: {"root"}},
    "rp-s1": {0: {"root"}},
    "rp-dko-s1": {0: {"root"}},
}
g3["serrano"][1] = {"cowifornia", "carol"}
g3["ynez"][3] = {"cowifornia"}
poll(s, g3)
check(
    "pod line is no preemption candidate",
    s.preempt_candidates() == ["dkoour_s2"],
    s.preempt_candidates(),
)
check("full lab: preemption would not help", not s.preempt_helps())
s.check_running("dko_s0")
END[("serrano", "dko_s0", 1)] = json.dumps({"how": "moved", "utc": "2026-11-05T07:00:00Z"})
CK[("dko_s0", "serrano", False)] = (500, f"{KD}/runs/dko_s0/seg1/ckpts/dko_s0_it500_rolling.pt")
s.g["serrano"][1] = {"carol"}
s.check_running("dko_s0")
check("full lab: primary NEEDS POD", "NEEDS POD: dko_s0 (primary)" in events(), events())
s.preempt()
check("full lab: no preemption", not s.st["dkoour_s2"]["segs"][-1].get("preempted_for"))

# ---------------- 3b. 2 lab GPUs free: the primary preempts instead; the secondary inherits
reset_events()
copies.clear()
END.clear()
st = {
    "dko_s0": {"status": "running", "segs": [seg("serrano", 1, 1)]},
    "dkoour_s2": {"status": "running", "segs": [seg("ynez", 3, 1)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
s = new_sup(st)
g3 = {
    **{h: dict(v) for h, v in FULL.items()},
    "rp-s0": {0: {"root"}},
    "rp-s1": {0: {"root"}},
    "rp-dko-s1": {0: {"root"}},
}
g3["serrano"][1] = {"cowifornia", "carol"}
g3["ynez"][3] = {"cowifornia"}
g3["temescal"][2] = set()
g3["ynez"][0] = set()
poll(s, g3)
check("2 free: preemption helps", s.preempt_helps())
s.check_running("dko_s0")
END[("serrano", "dko_s0", 1)] = json.dumps({"how": "moved", "utc": "2026-11-05T07:00:00Z"})
s.g["serrano"][1] = {"carol"}
s.check_running("dko_s0")
check(
    "2 free: primary waits without NEEDS POD",
    "NEEDS POD" not in events() and s.st["dko_s0"]["status"] == "waiting",
    events(),
)
s.preempt()
sg2 = s.st["dkoour_s2"]["segs"][-1]
check(
    "secondary preempted and inherits displacement",
    sg2.get("preempted_for") == "dko_s0" and "carol" in (sg2.get("displaced") or ""),
    sg2,
)
END[("ynez", "dkoour_s2", 1)] = json.dumps({"how": "moved", "utc": "2026-11-05T07:05:00Z"})
CK[("dkoour_s2", "ynez", False)] = (
    420,
    f"{KD}/runs/dkoour_s2/seg1/ckpts/dkoour_s2_it420_rolling.pt",
)
s.g["ynez"][3] = set()
s.check_running("dkoour_s2")  # ends -> waits behind dko_s0 (plain waiting)
w = s.st["dko_s0"].pop("wait_resume")
s.st["dko_s0"]["status"] = "running"
s.respawn("dko_s0", w["why"])
check(
    "primary resumed on a lab GPU",
    s.st["dko_s0"]["segs"][-1]["host"] in ("ynez", "temescal")
    and not s.st["dko_s0"]["segs"][-1].get("dead"),
    s.st["dko_s0"]["segs"][-1],
)
pg = s.st["dko_s0"]["segs"][-1]
s.g[pg["host"]][pg["gpu"]] = {"cowifornia"}
w = s.st["dkoour_s2"].pop("wait_resume")
s.st["dkoour_s2"]["status"] = "running"
s.respawn("dkoour_s2", w["why"])
check(
    "secondary displaced via preemption gets one NEEDS POD",
    events().count("NEEDS POD: dkoour_s2 (secondary)") == 1 and "preempted for dko_s0" in events(),
    events() + json.dumps(s.st["dkoour_s2"]) + str(s.lab_free()) + str(s.n_used()),
)

# ---------------- 4. POD NOT NEEDED when a lab GPU frees first
reset_events()
s.g["temescal"][0] = set()
s.g["temescal"][1] = set()
s.g["serrano"][4] = set()
w = s.st["dkoour_s2"].pop("wait_resume")
s.st["dkoour_s2"]["status"] = "running"
s.respawn("dkoour_s2", w["why"])
check("POD NOT NEEDED event", "POD NOT NEEDED: dkoour_s2 resumed on" in events(), events())

# ---------------- 5. idle pod warning; pod down -> NEEDS POD; teardown; spend; ceiling
reset_events()
st = {
    "dko_s1": {"status": "running", "segs": [seg("temescal", 0, 4)]},
    "dkoour_s1": {"status": "running", "segs": [seg("rp-s1", 0, 2)]},
    "dkoour_s0": {"status": "done", "segs": [seg("rp-s0", 0, 2)]},
    "_hosts": {
        h: {"fails": 0} for h in ("serrano", "temescal", "ynez", "rp-s0", "rp-s1", "rp-dko-s1")
    },
}
s = new_sup(st)
poll(s, {**FULL, "rp-s0": {0: set()}, "rp-s1": None, "rp-dko-s1": {0: set()}})
bd = KD / "backup" / "dkoour_s0"
(bd / "seg2" / "reads").mkdir(parents=True, exist_ok=True)
whole_ckpt(bd / "seg2_dkoour_s0_it650_rolling.pt", "rp-s0")
CK[("dkoour_s0", "rp-s0", False)] = (
    650,
    f"{KD}/runs/dkoour_s0/seg2/ckpts/dkoour_s0_it650_rolling.pt",
)
(bd / "seg2" / "metrics.jsonl").write_text("{}")
(bd / "seg2" / "reads" / "it0650.json").write_text("{}")
s.pods()
ev = events()
check(
    "idle pod warning (dko_s1 runs on temescal, pinned to rp-dko-s1)",
    "pod rp-dko-s1 (q1w2e3r4t5) is registered for dko_s1, which runs on temescal" in ev,
    ev,
)
READY = (
    "POD READY TO DELETE: rp-s0 vsr5wcacsoq4g8 (dkoour_s0); backup verified; still billing $0.59/h"
)
check(
    "a finished line's verified pod is reported, in the exact format",
    READY in ev
    and s.st["_pods"]["rp-s0"].get("ready_to_delete")
    and "POD READY TO DELETE: rp-s1" not in ev,
    ev,
)
check("nothing marks it deleted", not s.st["_pods"]["rp-s0"].get("deleted"), s.st["_pods"]["rp-s0"])
check("a ready pod is no longer polled", "rp-s0" not in s.hosts(), s.hosts())
spend_a = s.pod_spend()
s.pods()
check("the ready event fires once", events().count("POD READY TO DELETE") == 1, events())
check(
    "no way to reach Runpod is left in sup.py",
    not hasattr(sup.Sup, "rest")
    and not any(
        x in (sup.Path(sup.__file__)).read_text()
        for x in ("runpod.io", "runpod_env", "RUNPOD_API_KEY")
    ),
)
check("idle warning once per segment", events().count("is registered for dko_s1") == 1)
# pod down
s.st["_hosts"]["rp-s1"]["fails"] = 3
CK[("dkoour_s1", "serrano", True)] = (300, f"{KD}/backup/dkoour_s1/seg2_dkoour_s1_it300_rolling.pt")
copies.clear()
reset_events()
s.check_running("dkoour_s1")
ev = events()
check(
    "pod down -> NEEDS POD naming the pod",
    "NEEDS POD: dkoour_s1 (secondary): its pod rp-s1 is down" in ev
    and "backup/dkoour_s1/seg2_dkoour_s1_it300_rolling.pt (it 300)" in ev,
    ev,
)
check("no copy needed when the checkpoint is already in serrano's backup", copies == [], copies)
r = register("add", "rp-s1b", "z9y8x7w6", "dkoour_s1", "2026-11-05T08:00:00Z", "--replace")
check(
    "--replace retires rp-s1",
    r.returncode == 0 and "retired ['rp-s1']" in r.stdout,
    r.stdout + r.stderr,
)
s.read_pods()
check("replacement pinned", s.pinned()["dkoour_s1"] == "rp-s1b")
# spend and ceiling
cfg = json.loads((sup.SUP / "pods.json").read_text())
check("ceiling = 16.90 + 4.91", abs(s.ceiling() - 21.81) < 1e-6, s.ceiling())
sp = s.pod_spend()
exp = sum(
    max(
        0.0,
        min(
            time.time(),
            sup.M._utc(
                s.st["_pods"].get(h, {}).get("deleted")
                or v.get("retired")
                or "2100-01-01T00:00:00Z"
            ),
        )
        - sup.M._utc(v["created"]),
    )
    / 3600
    * v["rate"]
    for h, v in cfg.items()
)
check("spend over every pod (deleted / retired stop the clock)", abs(sp - exp) < 0.01, (sp, exp))
s.pod_spend = lambda: 30.0  # over the 21.81 ceiling
s.st["_pods"].pop("budget_warned_at", None)  # (real time may have crossed it already)
reset_events()
s.pods()
s.pods()
check("ceiling event once", events().count("has reached the ceiling $21.81") == 1, events())
s.pods_cfg["rp-s1b"]["budget_add"] = 23.1  # a new pod raises the ceiling to 44.91
reset_events()
s.pods()
check("no event under the raised ceiling", "has reached the ceiling" not in events(), events())
s.pod_spend = lambda: 45.0
s.pods()
s.pods()
check(
    "ceiling event re-arms for the raised ceiling",
    events().count("has reached the ceiling $44.91") == 1,
    events(),
)

# ---------------- 6. invalid pods.json keeps the last good read
good = dict(s.pods_cfg)
(sup.SUP / "pods.json").write_text("{not json")
s.read_pods()
check("invalid pods.json -> last good read kept", s.pods_cfg == good)

# ---------------- 7. the loop keeps passing until a finished line's pod is deleted (bounded)
reset_events()
st = {
    k: {"status": "done", "segs": [seg("rp-s1b", 0, 3)] if k == "dkoour_s1" else []}
    for k in sup.C.LINES
}
st["_hosts"] = {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")}
st["_pods"] = {
    "rp-s0": {"deleted": "2026-11-05T09:00:00Z"},
    "rp-dko-s1": {"deleted": "2026-11-05T09:00:00Z"},
}
(sup.SUP / "pods.json").write_text(json.dumps(good))
s = new_sup(st)
s.poll_hosts = lambda: None
s.reap_unknown = lambda hosts: None
check(
    "pods_pending lists the finished line's pod", s.pods_pending() == ["rp-s1b"], s.pods_pending()
)
passes = []


def one_pass_retires():
    passes.append(1)
    s.pods_cfg["rp-s1b"]["retired"] = sup.now()


s.one_pass = one_pass_retires
sup.PASS_S = 0
_sleep = sup.time.sleep
sup.time.sleep = lambda x: None
s.loop()
check(
    "loop passes until the pod is retired, then exits",
    len(passes) == 1 and "every line has ended; supervisor exits" in events(),
    (passes, events()),
)
reset_events()
s.pods_cfg["rp-s1b"].pop("retired")
s.one_pass = lambda: passes.append(1)
sup.POD_TEARDOWN_S = 0
s.loop()
check(
    "bounded: a pod never retired -> event, exit",
    "are not retired after" in events()
    and "may still be billing" in events()
    and "rp-s1b" in events(),
    events(),
)
sup.time.sleep = _sleep

# ---------------- 8. a broken host: lines there are ended and respawned elsewhere, never placed there
reset_events()
(sup.SUP / "broken_hosts.json").write_text(json.dumps({"temescal": "CUDA unknown error"}))
(sup.SUP / "pods.json").write_text("{}")
st = {
    "dd01_s1": {"status": "running", "segs": [seg("temescal", 1, 1)]},
    "dd01_s2": {"status": "running", "segs": [seg("temescal", 2, 1)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
ALIVE |= {("temescal", "dd01_s1", 1), ("temescal", "dd01_s2", 1)}
s = new_sup(st)
g8 = {
    "serrano": {0: {"x"}, 1: {"cowifornia"}, 2: {"x"}, 3: {"x"}, 4: {"x"}},
    "temescal": {1: set(), 2: set()},
    "ynez": {0: {"cowifornia"}, 1: set(), 2: set(), 3: set()},
}
poll(s, g8)
check("broken host: no GPU counts as free there", s.lab_free() == 3, s.lab_free())
s.check_running("dd01_s1")
l1 = s.st["dd01_s1"]["segs"][-1]
check(
    "line on a broken host respawned off it",
    l1["host"] == "ynez" and l1["gpu"] == 1 and "marked broken" in events(),
    (l1, events()),
)
s.check_running("dd01_s2")  # (nvidia-smi does not show the new segment yet)
check(
    "the second waits (2 lab GPUs must stay free), never on the broken host",
    s.st["dd01_s2"]["status"] == "waiting" and s.free_gpu("dd01_s2") is None,
    s.st["dd01_s2"],
)
(sup.SUP / "broken_hosts.json").unlink()

# ---------------- 9. hysteresis: a GPU another user used in the last 15 min is not free
reset_events()
ago = lambda m: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60 * m))
st = {
    "dd01_s1": {
        "status": "running",
        "segs": [
            seg(
                "ynez",
                1,
                2,
                moving=ago(3),
                displaced="displaced by another user's process on ynez GPU 1",
            )
        ],
    },
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
    "_gpu_seen": {"ynez": {"0": ago(5), "1": ago(2), "2": ago(5), "3": ago(20)}},
}
ALIVE |= {("ynez", "dd01_s1", 2)}
s = new_sup(st)
g9 = {
    "serrano": {i: {"x"} for i in range(5)},
    "temescal": {1: set(), 2: set()},
    "ynez": {0: set(), 1: {"cowifornia"}, 2: set(), 3: set()},
}
poll(s, g9)
check(
    "recently used GPUs are not free; a quiet one is",
    not s.quiet("ynez", 0) and s.quiet("ynez", 3),
    s.st["_gpu_seen"],
)
check("lab_free counts only quiet GPUs (temescal 1-2 + ynez 3)", s.lab_free() == 3, s.lab_free())
s.check_running("dd01_s1")
check(
    "move not cancelled during a gap",
    s.st["dd01_s1"]["segs"][-1].get("moving") and "move cancelled" not in events(),
    events(),
)
s.st["_gpu_seen"]["ynez"]["1"] = ago(16)
s.check_running("dd01_s1")
check(
    "move cancelled after 15 quiet min",
    "move cancelled" in events() and not s.st["dd01_s1"]["segs"][-1].get("moving"),
    events(),
)
pg = sup.G.gpus
poll_calls = []
s.g = {"ynez": {0: {"ColinDoherty"}, 1: set()}}

# ---------------- 10. pod reuse: a finished line's pod goes to a displaced line waiting for a pod
reset_events()
(sup.SUP / "pods.json").write_text(
    json.dumps({"rp-d1": {"pod": "aaaa1111", "line": "dd1_s0", "created": ago(60), "rate": 0.59}})
)
(sup.SUP / "no_pod.json").write_text(json.dumps(["dd01_s1"]))
st = {
    "dd1_s0": {"status": "done", "segs": [seg("rp-d1", 0, 1, end=ago(1))]},
    "dd1_s2": {
        "status": "waiting",
        "segs": [
            seg(
                "ynez",
                0,
                1,
                dead="moved",
                needs_pod=ago(2),
                displaced="displaced by another user's process on ynez GPU 0",
            )
        ],
    },
    "dd01_s1": {
        "status": "waiting",
        "segs": [
            seg("ynez", 1, 2, dead="moved", needs_pod=ago(3), displaced="displaced by another user")
        ],
    },
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
s = new_sup(st)
poll(s, {**g9, "rp-d1": {0: set()}})
bd = KD / "backup" / "dd1_s0"
(bd / "seg1" / "reads").mkdir(parents=True, exist_ok=True)
whole_ckpt(bd / "seg1_dd1_s0_it80_rolling.pt", "rp-d1")
CK[("dd1_s0", "rp-d1", False)] = (80, f"{KD}/runs/dd1_s0/seg1/ckpts/dd1_s0_it80_rolling.pt")
(bd / "seg1" / "metrics.jsonl").write_text("{}")
(bd / "seg1" / "reads" / "it0080.json").write_text("{}")
check(
    "reuse candidates skip held-back lines",
    s.reuse_candidates() == ["dd1_s2"],
    s.reuse_candidates(),
)
s.pods()
cfg = json.loads((sup.SUP / "pods.json").read_text())
check(
    "pod re-pointed, not deleted",
    cfg["rp-d1"]["line"] == "dd1_s2"
    and cfg["rp-d1"]["reused_from"] == "dd1_s0"
    and "POD READY TO DELETE" not in events()
    and "reused for dd1_s2" in events(),
    (cfg, events()),
)
check("dd1_s2 now pinned to rp-d1", s.pinned().get("dd1_s2") == "rp-d1")
# hold while a displacement is under way, then delete after the window
reset_events()
(sup.SUP / "pods.json").write_text(
    json.dumps({"rp-d2": {"pod": "bbbb2222", "line": "dd1_s1", "created": ago(60), "rate": 0.59}})
)
st = {
    "dd1_s1": {"status": "done", "segs": [seg("rp-d2", 0, 1, end=ago(1))]},
    "dd1_s2": {
        "status": "running",
        "segs": [seg("ynez", 0, 1, moving=ago(1), displaced="displaced by another user")],
    },
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
s = new_sup(st)
poll(s, {**g9, "rp-d2": {0: set()}})
bd = KD / "backup" / "dd1_s1"
(bd / "seg1" / "reads").mkdir(parents=True, exist_ok=True)
whole_ckpt(bd / "seg1_dd1_s1_it70_rolling.pt", "rp-d2")
CK[("dd1_s1", "rp-d2", False)] = (70, f"{KD}/runs/dd1_s1/seg1/ckpts/dd1_s1_it70_rolling.pt")
(bd / "seg1" / "metrics.jsonl").write_text("{}")
(bd / "seg1" / "reads" / "it0070.json").write_text("{}")
s.pods()
check(
    "pod held while a line is being displaced",
    "POD READY TO DELETE" not in events() and "held up to 20 min" in events(),
    events(),
)
s.st["dd1_s1"]["segs"][-1]["end"] = ago(21)
s.pods()
check(
    "reported ready after the 20-min window",
    "POD READY TO DELETE: rp-d2 bbbb2222 (dd1_s1); backup verified; still billing $0.59/h"
    in events(),
    events(),
)
# the spend clock runs until retire, which takes the pod out of pods()
s.read_pods()
now_a = s.pod_spend()
s.st["_pods"]["rp-d2"]["ready_to_delete"] = ago(30)
check("spend still counts a ready pod", now_a >= 0.59 * 59 / 60, now_a)
r = register("retire", "rp-d2")
s.read_pods()
check(
    "retire works on a ready pod",
    r.returncode == 0 and s.pods_cfg["rp-d2"].get("retired"),
    r.stdout + r.stderr,
)
spend_b = s.pod_spend()
reset_events()
s.st["_pods"]["rp-d2"].pop("ready_to_delete")
s.pods()
check(
    "a retired pod is out of pods()",
    "POD READY" not in events() and s.pods_pending() == [],
    (events(), s.pods_pending()),
)
check(
    "retire stops the spend clock",
    abs(s.pod_spend() - spend_b) < 1e-6 and spend_b <= now_a + 1e-3,
    (spend_b, now_a),
)
(sup.SUP / "no_pod.json").unlink()

# ---------------- 10b. "backup verified" means the pod's newest checkpoint, whole, in the backup
reset_events()
(sup.SUP / "pods.json").write_text(
    json.dumps({"rp-d5": {"pod": "cccc3333", "line": "dd01_s0", "created": ago(60), "rate": 0.59}})
)
st = {
    "dd01_s0": {"status": "done", "segs": [seg("rp-d5", 0, 2, end=ago(1))]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
s = new_sup(st)
s.last_logged = lambda line, host, k: 692
poll(s, {**g9, "rp-d5": {0: set()}})
bd = KD / "backup" / "dd01_s0"
(bd / "seg2" / "reads").mkdir(parents=True, exist_ok=True)
whole_ckpt(bd / "seg2_dd01_s0_it685_alarm.pt", "rp-d5")
(bd / "seg2" / "metrics.jsonl").write_text("{}")
(bd / "seg2" / "reads" / "it0690.json").write_text("{}")
CK[("dd01_s0", "rp-d5", False)] = (690, f"{KD}/runs/dd01_s0/seg2/ckpts/dd01_s0_it690_rolling.pt")
s.pods()
check(
    "an older checkpoint in the backup is not enough: the pod is kept",
    "POD READY" not in events()
    and "backup on serrano is incomplete (seg2_dd01_s0_it690_rolling.pt is not in the backup)"
    in events(),
    events(),
)
good = bd / "seg2_dd01_s0_it690_rolling.pt"
whole_ckpt(good, "rp-d5")
whole = good.read_bytes()
good.write_bytes(whole[: len(whole) // 2])
STAT["rp-d5"] = str(good.stat().st_size)  # the kill cut the pod's file too: the sizes agree
s.st["_pods"]["rp-d5"].pop("verify_warned", None)
reset_events()
s.pods()
check(
    "a checkpoint cut short by the kill is not enough: the pod is kept",
    "POD READY" not in events() and "is not a whole file" in events(),
    events(),
)
good.write_bytes(whole)
STAT["rp-d5"] = "1"
s.st["_pods"]["rp-d5"].pop("verify_warned", None)
reset_events()
s.pods()
check(
    "a size that differs from the pod's is not enough: the pod is kept",
    "POD READY" not in events() and "1 on rp-d5" in events(),
    events(),
)
STAT["rp-d5"] = str(good.stat().st_size)
reset_events()
s.pods()
check(
    "the ready event names the resume checkpoint and the last logged iteration",
    "POD READY TO DELETE: rp-d5 cccc3333 (dd01_s0); backup verified; still billing $0.59/h; "
    "resume checkpoint in the backup: it 690, last logged it 692" in events(),
    events(),
)
(sup.SUP / "pods.json").write_text("{}")

# ---------------- 11. the segment's own sighting counts: no cancel on a poll that fell in a gap
reset_events()
SAC_T = int(time.time()) - 30
_fake_sh = sup.sh


def sh_with_sac(h, c, timeout=60):
    if c.startswith("cat ") and "end.json" in c and "/runs/dd01_s1/seg3/" in c:
        return "\n===\n{}\n===\n" + str(int(time.time())) + "\n===\n" + str(SAC_T) + "\n"
    return _fake_sh(h, c, timeout)


sup.sh = sh_with_sac
st = {
    "dd01_s1": {
        "status": "running",
        "segs": [
            seg(
                "ynez",
                0,
                3,
                moving=ago(1),
                displaced="displaced by another user's process on ynez GPU 0 (the segment's own check)",
            )
        ],
    },
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
    "_gpu_seen": {"ynez": {"0": ago(35)}},
}
ALIVE |= {("ynez", "dd01_s1", 3)}
s = new_sup(st)
poll(s, {**g9, "ynez": {0: {"cowifornia"}, 1: set(), 2: set(), 3: set()}})
s.check_running("dd01_s1")
check(
    "own sighting recorded; move kept",
    "move cancelled" not in events()
    and s.st["dd01_s1"]["segs"][-1].get("moving")
    and not s.quiet("ynez", 0),
    (events(), s.st["_gpu_seen"]),
)
sup.sh = _fake_sh

# ---------------- 11b. a pod's old alarm checkpoints are pruned; a lab line's are not touched
st = {
    "dd01_s0": {"status": "running", "segs": [seg("rp-d5", 0, 2)]},
    "dd01_s1": {"status": "running", "segs": [seg("serrano", 1, 3)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
s = new_sup(st)
s.up = lambda h: True
cmds.clear()
s.prune_pod("dd01_s0")
pruned = [c for h, c in cmds if h == "rp-d5" and "_alarm.pt" in c]
check(
    "pod line: old alarm checkpoints pruned, the newest 4 kept",
    len(pruned) == 1
    and "/runs/dd01_s0/seg2/ckpts" in pruned[0]
    and "tail -n +5" in pruned[0]
    and "dd01_s0_it*_alarm.pt" in pruned[0]
    and "_rolling" not in pruned[0]
    and "_snap" not in pruned[0],
    cmds,
)
cmds.clear()
s.prune_pod("dd01_s1")
check("lab line: nothing pruned", not [c for _, c in cmds if "rm -f" in c], cmds)

# ---------------- 11c. the guard-off pilot (PREREG.md, Amendment 2)
C_ = sup.C
a_ref = C_.grpo_argv("dd01_s2", KD / "x", 1200, None)
a_pil = C_.grpo_argv("dd01g0_s2", KD / "x", 1200, None)
strip = lambda v, name: [q.replace(name, "LINE") for q in v]
ref_only = [q for q in strip(a_ref, "dd01_s2") if q not in strip(a_pil, "dd01g0_s2")]
pil_only = [q for q in strip(a_pil, "dd01g0_s2") if q not in strip(a_ref, "dd01_s2")]
check(
    "the pilot's flags differ from dd01_s2's only in the guard and the two storage/stop flags",
    set(ref_only) == {"--rollback_on_alarm", "--max_rollbacks", "3"}
    and set(pil_only) == {"--no_alarm_ckpt", "--stop_nonfinite"}
    and a_pil[a_pil.index("--init") + 1] == a_ref[a_ref.index("--init") + 1]
    and a_pil[a_pil.index("--seed") + 1] == "2",
    (ref_only, pil_only),
)
sr = sup.M.stop_reason
v0 = {"x_dan": 2.9, "x_cap": 5.71, "gpuh": 9.9, "cross_dan": None, "curve": []}
check("the pilot runs on below its limits", sr("dd01g0_s2", v0) is None, sr("dd01g0_s2", v0))
check("the pilot stops at x 3.0", "the pilot's 3" in (sr("dd01g0_s2", {**v0, "x_dan": 3.0}) or ""))
check("the pilot stops at 10 GPU-h", "≥ 10" in (sr("dd01g0_s2", {**v0, "gpuh": 10.0}) or ""))
check("dd01_s2 keeps its 15 GPU-h", sr("dd01_s2", {**v0, "gpuh": 10.0}) is None)
check("dd01_s2 keeps his round-16 x", sr("dd01_s2", {**v0, "x_dan": 3.0}) is None)
reset_events()
(sup.SUP / "pods.json").write_text("{}")
st = {
    "dd01g0_s2": {"status": "running", "segs": [seg("ynez", 3, 1)]},
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
}
ALIVE |= {("ynez", "dd01g0_s2", 1)}
END[("ynez", "dd01g0_s2", 1)] = json.dumps(
    {"how": "nonfinite", "rc": 4, "utc": ago(0), "host": "ynez"}
)
s = new_sup(st)
poll(s, {**g9, "ynez": {0: set(), 1: set(), 2: set(), 3: set()}})
s.check_running("dd01g0_s2")
check(
    "a non-finite stop ends the line as a result, with no respawn",
    s.st["dd01g0_s2"]["status"] == "done"
    and s.st["dd01g0_s2"].get("stop_reason") == "non-finite loss, grad norm or weights"
    and len(s.st["dd01g0_s2"]["segs"]) == 1
    and "ended (nonfinite)" in events(),
    (s.st["dd01g0_s2"], events()),
)
END.pop(("ynez", "dd01g0_s2", 1))
ALIVE.discard(("ynez", "dd01g0_s2", 1))

# ---------------- 12. a relaunching job: 2+ sightings in 2 h -> 60 quiet min needed
st = {
    "_hosts": {h: {"fails": 0} for h in ("serrano", "temescal", "ynez")},
    "_gpu_seen": {"ynez": {"2": ago(25), "3": ago(25)}},
    "_gpu_hits": {"ynez": {"2": [ago(40), ago(25)], "3": [ago(25)]}},
}
s = new_sup(st)
poll(s, {**g9, "ynez": {0: set(), 1: set(), 2: set(), 3: set()}})
check(
    "cycling GPU needs 60 quiet min; a single sighting 15",
    not s.quiet("ynez", 2) and s.quiet("ynez", 3),
    s.st["_gpu_hits"],
)
s.sighting("ynez", 3, ago(1))
check("a new sighting resets the clock", not s.quiet("ynez", 3))
print("\nFAILED:", FAILS if FAILS else "none")
