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


# ---------------- mocks
cmds = []
POD_RESUME = {}  # alias -> list of paths the pod "has"
END = {}  # (host, line, seg) -> end.json text
ALIVE = set()  # (host, line, seg) with a live dline


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
rest_calls = []


def new_sup(state):
    sup.STATE.write_text(json.dumps(state))
    s = sup.Sup()
    s.holds = lambda: True
    s.renew = lambda: None
    s.backup = lambda line: None
    s.collect = lambda line: None
    s.stop_rule = lambda line: None
    s.rest = lambda m, pod: (
        rest_calls.append((m, pod)) or ((200, {}) if m == "DELETE" else (404, {}))
    )
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
(bd / "seg2_dkoour_s0_it650_rolling.pt").write_text("x")
(bd / "seg2" / "metrics.jsonl").write_text("{}")
(bd / "seg2" / "reads" / "it0650.json").write_text("{}")
rest_calls.clear()
s.pods()
ev = events()
check(
    "idle pod warning (dko_s1 runs on temescal, pinned to rp-dko-s1)",
    "pod rp-dko-s1 (q1w2e3r4t5) is registered for dko_s1, which runs on temescal" in ev,
    ev,
)
check(
    "teardown of rp-s0 only",
    rest_calls == [("DELETE", "vsr5wcacsoq4g8"), ("GET", "vsr5wcacsoq4g8")]
    and s.st["_pods"]["rp-s0"].get("deleted"),
    rest_calls,
)
check("deleted pod no longer polled", "rp-s0" not in s.hosts(), s.hosts())
s.pods()
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


def one_pass_deletes():
    passes.append(1)
    s.st["_pods"].setdefault("rp-s1b", {})["deleted"] = sup.now()


s.one_pass = one_pass_deletes
sup.PASS_S = 0
_sleep = sup.time.sleep
sup.time.sleep = lambda x: None
s.loop()
check(
    "loop passes until the pod is deleted, then exits",
    len(passes) == 1 and "every line has ended; supervisor exits" in events(),
    (passes, events()),
)
reset_events()
s.st["_pods"]["rp-s1b"] = {}
s.one_pass = lambda: passes.append(1)
sup.POD_TEARDOWN_S = 0
s.loop()
check(
    "bounded: an undeletable pod -> event, exit",
    "are not deleted after" in events() and "rp-s1b" in events(),
    events(),
)
sup.time.sleep = _sleep
print("\nFAILED:", FAILS if FAILS else "none")
