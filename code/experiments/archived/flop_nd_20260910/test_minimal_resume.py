"""Exact recovery test for the separately frozen two-proof EMA experiment."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
import common

SNAPSHOT=common.ROOT/"snapshots/minimal_general_v1"
sys.path.insert(0,str(SNAPSHOT))
T=importlib.import_module("symbolic_general_train")


def arguments(out,resume=False):
    return SimpleNamespace(sources=str(common.ROOT/"data/symbolic_seeds.jsonl"),targets=str(common.ROOT/"data/minimal_no_targets.jsonl"),
        out=str(out),mode="reset",seed=4400,steps=512,width=16,layers=1,threads=1,batch=16,minibatch=8,epochs=2,
        lr=1e-4,clip=.2,entropy=.01,gamma=.95,gae_lambda=.95,source_portion=1.,curriculum_window=64,
        curriculum_threshold=.9,max_lines=6,max_scope=4,max_seconds=1e9 if resume else 1000,resume=resume)


def main():
    with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
        directory=Path(temporary);continuous=directory/"continuous";recovered=directory/"recovered"
        T.ppo(arguments(continuous))
        original=T.time.monotonic
        first=original()
        T.time.monotonic=lambda: first+2000 if (recovered/"symbolic_general_recovery.pt").exists() else original()
        try:T.ppo(arguments(recovered))
        finally:T.time.monotonic=original
        interrupted=json.loads((recovered/"symbolic_general_train_summary.json").read_text())
        assert interrupted["status"]=="time_limit" and interrupted["sampled_actions"]<512
        with (recovered/"minimal_curriculum_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"sampled_actions":1000000,"test_stale":True})+"\n")
        T.ppo(arguments(recovered,resume=True))
        a=torch.load(continuous/"symbolic_general_recovery.pt",weights_only=False)
        b=torch.load(recovered/"symbolic_general_recovery.pt",weights_only=False)
        assert all(torch.equal(a["state_dict"][name],b["state_dict"][name]) for name in a["state_dict"])
        for key in ("cuts","curriculum_ema","episodes_at_cut","sampled_actions","optimizer_steps","accounting"):
            assert a[key]==b[key],key
        assert not any(e.get("test_stale") for e in common.read_jsonl(recovered/"minimal_curriculum_events.jsonl"))
        report={"status":"passed","interrupted_after_actions":interrupted["sampled_actions"],"total_actions":512,
                "policy_critic_bitwise_equal":True,"ema_counters_equal":True,"accounting_equal":True,
                "stale_event_removed":True,"test_sha256":common.sha(__file__),
                "trainer_sha256":common.sha(SNAPSHOT/"symbolic_general_train.py")}
        common.write_json(common.ROOT/"reports/minimal_resume_test.json",report)
    print(json.dumps(report))


if __name__=="__main__":main()
