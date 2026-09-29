"""Evaluate fixed development checkpoints as they appear, without selecting models."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text()); config_hash=common.sha(args.config)
    out=Path(config["output"]); out.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat(config["deadline_utc"].replace("Z","+00:00")).timestamp()
    for path,digest in config["input_sha256"].items():
        if common.sha(path)!=digest:raise ValueError(f"Frozen input changed: {path}")
    result={"status":"active","pid":os.getpid(),"config_sha256":config_hash,
            "started_utc":datetime.now(timezone.utc).isoformat(),"evaluations":{},
            "policy":"Every declared checkpoint is evaluated on development; no selection or training changes."}
    child=None
    def interrupted(sig,frame):raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        while time.time()<deadline:
            if common.sha(args.config)!=config_hash:raise ValueError("Frozen watcher configuration changed")
            for task in config["evaluations"]:
                name=task["name"]
                if name in result["evaluations"]:continue
                checkpoint=Path(task["checkpoint"])
                if not checkpoint.exists():
                    if (checkpoint.parent/"result.json").exists():
                        result["evaluations"][name]={"status":"unavailable_at_training_endpoint"}
                    continue
                remaining=deadline-time.time()
                if remaining<=0:break
                digest=common.sha(checkpoint)
                cmd=[config["python"],config["evaluator"],"--checkpoint",str(checkpoint),
                     "--data",config["data"],"--audit",config["audit"],"--output",task["output"],
                     "--temperature","0","--attempts","1","--max-new-tokens","512",
                     "--batch-size","32","--seed","202609101935","--device","cpu"]
                record={"status":"active","checkpoint_sha256":digest,"command":cmd,
                        "started_utc":datetime.now(timezone.utc).isoformat()}
                result["evaluations"][name]=record
                with (out/f"{name}.log").open("w",buffering=1) as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=common.ROOT)
                    record["pid"]=child.pid;common.write_json(out/"progress.json",result)
                    started=time.monotonic()
                    try:exit_code=child.wait(timeout=min(config["per_evaluation_timeout_seconds"],remaining))
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                        record.update(status="timeout",elapsed_seconds=time.monotonic()-started)
                        common.write_json(out/"progress.json",result)
                        continue
                record.update(status="completed" if exit_code==0 else "failed",exit_code=exit_code,
                              elapsed_seconds=time.monotonic()-started)
                if common.sha(checkpoint)!=digest:raise ValueError("Checkpoint changed during evaluation")
                common.write_json(out/"progress.json",result)
                print(json.dumps({"name":name,**record}),flush=True)
            if len(result["evaluations"])==len(config["evaluations"]):
                result["status"]="completed";break
            time.sleep(min(10,max(0,deadline-time.time())))
        if result["status"]=="active":result["status"]="deadline_reached"
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        result["finished_utc"]=datetime.now(timezone.utc).isoformat()
        common.write_json(out/"result.json",result)


if __name__=="__main__":main()
