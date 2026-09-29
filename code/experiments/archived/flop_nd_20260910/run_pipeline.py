"""Sequential frozen stages under the launcher's single owned GPU process group."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import common

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text());config_sha=common.sha(args.config)
    out=Path(config["output"]);out.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat(config["deadline_utc"].replace("Z","+00:00")).timestamp()
    result={"started_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"config_sha256":config_sha,"stages":[],"status":"active"}
    common.write_json(out/"progress.json",result)
    for stage in config["stages"]:
        if common.sha(args.config)!=config_sha:raise ValueError("Frozen pipeline changed")
        for path,digest in stage.get("input_sha256",{}).items():
            if common.sha(path)!=digest:raise ValueError(f"Stage input changed: {path}")
        if time.time()>=deadline:raise TimeoutError("Pipeline deadline reached")
        current={"name":stage["name"],"command":stage["command"],"started_utc":datetime.now(timezone.utc).isoformat(),"status":"active"}
        result["stages"].append(current);common.write_json(out/"progress.json",result)
        started=time.monotonic()
        with (out/(stage["name"]+".log")).open("w",buffering=1) as log:
            child=subprocess.Popen(stage["command"],cwd=common.ROOT,stdout=log,stderr=subprocess.STDOUT)
            current["pid"]=child.pid;common.write_json(out/"progress.json",result)
            try:
                code=child.wait(timeout=min(stage["timeout_seconds"],deadline-time.time()))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                current.update(status="timeout",elapsed_seconds=time.monotonic()-started)
                result["status"]="failed";common.write_json(out/"result.json",result)
                raise
        current.update(exit_code=code,status="completed" if code==0 else "failed",elapsed_seconds=time.monotonic()-started)
        common.write_json(out/"progress.json",result);print(json.dumps(current),flush=True)
        if code:
            result["status"]="failed";common.write_json(out/"result.json",result);raise SystemExit(code)
    result.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat())
    common.write_json(out/"result.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":main()
