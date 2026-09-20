"""Run a fixed CPU task list once declared input completion markers exist."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import common


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);args=p.parse_args()
    spec=json.loads(args.config.read_text());spec_hash=common.sha(args.config)
    out=Path(spec["output"]);out.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat(spec["deadline_utc"].replace("Z","+00:00")).timestamp()
    status={"status":"active","pid":os.getpid(),"spec_sha256":spec_hash,"started_utc":datetime.now(timezone.utc).isoformat(),"tasks":{}}
    child=None
    try:
        while time.time()<deadline:
            if common.sha(args.config)!=spec_hash:raise ValueError("Frozen task list changed")
            for task in spec["tasks"]:
                name=task["name"]
                if name in status["tasks"] or not all(Path(p).exists() for p in task.get("ready_paths",[])):continue
                for path,digest in task["input_sha256"].items():
                    if common.sha(path)!=digest:raise ValueError(f"Frozen task input changed: {path}")
                endpoint_hashes={p:common.sha(p) for p in task.get("endpoint_paths",[])}
                record={"status":"active","started_utc":datetime.now(timezone.utc).isoformat(),"endpoint_sha256":endpoint_hashes,"command":task["command"]}
                status["tasks"][name]=record
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                with (out/(name+".log")).open("w") as log:
                    child=subprocess.Popen(task["command"],cwd=common.ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                    record["pid"]=child.pid;common.write_json(out/"progress.json",status);started=time.monotonic()
                    try:code=child.wait(timeout=max(.1,min(task["timeout_seconds"],deadline-time.time())))
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                        record.update(status="timeout",elapsed_seconds=time.monotonic()-started)
                        common.write_json(out/"progress.json",status);continue
                if any(common.sha(p)!=h for p,h in endpoint_hashes.items()):raise ValueError("Endpoint changed during CPU task")
                record.update(status="completed" if code==0 else "failed",exit_code=code,elapsed_seconds=time.monotonic()-started)
                common.write_json(out/"progress.json",status);print(json.dumps({"name":name,**record}),flush=True)
            if len(status["tasks"])==len(spec["tasks"]):
                status["status"]="completed" if all(r["status"]=="completed" for r in status["tasks"].values()) else "completed_with_failures"
                break
            time.sleep(min(10,max(0,deadline-time.time())))
        if status["status"]=="active":status["status"]="deadline_reached"
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        status["finished_utc"]=datetime.now(timezone.utc).isoformat()
        common.write_json(out/"result.json",status);common.write_json(out/"progress.json",status)


if __name__=="__main__":main()
