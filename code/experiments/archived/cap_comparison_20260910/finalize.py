"""Replay newly completed arms and refresh their portable comparison report."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    lock=(root / "reports/finalization.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    arms=json.loads((root / "plans/protocol.json").read_text())["arms"]
    last=[]
    while True:
        state=json.loads((root / "reports/recovery_state.json").read_text())
        # The coordinator publishes success only after dispatcher exit and the
        # final remote rsync. A copied completion marker alone can arrive first.
        complete=[name for name in arms if state["finished"].get(name,{}).get("success") is True
                  and (root / "runs" / name / "pipeline_completed.json").exists()]
        if complete and complete != last:
            full=len(complete)==len(arms)
            for command in (
                [sys.executable,str(root / "code/analyze.py"),"--root",str(root),*([] if full else ["--partial"])],
                *[[sys.executable,str(root / "code/training_diagnostics.py"),"--root",str(root),"--arm",name]
                  for name in complete if not (root / "reports" / f"{name}_training_information.json").exists()],
                [sys.executable,str(root / "code/report.py"),"--root",str(root)]):
                subprocess.run(command,cwd=root,check=True)
            last=complete
            status={"updated_utc":datetime.now(timezone.utc).isoformat(),"complete":full,"completed_arms":complete,
                    "pending_arms":[name for name in arms if name not in complete]}
            path=root / "reports/finalization_state.json"
            temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(status,indent=2)+"\n");temporary.replace(path)
            print(json.dumps(status),flush=True)
            if full:
                return
        time.sleep(20)


if __name__ == "__main__":
    main()
