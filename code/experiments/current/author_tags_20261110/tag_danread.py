"""Dan's held-out read (knockout_danbase_20261104/danread.py, unchanged) under one author tag.

    python tag_danread.py --tag {A,B,none} -- <danread.py arguments>

Sets `AUTHOR_TAG`, installs the tag tokenizer in Dan's tree and the spy on his sampler (every decoder input must carry
the tag right before <act>), then runs danread's `main()`. The result JSON gains `author_tag` and the spy's counts.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, choices=T.READ_TAGS)
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="CPU smoke only: danread loads on cuda; map it to the CPU",
    )
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--out", required=True)
    ga, _ = g.parse_known_args(rest)
    mode = T.ckpt_mode(ga.ckpt)
    if a.tag != "none" and mode != T.MODE:
        raise SystemExit(f"tag {a.tag} on a {mode} checkpoint: it has no tag vocabulary")
    T.set_process_tag(a.tag)
    T.install_dan()
    T.spy_dan(a.tag)
    if a.cpu:
        import model as dmodel  # type: ignore[import-not-found]
        import torch

        torch.set_num_threads(8)
        orig = dmodel.load_ckpt
        dmodel.load_ckpt = lambda path, device="cpu": orig(path, "cpu")
        os.environ.pop("DANREAD_MEM_FRAC", None)
    spec = importlib.util.spec_from_file_location("ko_danread", T.KO / "danread.py")
    dr = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dr)
    sys.argv = ["danread.py", *rest]
    dr.main()
    p = Path(ga.out)
    res = json.loads(p.read_text())
    res.update(author_tag=a.tag, tag_spy={"states": T.SPY["states"], "by_tag": T.SPY["by_tag"]})
    p.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
