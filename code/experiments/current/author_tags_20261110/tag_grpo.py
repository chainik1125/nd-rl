"""GRPO under one author tag: Leon's `code.thinking.grpo` (snapshot f35d3daf_danbase), wrapped, not edited.

    python tag_grpo.py --tag {A,B,none} [--variant pNN] -- <code.thinking.grpo arguments>

Sets the process tag (`AUTHOR_TAG`, so every state of every prompt carries it), installs the tag tokenizer
(`tag_core.install_thinking`), installs the spy (`tag_core.spy_thinking`: every decode wave and the canary must carry the
tag right before <act>, else the run stops with TagError), refuses a tagged run on a checkpoint without the tag
vocabulary, refuses an --init whose Stage-1 corpus variant (`extra['author_tags']['variant']`, None for the default
corpus) is not `--variant` (default: the default corpus), writes `<out>/author_tag.json` (the tag, the init, its mode
and variant, and at exit the number of states checked per tag), then runs `grpo.main(argv)` and exits with its code
(0 done, 4 non-finite stop, ...).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", required=True, choices=T.READ_TAGS)
    ap.add_argument(
        "--variant", choices=sorted(T.VARIANTS), default=None, help="the --init's corpus variant"
    )
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--init", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--resume", default=None)
    ga, _ = g.parse_known_args(rest)
    mode = T.ckpt_mode(ga.resume or ga.init)
    if a.tag != "none" and mode != T.MODE:
        raise SystemExit(f"tag {a.tag} on a {mode} checkpoint: it has no tag vocabulary")
    init_variant = T.ckpt_meta(ga.init)["variant"]
    if init_variant != a.variant:
        raise SystemExit(
            f"--init {ga.init} trained on variant {init_variant!r}; this run is for {a.variant!r}"
        )
    T.set_process_tag(a.tag)
    T.install_thinking()
    T.spy_thinking(a.tag)
    from code.thinking import grpo  # type: ignore[import-not-found]

    out = Path(ga.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"tag": a.tag, "init": ga.init, "resume": ga.resume, "init_mode": mode, "init_variant": init_variant,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip

    def write() -> None:
        (out / "author_tag.json").write_text(
            json.dumps({**info, "spy": T.SPY}, indent=1, default=str)
        )

    write()
    atexit.register(write)
    print(f"tag_grpo: tag {a.tag}, init mode {mode}", flush=True)
    return grpo.main(rest)


if __name__ == "__main__":
    sys.exit(main())
