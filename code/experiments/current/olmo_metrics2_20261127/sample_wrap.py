#!/usr/bin/env python3
"""Wrapper around Dan's sampler (olmo_ifladder/sample.py) that leaves his file unedited.

    python sample_wrap.py [--dan-root D] [--temperature F] [--top-p F] [--patch mod.py] [--sidecar out.json] -- <Dan's argv>

Dan's module constants TEMPERATURE and TOP_P are read at call time (in `submit` for SamplingParams and in `main` for his
meta dict), so setting them on the imported module changes both the sampling and his own meta record. After the run the
wrapper asserts that the last line of his --meta file carries the overridden values.

--patch <python file>: a module with `apply(args)`; called immediately before vLLM's LLM() is constructed (the wrapper
replaces `vllm.LLM` with a factory, because Dan imports `from vllm import LLM` inside main() at call time). `args` is a
SimpleNamespace(dan_argv, llm_kwargs, temperature, top_p, dan_root, sidecar, patch); llm_kwargs are the kwargs Dan passes
to LLM() (read only: changes are not fed back).
The sidecar records temperature, top_p, patch path and sha256, sha256 of Dan's sample.py, his argv and UTC start/end.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meta_path_of(dan_argv):
    for i, a in enumerate(dan_argv):
        if a == "--meta" and i + 1 < len(dan_argv):
            return dan_argv[i + 1]
        if a.startswith("--meta="):
            return a.split("=", 1)[1]
    return None


def count_lines(p):
    try:
        with open(p) as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def last_line(p):
    last = None
    with open(p) as f:
        for ln in f:
            if ln.strip():
                last = ln
    return json.loads(last) if last else None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        raise SystemExit("usage: sample_wrap.py [wrapper flags] -- <Dan's argv>")
    i = argv.index("--")
    own, dan_argv = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dan-root", default="/workspace/nd-takehome")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--sidecar", default=None)
    a = ap.parse_args(own)

    sys.dont_write_bytecode = True  # never write __pycache__ into Dan's tree
    dan_file = Path(a.dan_root) / "olmo_ifladder" / "sample.py"
    dan_sha = sha256_file(dan_file)
    dan = load_module(dan_file, "dan_olmo_ifladder_sample")
    if a.temperature is not None:
        dan.TEMPERATURE = a.temperature
    if a.top_p is not None:
        dan.TOP_P = a.top_p
    want_t, want_p = dan.TEMPERATURE, dan.TOP_P

    side = {
        "temperature": want_t,
        "top_p": want_p,
        "temperature_overridden": a.temperature is not None,
        "top_p_overridden": a.top_p is not None,
        "patch": None,
        "patch_sha256": None,
        "dan_sample_py": str(dan_file),
        "dan_sample_py_sha256": dan_sha,
        "dan_argv": dan_argv,
        "utc_start": utc(),
        "utc_end": None,
        "status": "started",
    }

    def write_side():
        if a.sidecar:
            Path(a.sidecar).parent.mkdir(parents=True, exist_ok=True)
            tmp = a.sidecar + ".tmp"
            with open(tmp, "w") as f:
                json.dump(side, f, indent=1)
            Path(tmp).replace(a.sidecar)

    patch = None
    if a.patch:
        side["patch"] = str(Path(a.patch).resolve())
        side["patch_sha256"] = sha256_file(a.patch)
        patch = load_module(a.patch, "m2_patch")
        if not callable(getattr(patch, "apply", None)):
            raise SystemExit(f"--patch {a.patch}: no apply(args) function")
    write_side()

    if patch is not None:
        import vllm  # Dan imports `from vllm import LLM` inside main(), after this point

        real_llm = vllm.LLM
        pargs = types.SimpleNamespace(
            dan_argv=dan_argv,
            llm_kwargs=None,
            temperature=want_t,
            top_p=want_p,
            dan_root=a.dan_root,
            sidecar=a.sidecar,
            patch=a.patch,
        )

        def llm_factory(*args, **kw):
            pargs.llm_kwargs = dict(kw)
            patch.apply(pargs)
            side["patch_applied_utc"] = utc()
            write_side()
            return real_llm(*args, **kw)

        vllm.LLM = llm_factory

    mp = meta_path_of(dan_argv)
    n_before = count_lines(mp) if mp else 0
    sys.argv = [str(dan_file)] + dan_argv
    rc = dan.main()
    side["utc_end"] = utc()
    side["dan_main_rc"] = rc
    # his own meta must record the values that were actually used
    if mp is None:
        side["meta_check"] = "skipped: no --meta in Dan's argv"
    elif count_lines(mp) <= n_before:
        side["meta_check"] = "skipped: Dan's main wrote no new meta line (nothing to run)"
    else:
        m = last_line(mp)
        if m.get("temperature") != want_t or m.get("top_p") != want_p:
            side["status"] = "META_MISMATCH"
            write_side()
            raise AssertionError(
                f"meta last line has T={m.get('temperature')} top_p={m.get('top_p')}, "
                f"expected {want_t} / {want_p}"
            )
        side["meta_check"] = f"ok: last meta line has temperature={want_t} top_p={want_p}"
    if dan_sha != sha256_file(dan_file):
        side["status"] = "DAN_FILE_CHANGED"
        write_side()
        raise AssertionError("Dan's sample.py changed during the run")
    side["status"] = "done"
    write_side()
    return rc


if __name__ == "__main__":
    sys.exit(main())
