"""Launcher checks without the real seed: arm files from a stand-in seed differ from it exactly in the knob lines
(max_len_train and max_proof_lines_train, both set to the arm's value; max_len_train only with proof_cap=False).

    python tests/test_launch.py
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import launch  # noqa: E402

launch.LOGS = tempfile.mkdtemp()   # keep test log lines out of logs/launch.log


def test_make_arms():
    seed = os.path.join(HERE, 'standin', 'rl_seed.py')
    with tempfile.TemporaryDirectory() as d:
        info = launch.make_arms(seed, root=d)
        src = open(seed).read().splitlines()
        for arm, v in launch.ARMS.items():
            out = open(info['arms'][arm]['file']).read().splitlines()
            diff = [i for i, (x, y) in enumerate(zip(src, out)) if x != y]
            assert len(src) == len(out) and len(diff) == (0 if v == 11 else 1) + 1, (arm, diff)
            assert re.search(rf'max_len_train={v},', '\n'.join(out)), arm
            assert re.search(rf'max_proof_lines_train={v},', '\n'.join(out)), arm
            assert info['arms'][arm]['max_proof_lines_train'] == v
            subprocess.run([sys.executable, '-m', 'py_compile', info['arms'][arm]['file']], check=True)
    with tempfile.TemporaryDirectory() as d:                 # --no-proof-cap: the old one-knob arms
        info = launch.make_arms(seed, root=d, proof_cap=False)
        for arm, v in launch.ARMS.items():
            out = open(info['arms'][arm]['file']).read().splitlines()
            assert sum(x != y for x, y in zip(src, out)) == (0 if v == 11 else 1), arm
            assert 'max_proof_lines_train=None,' in '\n'.join(out)
    # a seed without the knob, or with two settings, is refused
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, 'rl.py')
        open(bad, 'w').write('CONFIG = dict(a=1)\n')
        try:
            launch.make_arms(bad, root=d)
            raise AssertionError('accepted a seed without the knob')
        except SystemExit:
            pass
        open(bad, 'w').write('CONFIG = dict(max_len_train=None)\nprint(CONFIG["max_len_train"])\n')
        try:
            launch.make_arms(bad, root=d)                       # the proof cap is asked for but missing
            raise AssertionError('accepted a seed without max_proof_lines_train')
        except SystemExit:
            pass
        open(bad, 'w').write('CONFIG = dict(max_len_train=None, max_proof_lines_train=None)\n'
                             'print(CONFIG["max_len_train"], CONFIG.get("max_proof_lines_train"))\n')
        try:
            launch.make_arms(bad, root=d)                       # both knobs on one line: refused
            raise AssertionError('accepted two knobs on one line')
        except SystemExit:
            pass
        open(bad, 'w').write('"""docstring: max_len_train=7"""\nCONFIG = dict(max_len_train=None, k=3)\nX = dict(max_len_train=9)\nprint(CONFIG["max_len_train"])\n')
        info = launch.make_arms(bad, root=d, proof_cap=False)   # only CONFIG's entry changes (not the docstring, not X)
        out = open(info['arms']['A']['file']).read()
        assert 'CONFIG = dict(max_len_train=7, k=3)' in out and 'docstring: max_len_train=7' in out and 'X = dict(max_len_train=9)' in out, out
        open(bad, 'w').write("CONFIG = dict(max_len_train=7, max_len_train2=3)\n")
        try:
            launch.make_arms(bad, root=d, proof_cap=False)
            raise AssertionError('accepted a seed that never reads the knob')
        except SystemExit:
            pass
        open(bad, 'w').write("CONFIG = {'max_len_train': 6, 'k': 3}\nprint(CONFIG['max_len_train'])\n")
        info = launch.make_arms(bad, root=d, proof_cap=False)
        assert "'max_len_train': 9," in open(info['arms']['B']['file']).read()


def test_default_flops_parse():
    v = launch.default_flops()
    assert v is None or isinstance(v, (int, float, str)), v


if __name__ == '__main__':
    test_make_arms()
    test_default_flops_parse()
    print('launch tests ok; harness DEFAULT_FLOPS =', launch.default_flops())
