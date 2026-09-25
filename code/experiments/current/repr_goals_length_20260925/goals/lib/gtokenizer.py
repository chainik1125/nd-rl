"""The 'abs' tokenizer of frozen/lib/tokenizer.py plus one token, GOAL, for self-written goal annotations.

Checkpoint tok_mode 'abs_goals'. Vocabulary: <pad>, the 19 symbols, the 15 rules, GOAL, then N1..N64, so the
line-index tokens stay the last MAXN ids (every `x >= tok.ref0` test of the frozen code, the 039 pretrain recipe's
shift augmentation and the seed's capped index shift, keeps meaning "a line-index token"). Vocab 100 (abs: 99).
Encoding and decoding are verbatim, as in abs mode; annotation groups contain no N<k> tokens, so the index shift
never touches them.
"""
import os, sys

_LIB = '/scratch/cowifornia/nd-rl/ab3/frozen/lib'
if _LIB not in sys.path:      # appended: a harness copy's own lib/ (same files) stays first
    sys.path.append(_LIB)
import tokenizer as _ftok  # noqa: E402
from tokenizer import SYMS, RULES, MAXN  # noqa: E402,F401

GOAL = 'GOAL'
MODE = 'abs_goals'


class GoalTokenizer(_ftok.Tokenizer):
    def __init__(self, mode=MODE):
        assert mode == MODE, mode
        self.mode = MODE
        self.itos = ['<pad>'] + SYMS + RULES + [GOAL]
        self.ref0 = len(self.itos)
        self.itos += [f'N{k}' for k in range(1, MAXN + 1)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad = 0
        self.eos = self.stoi['QED']
        self.goal = self.stoi[GOAL]
        self.shift = True

    def encode_proof(self, body):
        """Verbatim (abs) encoding: every N<k> -> its index token, everything else by symbol (QED included)."""
        out = []
        for t in body.split():
            if t.startswith('N') and t[1:].isdigit():
                out.append(self._ref(int(t[1:])))
            else:
                out.append(self.stoi[t])
        return out

    def decode(self, ids):
        toks = []
        for x in ids:
            if x == self.pad:
                continue
            s = self.itos[x]
            toks.append(s)
            if s == 'QED':
                break
        return ' '.join(toks)


def make_tokenizer(mode):
    """Tokenizer for a checkpoint's tok_mode ('abs' -> the frozen one, 'abs_goals' -> GoalTokenizer)."""
    if mode == MODE:
        return GoalTokenizer()
    return _ftok.Tokenizer(mode)
