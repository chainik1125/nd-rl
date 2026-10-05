"""One symbol per token. Two reference schemes, chosen at training time and stored in the ckpt.

  mode='rel': line numbers are dropped from the text; a citation N<j> on line i becomes B<i-j>
              ("k lines back"). decode() regenerates N1, N2, ... and resolves B<k>.
  mode='abs': the text is kept verbatim (N<i> tokens). Train-time augmentation shifts every
              index in a proof by a random offset so that N1..N<MAXN> are all trained tokens
              (the verifier accepts any starting index).
"""
import random

SYMS = ['THM', ',', 'SEQ', 'PRF', 'QED', '|', ':', ';', '(', ')', '~', '&', 'v', '>', 'P', 'Q', 'R', 'S', 'F']
RULES = ['PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN']
MAXN = 64


class Tokenizer:
    def __init__(self, mode='rel'):
        assert mode in ('rel', 'abs')
        self.mode = mode
        self.itos = ['<pad>'] + SYMS + RULES
        if mode == 'rel':
            self.ref0 = len(self.itos)
            self.itos += [f'B{k}' for k in range(1, MAXN + 1)]
        else:
            self.ref0 = len(self.itos)
            self.itos += [f'N{k}' for k in range(1, MAXN + 1)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad = 0
        self.eos = self.stoi['QED']
        self.shift = True

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode_prompt(self, prompt):
        return [self.stoi[t] for t in prompt.split()]

    def encode_proof(self, body):
        """body: 'N1 ... ; ... QED' -> ids (QED included)."""
        toks = body.split()
        out = []
        i = 0
        cur = None
        while i < len(toks):
            t = toks[i]
            if t == 'QED':
                out.append(self.eos)
                i += 1
                continue
            if t.startswith('N') and t[1:].isdigit():
                n = int(t[1:])
                if cur is None or toks[i - 1] == ';':
                    cur = n              # line index
                    if self.mode == 'abs':
                        out.append(self._ref(n))
                else:                    # citation
                    if self.mode == 'rel':
                        out.append(self._ref(cur - n))
                    else:
                        out.append(self._ref(n))
            else:
                out.append(self.stoi[t])
            i += 1
        return out

    def _ref(self, k):
        if not (1 <= k <= MAXN):
            raise ValueError(f'ref {k} out of range')
        return self.ref0 + k - 1

    def decode(self, ids):
        """ids (proof part, may include QED and trailing pads) -> body string in spec format."""
        toks = []
        cur = 0
        at_line_start = True
        for x in ids:
            if x == self.pad:
                continue
            s = self.itos[x]
            if s == 'QED':
                toks.append('QED')
                break
            if self.mode == 'rel':
                if at_line_start:
                    cur += 1
                    toks.append(f'N{cur}')
                    at_line_start = False
                if x >= self.ref0:
                    k = x - self.ref0 + 1
                    toks.append(f'N{cur - k}')
                else:
                    toks.append(s)
            else:
                toks.append(s)
            if s == ';':
                at_line_start = True
        return ' '.join(toks)

    def shift_abs(self, ids, rng):
        """abs mode augmentation: add a random offset to every N<k> id in the proof."""
        if self.mode != 'abs' or not self.shift:
            return ids
        mx = max((x - self.ref0 + 1 for x in ids if x >= self.ref0), default=0)
        if mx == 0:
            return ids
        s = rng.randint(0, MAXN - mx)
        return [x + s if x >= self.ref0 else x for x in ids]


def make_tokenizer(mode):
    if mode.startswith('lean'):
        from lean_tok import LeanTokenizer
        return LeanTokenizer(mode)
    return Tokenizer(mode)


def selftest():
    import json, sys
    from nd_verify import verify_text
    rs = [json.loads(l) for l in open(sys.argv[1])]
    for mode in ('rel', 'abs'):
        tk = Tokenizer(mode)
        rng = random.Random(0)
        bad = 0
        for r in rs:
            ids = tk.encode_proof(r['proof'])
            d = tk.decode(ids)
            if d != r['proof']:
                bad += 1
                if bad < 3:
                    print(mode, 'MISMATCH', r['proof'], '||', d)
            if mode == 'abs':
                sh = tk.shift_abs(ids, rng)
                ok, reason, nl = verify_text(r['prompt'] + ' ' + tk.decode(sh))
                if not ok:
                    bad += 1
                    print('shift fail', reason)
        print(mode, 'vocab', tk.vocab_size, 'roundtrip mismatches', bad, '/', len(rs))


if __name__ == '__main__':
    selftest()
