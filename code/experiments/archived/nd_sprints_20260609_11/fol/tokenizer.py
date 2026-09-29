"""Fixed vocabulary tokenizer for first-order ND proofs."""
MAX_LINES = 64
PRED_TOKENS = ['P', 'Q', 'R', 'S']
VAR_TOKENS = ['x', 'y', 'z', 'w']
CONST_TOKENS = ['a', 'b', 'c', 'd', 'e']
LOGIC_TOKENS = ['F', '(', ')', '~', '&', 'v', '>', 'A', 'E']
STRUCT_TOKENS = ['THM', ',', 'SEQ', 'PRF', 'QED', '|', ':', ';']
RULE_TOKENS = ['PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI',
               'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN',
               'ALLI', 'ALLE', 'EXI', 'EXE']
NUM_TOKENS = [f'N{i}' for i in range(1, MAX_LINES + 1)]

VOCAB = (['<pad>'] + PRED_TOKENS + VAR_TOKENS + CONST_TOKENS + LOGIC_TOKENS
         + STRUCT_TOKENS + RULE_TOKENS + NUM_TOKENS)
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for i, t in enumerate(VOCAB)}
PAD = 0
VOCAB_SIZE = len(VOCAB)


def encode(tokens):
    return [STOI[t] for t in tokens]


def decode(ids):
    return [ITOS[i] for i in ids if i != PAD]
