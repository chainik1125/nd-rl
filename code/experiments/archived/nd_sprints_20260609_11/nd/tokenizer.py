"""Fixed vocabulary tokenizer for ND proof sequences."""

MAX_LINES = 64

FORMULA_TOKENS = ['P', 'Q', 'R', 'S', 'F', '(', ')', '~', '&', 'v', '>']
STRUCT_TOKENS = ['THM', ',', 'SEQ', 'PRF', 'QED', '|', ':', ';']
RULE_TOKENS = ['PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI',
               'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN']
NUM_TOKENS = [f'N{i}' for i in range(1, MAX_LINES + 1)]

VOCAB = ['<pad>'] + FORMULA_TOKENS + STRUCT_TOKENS + RULE_TOKENS + NUM_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for i, t in enumerate(VOCAB)}
PAD = 0
VOCAB_SIZE = len(VOCAB)  # 88


def encode(tokens):
    return [STOI[t] for t in tokens]


def decode(ids):
    return [ITOS[i] for i in ids if i != PAD]
