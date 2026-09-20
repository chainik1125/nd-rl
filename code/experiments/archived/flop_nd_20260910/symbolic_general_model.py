"""Order-aware fixed symbolic features and configurable shared MLP scorer.

No source-proof position, reference continuation, search oracle, or final-set
information enters features. Atom counts expose syntactic content; they are not
claimed to be a learned formula representation. Deeper order is summarized by a
fixed signed path sketch, while root and immediate arguments are explicit.
"""
from functools import lru_cache
import hashlib
import math
import numpy as np
import torch
from torch import nn

from symbolic_general_env import OPS, RULES, formula_size, context_prefix

FORMULA_DIM = 62
STATE_DIM = 5 * FORMULA_DIM + 12
ACTION_DIM = len(RULES) + FORMULA_DIM + 10 + 5 * (FORMULA_DIM + 4)
FEATURE_VERSION = "nd-ordered-root-children-counts-pathsketch-refroles-v1"


def onehot_op(f):
    out = np.zeros(len(OPS), np.float32)
    if f:
        out[OPS.index(f[0])] = 1.
    return out


@lru_cache(maxsize=100000)
def formula_features(f):
    if f is None:
        return np.zeros(FORMULA_DIM, np.float32)
    children = [x for x in f[1:] if isinstance(x, tuple)]
    direct = np.concatenate([onehot_op(f), onehot_op(children[0] if children else None),
                             onehot_op(children[1] if len(children) > 1 else None)])
    atoms, operators, sketch = np.zeros(4), np.zeros(6), np.zeros(32)
    max_depth = 0
    def visit(node, path):
        nonlocal max_depth
        max_depth = max(max_depth, len(path))
        operators[OPS.index(node[0])] += 1
        if node[0] == "atom":
            atoms["PQRS".index(node[1])] += 1
        symbol = node[1] if node[0] == "atom" else node[0]
        key = ("/".join(map(str, path[-3:])) + ":" + symbol).encode()
        digest = hashlib.blake2b(key, digest_size=2).digest()
        sketch[digest[0] % 32] += 1 if digest[1] % 2 else -1
        for i, child in enumerate(x for x in node[1:] if isinstance(x, tuple)):
            visit(child, path + (i,))
    visit(f, ())
    n = formula_size(f)
    sizes = [math.log1p(n) / (1 + math.log1p(n)), max_depth / (1 + max_depth)]
    output = np.concatenate([direct, atoms / n, operators / n, sizes, sketch / math.sqrt(n)]).astype(np.float32)
    assert output.shape == (FORMULA_DIM,)
    output.flags.writeable = False
    return output


def features(env, actions):
    zero = formula_features(None)
    all_features = [zero] + [formula_features(line.formula) for line in env.lines]
    visible = env.visible(env.stack)
    visible_features = np.stack([all_features[i] for i, _ in visible]) if visible else np.zeros((1, FORMULA_DIM), np.float32)
    current_assumption = all_features[env.stack[-1]] if env.stack else zero
    last = all_features[-1] if env.lines else zero
    n, depth = len(env.lines), len(env.stack)
    closed = [b for b in env.boxes.values() if b.start not in env.stack]
    scalars = np.asarray([n / 24, depth / 4, len(visible) / 24, len(closed) / 24,
        len(env.base) / 32, math.log1p(len(actions)) / 6, len(env.premises) / 4,
        sum(l.formula == env.goal for _, l in visible) / max(1, len(visible)),
        float(last is not zero), float(bool(env.stack)),
        sum(l.formula[0] == "or" for _, l in visible) / max(1, len(visible)),
        sum(l.formula[0] == "imp" for _, l in visible) / max(1, len(visible))], np.float32)
    state = np.concatenate([formula_features(env.goal), visible_features.mean(0), visible_features.max(0),
                            current_assumption, last, scalars])
    rows = []
    for action in actions:
        rule = np.zeros(len(RULES), np.float32)
        rule[RULES.index(action.rule)] = 1
        destination = env.stack[:action.depth - 1] if action.rule == "AS" else env.stack[:action.depth]
        available_formulas = {line.formula for _, line in env.visible(destination)}
        action_scalars = np.asarray([action.depth / 4, (action.depth - depth) / 4,
            float(action.rule == "AS"), max(0, depth - action.depth) / 4,
            float(action.formula == env.goal), float(action.formula in available_formulas),
            float(bool(env.lines) and action.formula == env.lines[-1].formula),
            float(action.formula in env.base), len(action.refs) / 5,
            float(not destination and action.rule != "AS")], np.float32)
        refs = []
        for k in range(5):
            if k >= len(action.refs):
                refs.append(np.zeros(FORMULA_DIM + 4, np.float32))
                continue
            idx = action.refs[k]
            line = env.lines[idx - 1]
            rel = np.asarray([(n - idx) / max(1, n), len(line.context) / 4,
                float(context_prefix(line.context, destination)), float(idx in env.boxes)], np.float32)
            refs.append(np.concatenate([all_features[idx], rel]))
        rows.append(np.concatenate([rule, all_features[0] if action.formula is None else formula_features(action.formula),
                                    action_scalars] + refs))
    matrix = np.stack(rows) if rows else np.zeros((0, ACTION_DIM), np.float32)
    assert matrix.shape == (len(actions), ACTION_DIM) and state.shape == (STATE_DIM,)
    return matrix, state


class Policy(nn.Module):
    def __init__(self, width=500, layers=2):
        super().__init__()
        self.width, self.layers = width, layers
        def mlp(inputs):
            modules = []
            for _ in range(layers):
                modules += [nn.Linear(inputs, width), nn.Tanh()]
                inputs = width
            modules.append(nn.Linear(inputs, 1))
            return nn.Sequential(*modules)
        self.scorer = mlp(ACTION_DIM + STATE_DIM)
        self.critic = mlp(STATE_DIM)

    def forward(self, actions, state, mask=None):
        expanded = state.unsqueeze(-2).expand(*actions.shape[:-1], state.shape[-1])
        logits = self.scorer(torch.cat([actions, expanded], -1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf)
        return logits, self.critic(state).squeeze(-1)


def pack(transitions):
    count, max_actions = len(transitions), max(len(t["actions"]) for t in transitions)
    actions = torch.zeros(count, max_actions, ACTION_DIM)
    mask = torch.zeros(count, max_actions, dtype=torch.bool)
    for i, t in enumerate(transitions):
        length = len(t["actions"])
        actions[i, :length] = torch.from_numpy(t["actions"])
        mask[i, :length] = True
    states = torch.from_numpy(np.stack([t["state"] for t in transitions]))
    return actions, states, mask
