"""Check exact token costs of the structured projection reference certificates."""
from collections import Counter
import json
from pathlib import Path
import common
from nd_environment import Tokenizer
import plot_results


def response_tokens(depth, compound=False):
    # Every projection cites the immediately previous line. The relative-unary
    # citation therefore has constant cost; the copied formulas shrink by one
    # conjunction edge per inference. FINAL and QED are included.
    return 4 * depth * depth + 12 * depth + 7 if compound else 2 * depth * depth + 10 * depth + 7


def prompt_tokens(depth, compound=False):
    return (8 if compound else 4) * depth + 7


def main():
    path = common.ROOT / 'data/structured_v1/audit/development.jsonl'
    rows = common.read_jsonl(path)
    tok = Tokenizer()
    counts, examples = Counter(), []
    for row in rows:
        if row['family'] not in ('projection_atomic', 'projection_compound'):
            continue
        depth, compound = row['n_lines'] - 1, row['family'] == 'projection_compound'
        actual = 1 + len(tok.encode_record(row['prompt'] + ' ' + row['body'])) - len(tok.encode_prompt(row['prompt']))
        assert actual == row['reference_tokens'] == response_tokens(depth, compound)
        assert len(tok.encode_prompt(row['prompt'])) + 1 == row['prompt_tokens'] == prompt_tokens(depth, compound)
        counts[row['family']] += 1
        examples.append({'source_id': row['source_id'], 'family': row['family'], 'lines': row['n_lines'],
                         'response_tokens': actual, 'prompt_tokens': row['prompt_tokens']})
    table = []
    for length in (6, 7, 9, 13, 17, 25, 64, 100, 1000):
        depth = length - 1
        table.append({'lines': length, 'symbolic_inferences_after_premise': depth,
                      'atomic_response_tokens': response_tokens(depth), 'compound_response_tokens': response_tokens(depth, True),
                      'atomic_prompt_tokens': prompt_tokens(depth), 'compound_prompt_tokens': prompt_tokens(depth, True),
                      'basis': 'Validated challenge reference length' if length in (7, 9, 13, 17, 25) else 'Exact formula for the declared projection schema'})
    limits = {}
    for compound, family in ((False, 'atomic'), (True, 'compound')):
        limits[family] = {'response_cap_' + str(b): max(d + 1 for d in range(1000) if response_tokens(d, compound) <= b)
                          for b in (512, 1024)}
        limits[family]['prompt_plus_response_context_4096'] = max(d + 1 for d in range(1000)
            if prompt_tokens(d, compound) + response_tokens(d, compound) <= 4096)
    report = {'status': 'passed', 'depth_definition': 'd = number of conjunction eliminations; total reference lines = d+1',
        'formulas': {'atomic_response': '2*d^2 + 10*d + 7', 'compound_response': '4*d^2 + 12*d + 7',
                     'atomic_prompt': '4*d + 7', 'compound_prompt': '8*d + 7'},
        'checked_references': dict(counts), 'table': table, 'reference_line_limits': limits, 'examples': examples,
        'input_sha256': {str(path): common.sha(path), str(Path(__file__).resolve()): common.sha(__file__)},
        'limits': ['These are token costs of the specified explicit certificates, not unrestricted proof lower bounds.',
                   'The current 7/9/13-line LM challenge admits every reference within its 1024-response-token cap; serialization does not explain its low success by itself.',
                   'A legal-action policy still incurs symbolic feature/candidate computation, which is not counted as language-model output tokens.']}
    out = common.ROOT / 'reports/serialization_scaling.json'
    common.write_json(out, report)
    import matplotlib.pyplot as plt
    plot_results.style()
    fig, ax = plt.subplots(figsize=(7, 4))
    lengths = list(range(6, 26))
    for compound, label, color in ((False, 'Atomic distractors', '#0072B2'), (True, 'Compound distractors', '#E69F00')):
        ax.plot(lengths, [response_tokens(n - 1, compound) for n in lengths], color=color, label=label)
        ax.scatter([7, 9, 13, 17, 25], [response_tokens(n - 1, compound) for n in (7, 9, 13, 17, 25)], color=color, s=25)
    for cap in (512, 1024):
        ax.axhline(cap, color='#777777', linestyle='--', linewidth=.8)
        ax.text(6.1, cap + 35, f'{cap:,} response tokens', color='#555555', fontsize=9)
    ax.set(xlabel='Explicit natural-deduction certificate lines', ylabel='Generated tokens, including FINAL and QED',
           xlim=(6, 25.5), ylim=(0, 2800), title='Copying formulas makes the projection certificate grow quadratically')
    ax.set_xticks([6, 7, 9, 13, 17, 25])
    ax.legend(frameon=False)
    fig.tight_layout()
    plot_results.save(fig, common.ROOT / 'figures/serialization_scaling_v1', [out, Path(__file__)],
        'Exact relative-unary serialization of the structured projection reference proofs. Each new inference writes the remaining conjunction formula again; d eliminations cost 2d²+10d+7 response tokens with atomic distractors or 4d²+12d+7 with compound distractors. Dots summarize all 160 checked development references. The dashed lines show the LM response caps. This counts a particular certificate schema, not a minimum over all natural-deduction proofs.')
    print(json.dumps({'status': 'passed', 'checked': dict(counts), 'reference_line_limits': limits}))


if __name__ == '__main__':
    main()
