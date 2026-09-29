"""Read-only endpoint fit checks and trace descriptions for the detour diagnostic."""
from collections import Counter
import json
from pathlib import Path
import torch
import common
from nd_environment import strict_complete
from symbolic_general_env import Environment
from symbolic_general_model import Policy, features


@torch.no_grad()
def logits_for(model, env, actions):
    af, sf = features(env, actions)
    return model(torch.from_numpy(af), torch.from_numpy(sf))[0]


def main():
    torch.set_num_threads(4)
    root = common.ROOT
    endpoints_path = root / 'configs/detour_endpoints_v1.json'
    endpoints = json.loads(endpoints_path.read_text())['endpoints']
    reports, inputs = {}, {str(endpoints_path): common.sha(endpoints_path)}
    for endpoint in endpoints:
        path = Path(endpoint['checkpoint'])
        payload = torch.load(path, map_location='cpu', weights_only=False)
        config = payload['config']
        for name in ('symbolic_general_env.py', 'symbolic_general_model.py'):
            assert common.sha(Path(__file__).with_name(name)) == config['code_hashes'][name]
        assert common.sha(config['sources']) == config['source_sha256']
        model = Policy(config['width'], config['layers'])
        model.load_state_dict(payload['state_dict'])
        model.eval()
        sources = common.read_jsonl(config['sources'])
        correct, losses, examples = 0, [], []
        for row in sources:
            env = Environment(row['prompt'], max_lines=6, max_scope=4)
            for desired in env.source_actions(row):
                actions = env.actions()
                logits = logits_for(model, env, actions)
                index = actions.index(desired)
                correct += int(logits.argmax().item() == index)
                losses.append(-logits.log_softmax(-1)[index].item())
                env.step(desired)
            env = Environment(row['prompt'], max_lines=6, max_scope=4)
            rules = []
            while not env.success():
                actions = env.actions()
                if not actions:
                    break
                action = actions[logits_for(model, env, actions).argmax().item()]
                rules.append(action.rule)
                reward, done = env.step(action)
                if done:
                    break
            verdict = strict_complete(row['prompt'], env.body())
            assert bool(verdict['ok']) == env.success()
            examples.append({'source_id': row['source_id'], 'prompt': row['prompt'], 'demonstration': row['body'],
                'greedy_body': env.body(), 'success': verdict['ok'], 'greedy_raw_lines': verdict.get('n_lines'),
                'greedy_rules': rules, 'greedy_pruned_body': env.pruned_body() if env.success() else None})
        trace_path = root / 'results' / (endpoint['name'].removesuffix('_v1') + '_structured_development_greedy_v1') / 'symbolic_general_attempts.jsonl'
        counts, rule_counts, prefixes = Counter(), Counter(), Counter()
        for row in common.read_jsonl(trace_path):
            rules = [a['rule'] for a in row['action_trace']]
            counts['attempts'] += 1
            counts['successes'] += int(row['success'])
            counts['attempts_with_reiteration'] += int('R' in rules)
            counts['attempts_using_only_reiteration'] += int(bool(rules) and all(r == 'R' for r in rules))
            counts['attempts_with_assumptions'] += int('AS' in rules)
            rule_counts.update(rules)
            prefixes[','.join(rules[:8])] += 1
        reports[endpoint['name']] = {'teacher_forced_states': len(losses), 'teacher_forced_correct_actions': correct,
            'teacher_forced_accuracy': correct / len(losses), 'teacher_forced_mean_nll': sum(losses) / len(losses),
            'greedy_source_successes': sum(r['success'] for r in examples), 'source_examples': examples,
            'development_greedy_counts': dict(counts), 'development_greedy_rule_counts': dict(rule_counts),
            'most_common_first_eight_rules': prefixes.most_common(8)}
        for item in (path, Path(config['sources']), trace_path):
            inputs[str(item)] = common.sha(item)
    for name in ('probe_detour_fit.py', 'symbolic_general_env.py', 'symbolic_general_model.py', 'nd_environment.py'):
        path = Path(__file__).with_name(name)
        inputs[str(path.resolve())] = common.sha(path)
    report = {'status': 'passed', 'endpoints': reports, 'input_sha256': inputs,
        'scope': 'Read-only post hoc mechanism diagnostic. Exact final checkpoint source accuracy and original-theorem greedy source attempts; descriptive development rule counts, not a separate causal mediation test.'}
    common.write_json(root / 'reports/detour_training_fit.json', report)
    selected = ['detour_noisy_bc_s4501_v1', 'detour_noisy_reset_s4501_v1']
    parts = ['# How the two methods use a distracted seed proof\n',
        'This illustration uses the first supplied seed theorem and training seed 4501, selected by fixed input order. Both models received the same six-line demonstration. The source-level and development aggregates are in `detour_training_fit.json`; all development comparisons remain those in the frozen diagnostic.\n']
    for name in selected:
        row = reports[name]['source_examples'][0]
        parts += [f'## {name}\n', f'Accepted from the original theorem: {row["success"]}.\n',
                  '```text\n' + row['prompt'] + '\n' + row['greedy_body'].replace('; ', ';\n') + '\n```\n']
    parts.append('These are learned choices within the general legal-action environment. Removing irrelevant reiterations is a narrower result than discovering a new mathematical strategy. The diagnostic used different PPO and cloning training budgets.\n')
    (root / 'reports/detour_source_witness.md').write_text('\n'.join(parts))
    print(json.dumps({k: {f: v[f] for f in ('teacher_forced_accuracy', 'greedy_source_successes', 'development_greedy_counts')}
                      for k, v in reports.items()}))


if __name__ == '__main__':
    main()
