"""Post-hoc illustration from existing audited final LM outputs; no new inference."""
from pathlib import Path
import json
import common
from nd_environment import strict_complete
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof

R=common.ROOT
analysis_path=R/'reports/analysis_primary_final_greedy_v1.json'
sampled_path=R/'reports/analysis_primary_final_sample4_v1.json'
d=json.loads(analysis_path.read_text());sampled=json.loads(sampled_path.read_text())
assert d['status']==sampled['status']=='completed'
endpoint='main_relative_full_seed2_v1';retention='retention_seed2_step005932'
baseline=d['per_theorem']['baseline'];model=d['per_theorem'][endpoint]
eligible={sid for sid,a in model.items() if a['first'] and not baseline[sid]['first'] and a.get('shortest_compressed')==7}
assert len(eligible)==9
paths={name:R/f'results/{name}_final_greedy/attempts.jsonl' for name in ('baseline',endpoint,retention)}
rows={name:common.read_jsonl(path) for name,path in paths.items()}
chosen=min((a for a in rows[endpoint] if a['source_id'] in eligible),key=lambda a:(a['generated_tokens'],a['source_id']))
sid=chosen['source_id']
selected={name:next(a for a in attempts if a['source_id']==sid) for name,attempts in rows.items()}
verdicts={name:strict_complete(row['prompt'],row['body']) for name,row in selected.items()}
assert verdicts[endpoint]['ok'] and verdicts[endpoint]['independent_ok']
assert not verdicts['baseline']['ok'] and not verdicts[retention]['ok']
compressed=compress_proof(chosen['prompt'],chosen['body'])
assert compressed['compressed_length']==compressed['dependency_pruned_length']==7
assert strict_complete(chosen['prompt'],compressed['compressed_proof'])['independent_ok']
rule_sequences={name:[a['rule'] for a in parse_proof_tokens(row['body'].split())] for name,row in selected.items()}
assert rule_sequences[endpoint]==rule_sequences['baseline']
classical_path=R/'results/classical_primary_final_v1/certificates_NEVER_TRAIN.jsonl'
classical=next(a for a in common.read_jsonl(classical_path) if a['source_id']==sid and a['method']=='heuristic_4096')
cv=strict_complete(classical['prompt'],classical['raw_certificate']);assert cv['ok'] and cv['independent_ok']
profiles={name:{'greedy_success':d['per_theorem'][name][sid]['first'],
                'four_sample_successful_attempts':sampled['per_theorem'][name][sid]['attempt_successes'],
                'four_sample_attempts':sampled['per_theorem'][name][sid]['attempts']} for name in selected}
inputs=[analysis_path,sampled_path,*paths.values(),classical_path,R/'configs/all_lm_endpoints_v1.json']
for name in selected:
    a=R/f'reports/audit_lm_{name}_final_greedy.json'
    if not a.exists():raise ValueError(f'Missing full raw audit: {a}')
    audit=json.loads(a.read_text());assert audit['status']=='passed';inputs.append(a)
selection='Added after final evaluation as an illustration of the reported nine gained seven-line certificates in relative/full seed 2. Select the fewest generated tokens among those nine, breaking ties by theorem identity. No endpoint or inference profile was changed.'
payload={'status':'verified','selection':selection,'eligible_count':len(eligible),'source_id':sid,'endpoint':endpoint,
 'prompt':chosen['prompt'],'greedy_outputs':selected,'verdicts':verdicts,'compression':compressed,
 'rule_sequences':rule_sequences,'profiles':profiles,'classical':classical,'classical_verdict':cv,
 'input_sha256':{str(p):common.sha(p) for p in inputs},'code_sha256':common.sha(__file__),
 'interpretation':'Fixed-allowance proof-construction gain. The initial LM already emits the same rule sequence with invalid citations. This example does not establish a newly acquired rule-order strategy, an intrinsic inability of initialization to prove the theorem, or a net coverage gain.'}
common.write_json(R/'reports/witness_lm_gained_final_v1.json',payload)
text='''# A newly solved LM statement at the fixed evaluation allowance

The relative/full LM endpoint with training seed 2 proves the statement below, while initialization and the matched continued-supervision checkpoint miss it in both recorded profiles. This is a post-hoc illustration, selected for readability from the nine gained greedy statements with a further-compressed seven-line certificate in that endpoint. Selection uses the fewest generated tokens, then theorem identity. The complete endpoint comparison remains 149/768 versus initialization's 173/768: 41 statements gained and 65 lost.

The theorem is **P, P → S, R ⊢ R → (S ∧ R)**. The accepted RL certificate is:

| Line | Formula | Justification |
|---|---|---|
| 1 | P | Premise |
| 2 | P → S | Premise |
| 3 | R | Premise |
| 4 | S | Implication elimination, lines 2 and 1 |
| 5 | [R] | Open an assumption |
| 6 | S ∧ R | Conjunction introduction, lines 4 and 3, inside the assumption scope |
| 7 | R → (S ∧ R) | Implication introduction, discharge the scope from line 5 through line 6 |

Line 6 uses the outer premise R. The assumption at line 5 may remain unused: weakening and vacuous implication introduction are valid in this calculus. The original attempt passes both rule/scope checkers; pruning and further reiteration compression preserve all seven lines, including the three mandatory premises. This certificate contains four non-premise deductions; the seven-line count does not by itself demonstrate a longer inference horizon. The supplied audit reference has twelve lines, and the independent classical heuristic also constructs a seven-line proof. No minimum-proof-length claim is needed.

| Frozen policy | Greedy successful attempts | Successful attempts among four samples |
|---|---:|---:|
| Initialization | 0/1 | 0/4 |
| Relative/full, seed 2 | 1/1 | 4/4 |
| Matched continued supervision | 0/1 | 0/4 |

The initial LM already emits the same seven-rule sequence. Its line 4 cites R as the antecedent for P → S, and its line 6 cites P → S where R is required. Both are invalid deductions. The continued-supervision output instead attempts to infer S ∧ R directly from P → S and R, also invalid. The improved endpoint corrects these citation choices. This is useful proof construction, with evidence specific to an already produced rule pattern. It does not show a newly acquired rule-order strategy, and more sampling from initialization remains an untested counterfactual.

The classical witness opens the R assumption before deriving S and then uses the assumed R in the conjunction. It is a separate existing output, never a training certificate.

The [witness JSON](witness_lm_gained_final_v1.json) retains every original response, both verifier verdicts, the classical certificate, selection rule and input hashes. The [complete gained/lost table](lm_gained_lost_final_v1.md) includes all endpoints and both profiles. No new training or inference was performed for this illustration.
'''
(R/'reports/witness_lm_gained_final_v1.md').write_text(text)
print(json.dumps({'status':'verified','source_id':sid,'eligible_count':len(eligible),'generated_tokens':chosen['generated_tokens'],'pruned_lines':7,'same_rule_sequence_as_initialization':True,'profiles':profiles}))
