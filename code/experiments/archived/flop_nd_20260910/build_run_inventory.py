"""Summarize completed training resources and controller outcomes for the report."""
import datetime
import json
from pathlib import Path
import common

R=common.ROOT
inputs={}
def read(p):
    p=Path(p);p=p if p.is_absolute() else R/p
    inputs[str(p)]=common.sha(p)
    return json.loads(p.read_text())
main=[]
for condition in ('relative_full','relative_reset','ppo_full','ppo_reset'):
    for seed in (1,2):
        name=f'main_{condition}_seed{seed}_v1';d=read(f'results/{name}/result.json');a=read(f'reports/audit_{name}.json')
        main.append({'name':name,'config':f'configs/{name}.json','result':f'results/{name}/result.json',
            'updates':d['step'],'optimizer_steps':d['optimizer_steps'],'generated_tokens':d['generated_tokens'],
            'prompt_and_prefix_prefill_tokens':d['ledger']['generation']['prefill_actual_tokens'],
            'retention_record_presentations':d['retention_examples'],'retention_response_tokens':d['retention_response_tokens'],
            'flop_proxy':d['cumulative_flop_proxy'],'flop_overshoot_fraction':d['flop_proxy_overshoot']/d['max_flop_proxy'],
            'training_wall_seconds':d['elapsed_seconds'],'stop_reason':d['stop_reason'],
            'source_attempts':d['attempts']['source'],'target_attempts':d['attempts']['target'],
            'unique_training_targets_proved':a['unique_discovered_targets'],'checkpoint_sha256':d['final_checkpoint_sha256']})
online=[]
for seed in (1,2):
    for mode in ('off','on'):
        name=f'online_{mode}_seed{seed}_v1';d=read(f'results/{name}/result.json');a=read(f'reports/audit_{name}.json')
        online.append({'name':name,'result':f'results/{name}/result.json','updates':d['step'],'generated_tokens':d['generated_tokens'],
            'prompt_and_prefix_prefill_tokens':d['ledger']['generation']['prefill_actual_tokens'],
            'flop_proxy':d['cumulative_flop_proxy'],'training_wall_seconds':d['elapsed_seconds'],'stop_reason':d['stop_reason'],
            'initial_target_proof_bank':a['initial_bank_count'],'final_target_proof_bank':a['final_bank_count'],
            'new_first_target_proofs':a['new_first_proofs'],'checkpoint_sha256':d['final_checkpoint_sha256']})
retention=[]
for seed in (1,2):
    name=f'retention_seed{seed}_v1';d=read(f'results/{name}/result.json')
    retention.append({'name':name,'updates':d['step'],'optimizer_steps':d['optimizer_steps'],'retention_record_presentations':d['retention_examples'],
        'retention_response_tokens':d['retention_response_tokens'],'flop_proxy':d['cumulative_flop_proxy'],'training_wall_seconds':d['elapsed_seconds'],
        'result':f'results/{name}/result.json'})
symbolic=[]
expected_symbolic_names = {f'symbolic_general_{c}_s{s}' for c in ('full','reset','bc') for s in (4301,4302)}
expected_symbolic_names |= {f'minimal_general_{c}_s{s}_v1' for c in ('full','reset','bc') for s in (4401,4402)}
expected_symbolic_names |= {f'detour_{v}_{c}_s{s}_v1' for v in ('clean','noisy') for c in ('reset','bc') for s in (4501,4502)}
patterns=[('general','results/symbolic_general_*/symbolic_general_train_summary.json'),
          ('minimal','results/minimal_general_*/symbolic_general_train_summary.json'),
          ('detour','results/detour_*/symbolic_general_train_summary.json')]
for study,pattern in patterns:
    for p in sorted(R.glob(pattern)):
        if p.parent.name not in expected_symbolic_names:continue
        d=read(p)
        if d.get('status')!='complete':raise AssertionError((str(p),d.get('status')))
        symbolic.append({'study':study,'name':p.parent.name,'result':str(p.relative_to(R)),'config':d.get('config'),
                         'sampled_actions':d.get('sampled_actions',0),'updates':d.get('updates'),
                         'optimizer_steps':d.get('optimizer_steps'),'training_wall_seconds':d['elapsed_seconds'],
                         'accounting':d.get('accounting')})
# Cloning summaries use a separate filename; preserve their exact original accounting.
for study,pattern in [('general','results/symbolic_general_bc_*/symbolic_general_bc_summary.json'),
                      ('minimal','results/minimal_general_bc_*/symbolic_general_bc_summary.json'),
                      ('detour','results/detour_*_bc_*/symbolic_general_bc_summary.json')]:
    for p in sorted(R.glob(pattern)):
        if p.parent.name not in expected_symbolic_names:continue
        d=read(p);symbolic.append({'study':study,'name':p.parent.name,'result':str(p.relative_to(R)),'summary':d})
if {s['name'] for s in symbolic} != expected_symbolic_names or len(symbolic) != 20:
    raise AssertionError('The inventory requires exactly the 20 declared symbolic training endpoints')
jobs=[]
for p in sorted((R/'jobs').glob('*/completed.json')):jobs.append({'job':p.parent.name,**read(p)})
controllers=[]
for p in sorted((R/'jobs').glob('*/result.json')):
    d=read(p);controllers.append({'job':p.parent.name,'status':d.get('status'),'result':str(p.relative_to(R)),'error':d.get('error')})
noncompleted=[c for c in controllers if c['status'] not in ('complete','completed')]
if [c['job'] for c in noncompleted] != ['remaining_v1']:raise AssertionError(noncompleted)
output={'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main_lm':main,'online_lm':online,'retention_lm':retention,
        'symbolic_training':symbolic,'gpu_jobs':jobs,'controller_results':controllers,'input_sha256':inputs,'code_sha256':common.sha(__file__),
        'gpu_leased_wall_seconds':sum(j['elapsed_seconds'] for j in jobs),'new_paid_spending_usd':0,
        'interpretation':'Leased GPU job wall time includes evaluation and setup; it is not measured kernel utilization. CPU task wall times can overlap. Original per-run ledgers retain padded token slots and stage-level FLOP proxies.'}
common.write_json(R/'reports/run_inventory_v1.json',output)
lines=['# Completed runs and resource accounting','',
       'All mandatory frozen model conditions and final evaluation cells completed. Training resources below are additional to the shared LM initialization. The declared FLOP count is a dense-operation proxy, including padded work and attention; it is not a hardware counter.','',
       '## Main token-LM training','',
       '| Endpoint | Updates | Generated tokens | Prompt + prefix prefill tokens | FLOP proxy / 10¹⁶ | Training seconds | Unique training targets proved |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in main:lines.append(f"| {r['name']} | {r['updates']:,} | {r['generated_tokens']:,} | {r['prompt_and_prefix_prefill_tokens']:,} | {r['flop_proxy']/1e16:.6f} | {r['training_wall_seconds']:.1f} | {r['unique_training_targets_proved']} |")
lines += ['', 'Every main arm stops at the same 10¹⁶ proxy ceiling after finishing the active update. The maximum overshoot is '+f"{100*max(r['flop_overshoot_fraction'] for r in main):.4f}%"+'. Prompt/prefix prefill includes the theorem prompt and any supplied proof prefix. Policy training, behavior-probability auditing and continued-supervision work are additional ledger entries already included in the total proxy.','',
          'The last column counts distinct training statements with a verified discovered proof, from the raw audit. It does not count separate certificate variants as different targets and does not establish an improvement over equal-budget sampling from initialization.','',
          '## Continued supervision and discovery admission','',
          '| Continuation | Updates | Generated tokens | FLOP proxy / 10¹⁵ | Training seconds | Imported → final target bank | New first proofs |',
          '|---|---:|---:|---:|---:|---:|---:|']
for r in online:lines.append(f"| {r['name']} | {r['updates']:,} | {r['generated_tokens']:,} | {r['flop_proxy']/1e15:.6f} | {r['training_wall_seconds']:.1f} | {r['initial_target_proof_bank']} → {r['final_target_proof_bank']} | {r['new_first_target_proofs']} |")
lines += ['', 'Admission on/off shares the same parent and continuation restart within a seed. Both arms archive first proofs; only on replays target proof prefixes. The comparison includes imported parent discoveries.','',
          '| Supervision-only trajectory | Updates | Labeled record presentations | FLOP proxy / 10¹⁵ | Training seconds |',
          '|---|---:|---:|---:|---:|']
for r in retention:lines.append(f"| {r['name']} | {r['updates']:,} | {r['retention_record_presentations']:,} | {r['flop_proxy']/1e15:.6f} | {r['training_wall_seconds']:.1f} |")
lines += ['', 'These two trajectories supply eight exposure-matched checkpoints. Their exact draws, token counts, optimizer exposure and checkpoint mapping were audited. They match supervision exposure, not total compute.','',
          '## Local GPU lease accounting','', '| Job | Leased wall seconds | Completion UTC | Exit code |','|---|---:|---|---:|']
for j in jobs:lines.append(f"| {j['job']} | {j['elapsed_seconds']:.1f} | {j['utc']} | {j['exit_code']} |")
lines += ['',f"Total: **{output['gpu_leased_wall_seconds']/3600:.4f} leased GPU-hours** on local GPU 3. This includes setup and evaluation inside the leases. New paid spending: **$0**. No cloud jobs, uploads or publications were launched.",'',
          '## Completed CPU studies','',
          '- Broader symbolic policy: four PPO conditions and two cloning controls; 131,072 sampled actions per PPO run, 278,000 labeled action presentations per cloning run.',
          '- Minimal symbolic policy: four PPO conditions and two cloning controls; 131,072 sampled actions per PPO run, 10,000 labeled action presentations per cloning run.',
          '- Detour diagnostic: four PPO conditions and four cloning controls; 131,072 sampled actions per PPO run, 6,000 clean or 10,000 noisy labeled action presentations per cloning run. Development evaluation only.',
          '- Extra depth challenge: all 27 frozen-policy evaluation cells and all 27 raw-action/proof audits; no new training.',
          '- Final evaluation: 84 LM cells and 84 CPU comparator cells. CPU cells comprise 72 symbolic evaluations, ten legal-decoder evaluations and two classical populations, each with three constructors/budgets.',
          '', 'CPU runtime, candidate counts, optimizer counts, configurations and individual result paths are retained in [the numeric inventory](run_inventory_v1.json). CPU wall times overlap and should not be added to GPU lease time. PPO and cloning compute budgets differ.',
          '', '## Interrupted, superseded and unlaunched work','',
          '- Five waiting controllers ended with SIGTERM/exit 143: minimal development, primary raw audits, final LM analysis, environment reachability and online raw audits. Completed outputs were preserved. Immutable continuation configurations ran only unfinished tasks with the same frozen code and inputs. No training run was interrupted by these events.',
          '- `remaining_v1` was deliberately stopped while waiting and replaced by `remaining_v2`; it launched no training. Its failed result with `KeyboardInterrupt` is retained.',
          '- Initial source-only evaluation preparations failed proof-field, protected-output or missing-name checks before generation. The successful `baseline_sources_greedy_v3` diagnostic retains the corrected theorem-only input and its audit.',
          '- Length-audit configuration v1 was never launched because preflight found an incompatible watcher key. Version 2 fixed scheduling metadata; all 27 tasks completed.',
          '- The earlier restricted projection pilot is retained in `reports/symbolic_report.md`. Its uniform control exposed a weak challenge, so it is not evidence for the reported general legal-action result.',
          '- A standalone gamma=1/lambda=.95 full-versus-reset LM ablation was a prospective option, but was not run. GAE appeared in the matched online continuations together with other revised settings; those runs do not isolate its effect.',
          '- The prepared structured theorem-only training subset was not used. Training an LM through the legal-action decoder, a compact state representation, an explicit active-subgoal ablation and an equal-budget frozen-initial-policy training-target sampling counterfactual remain proposed.',
          '- Three assistants reached the account usage limit after their bounded contributions. Root completed the subsequent implementation and audits. A fresh-context final-report review by another agent was therefore unavailable; the report records root factual, visual and adversarial checks.',
          '', 'Old frozen progress or training-manifest fields may retain a waiting/active or unreleased value. Completed result files, the shared final endpoint release and the completed final controller results provide the final status. Frozen historical files have not been rewritten to erase this distinction.',
          '', 'The parent directory has no usable Git repository. This sprint is isolated in its own directory and uses pinned code/data/checkpoint hashes. No branch, commit, PR or publication is claimed.']
(R/'reports/run_inventory_v1.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'completed','main_lm':len(main),'online_lm':len(online),'retention_trajectories':len(retention),'symbolic_training_records':len(symbolic),'controller_results':len(controllers),'gpu_leased_hours':output['gpu_leased_wall_seconds']/3600}))
