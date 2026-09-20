"""Freeze the time-limited remaining controls and final evaluation before final access."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/users/dmitry/simplex-research/.venv/bin/python'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')

def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    management = ROOT/'snapshots/finish_management_v1'
    management.mkdir()
    shutil.copy2(ROOT/'tools/release_checkpoints.py', management/'release_checkpoints.py')
    shutil.copy2(ROOT/'tools/close_training.py', management/'close_training.py')
    commands, inputs, snapshots, training_outputs = [], set(), {str(management)}, []
    # Preserve each previously frozen scientific command; change scheduling only.
    for name in ['retention_control_seed1_v2', 'main_oracle_control_v1', 'prefix_comparison_v1', 'supplementary_diagnostics_v1']:
        path = ROOT/'configs'/f'pipeline_{name}.json'
        plan = json.loads(path.read_text())
        inputs.add(str(path)); inputs.update(plan['inputs']); snapshots.update(plan['source_snapshots'])
        commands.extend(plan['commands'])
        ntrain = 2 if name == 'prefix_comparison_v1' else (0 if name == 'supplementary_diagnostics_v1' else 1)
        training_outputs.extend(c[c.index('--output')+1] for c in plan['commands'][:ntrain])

    checklist = json.loads((ROOT/'code/tests/projected_gradient_frozen_input_checklist.json').read_text())
    projected = [PYTHON, checklist['frozen_program']]
    for k,v in checklist['launch_arguments'].items():
        projected += ['--'+k, str(ROOT/v) if isinstance(v,str) and (v.startswith('results/') or v.startswith('data/') or v.startswith('configs/')) else str(v)]
    projected += ['--device','cuda','--cpu-threads','4']
    commands.append(projected)
    snapshots.add(str(ROOT/'snapshots/projected_gradient_v1'))
    future_inputs = set(checklist['currently_missing_required_inputs'])
    for p in checklist['required_inputs']:
        if p not in future_inputs:
            inputs.add(p)
    inputs.add(str(ROOT/'code/tests/projected_gradient_frozen_input_checklist.json'))

    development = json.loads((ROOT/'configs/analysis_main_development_seed1_v1.json').read_text())
    model_checkpoints = {}
    for arm in ['selected_full','selected_reset','uniform_reset','uniform_full']:
        for label, ceiling in [('f05',500000000000000),('f10',1000000000000000),('f20',2000000000000000),('f40',4000000000000000)]:
            filename = 'final.pt' if label == 'f40' else f'flops_{ceiling}.pt'
            model_checkpoints[f'{arm}_s1_{label}'] = str(ROOT/'results'/f'rl_{arm}_seed1_v1'/filename)
    for name, run in [('adapt','adapt'),('transfer','mixture_transfer'),('surprise','mixture_surprise'),('uniform','mixture_uniform'),('uniform_matched','mixture_uniform_matched'),('oracle_matched','oracle_matched'),('oracle_total','oracle_total')]:
        model_checkpoints[name] = str(ROOT/'results'/f'{run}_v1/final.pt')
    for step in [8827,9054,9056,9096]:
        model_checkpoints[f'retention_step{step}'] = str(ROOT/'results/retention_control_seed1_v1'/f'step_{step:06d}.pt')
    model_checkpoints['retention_compute'] = str(ROOT/'results/retention_control_seed1_v1/final.pt')
    model_checkpoints['main_oracle_matched'] = str(ROOT/'results/main_oracle_control_v1/step_003762.pt')
    model_checkpoints['main_oracle_total'] = str(ROOT/'results/main_oracle_control_v1/final.pt')
    for mode in ['reference','mined']:
        model_checkpoints[f'prefix_{mode}'] = str(ROOT/'results'/f'prefix_{mode}_sft_v1/final.pt')
    families = json.loads((ROOT/'configs/recipe_calibration_confirmation_v1.json').read_text())['checkpoint_paths'][:24]
    final_data = str(ROOT/'data/released_v1/final_confirmation.jsonl')
    final_release = str(ROOT/'configs/release_final_seed1_v1.json')
    models = {n: {'nll':str(ROOT/'results'/f'final_{n}_v1/whole_proof_nll.jsonl'), 'attempts':str(ROOT/'results'/f'final_{n}_v1/attempts.jsonl')} for n in model_checkpoints}
    comparisons = development['comparisons'] + [[n,'adapt'] for n in model_checkpoints if n not in development['models']]
    comparisons += [['transfer','uniform_matched'],['transfer','surprise'],['prefix_mined','prefix_reference']]
    for arm,step in [('selected_full',9056),('selected_reset',9054),('uniform_full',9096),('uniform_reset',8827)]:
        comparisons += [[f'{arm}_s1_f40',f'retention_step{step}'],[f'{arm}_s1_f40','retention_compute'],[f'{arm}_s1_f40','main_oracle_matched']]
    analysis_path = ROOT/'configs/analysis_final_seed1_v1.json'
    write(analysis_path, dict(data=final_data,samples=8,bootstrap_draws=5000,bootstrap_seed=20260911,models=models,comparisons=comparisons,contrasts=development['contrasts'],output=str(ROOT/'reports/final_seed1_v1.json'),status='frozen_before_final_access',note='One online training seed; context uncertainty does not estimate training-seed variability. All four main milestones retained. Final 4e15 endpoint is the primary comparison; earlier checkpoints are learning-curve diagnostics.'))

    protocol = dict(utc=now,no_further_training_after_closure=True,declared_training_outputs=training_outputs,
        final_evaluation_models=model_checkpoints,calibration_score_only= families,
        unrun_cancelled_for_this_sprint=['factorial_seed2_v1','rl_calibration_v1','rl_calibration_scoring_v2','new_RL_selected_mixture','difficulty_initialized_RL'],
        selection_rule='No checkpoint chosen by development outcomes; all declared main milestones and completed controls are included.',
        closure_output=str(ROOT/'reports/training_closed_v1.json'),
        main_provenance=str(ROOT/'reports/main_development_seed1_provenance_v1.json'))
    closure_plan = ROOT/'configs/training_closure_plan_v1.json'; write(closure_plan,protocol)
    recipe_path = ROOT/'configs/recipe_final_seed1_v1.json'
    bound = [closure_plan,analysis_path,ROOT/'configs/frozen_selectors_v1.json',ROOT/'configs/recipe_main_development_seed1_v1.json',ROOT/'theory/final_prefix_exposure_analysis_rule_v1.json',ROOT/'data/released_v1/manifest.json',ROOT/'tools/analyze_evaluations.py',Path(__file__).resolve(),ROOT/'tools/close_training.py']
    recipe = dict(frozen=True,utc=now,role='final_confirmation',no_further_training=True,data=final_data,
        population_manifest=str(ROOT/'data/released_v1/manifest.json'),checkpoint_paths=list(model_checkpoints.values())+families,
        output=final_release,decoding=dict(start='full',samples=8,temperature=.8,max_new_tokens=512,seed=20260910),
        prospective_input_sha256={str(p):sha(p) for p in bound},
        recipe_note='Declared before final payload access. Closure stage must complete and no training may follow it. Scientific training commands inherited unchanged from frozen plans; scheduling consolidated after queue failure.')
    write(recipe_path,recipe)
    commands += [[PYTHON,str(management/'close_training.py'),str(closure_plan)], [PYTHON,str(management/'release_checkpoints.py'),str(recipe_path)]]
    source = ROOT/'snapshots/confirmation_v1'
    snapshots.add(str(source))
    commands.append([PYTHON,str(source/'score_only.py'),'--checkpoints',*families,'--data',final_data,'--release',final_release,'--output',str(ROOT/'results/final_calibration_scores_v1'),'--device','cuda','--precision','fp32','--cpu-threads','4'])
    template = json.loads((ROOT/'configs/pipeline_main_development_seed1_v1.json').read_text())['commands'][1]
    for name, checkpoint in model_checkpoints.items():
        command = list(template)
        for flag,value in [('--checkpoint',checkpoint),('--data',final_data),('--release',final_release),('--output',str(ROOT/'results'/f'final_{name}_v1'))]:
            command[command.index(flag)+1] = value
        commands.append(command)
    inputs.update(str(p) for p in bound)
    inputs.update([str(recipe_path),final_data])
    inputs.update(p for p in recipe['checkpoint_paths'] if Path(p).exists())
    missing = [p for p in inputs if not Path(p).is_file()]
    if missing: raise ValueError(missing)
    plan = dict(name='finish_seed1_v1',question='Finish the prespecified short controls and evaluate every declared completed endpoint on untouched final contexts after global training closure.',
        expected_seconds=2400,timeout_seconds=3600,deadline_utc='2026-09-11T02:58:00+00:00',
        source_snapshots=sorted(snapshots),inputs=sorted(inputs),commands=commands,status='frozen_ready',
        scientific_amendment='Time-driven cancellation of seed2 and RL-specific calibration; commands for executed controls unchanged.',
        final_models=len(model_checkpoints),final_score_only_models=len(families),global_closure_stage=len(commands)-len(model_checkpoints)-3)
    write(ROOT/'configs/pipeline_finish_seed1_v1.json',plan)
    print(json.dumps({'pipeline':str(ROOT/'configs/pipeline_finish_seed1_v1.json'),'stages':len(commands),'final_models':len(model_checkpoints),'final_score_only':len(families)}))

if __name__ == '__main__': main()
