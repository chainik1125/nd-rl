"""Declare every saved retention exposure comparison on reused development data."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data/users/dmitry/simplex-research/.venv/bin/python'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    mapping_path=ROOT/'configs/comparisons_retention_control_seed1_v1.json'
    mapping=json.loads(mapping_path.read_text()); models={}
    for item in mapping['exposure_controls'].values(): models[f'retention_step{item["main_step"]}']=item['control_checkpoint']
    models['retention_compute']=mapping['total_compute_control']
    models['main_oracle_matched']=str(ROOT/'results/main_oracle_control_v1/step_003762.pt')
    models['main_oracle_total']=str(ROOT/'results/main_oracle_control_v1/final.pt')
    original_path=ROOT/'configs/analysis_main_development_seed1_v1.json'
    analysis=json.loads(original_path.read_text())
    for n in models:
        analysis['models'][n]={'nll':str(ROOT/'results'/f'confirmation_{n}_v1/whole_proof_nll.jsonl'),'attempts':str(ROOT/'results'/f'confirmation_{n}_v1/attempts.jsonl')}
        analysis['comparisons'].append([n,'adapt'])
    for main,item in mapping['exposure_controls'].items(): analysis['comparisons'].append([main,f'retention_step{item["main_step"]}'])
    for label in ['f05','f10','f20','f40']:
        for contrast in ['interaction','selection_main_effect','reset_main_effect']:
            coefficients=analysis['contrasts'][f'{contrast}_{label}']; ret={}; residual=dict(coefficients)
            for model,weight in coefficients.items():
                name=f'retention_step{mapping["exposure_controls"][model]["main_step"]}'
                ret[name]=ret.get(name,0)+weight;residual[name]=residual.get(name,0)-weight
            analysis['contrasts'][f'retention_reference_{contrast}_{label}']={k:v for k,v in ret.items() if v}
            analysis['contrasts'][f'residual_{contrast}_{label}']={k:v for k,v in residual.items() if v}
    analysis.update(output=str(ROOT/'reports/retention_development_v1.json'),utc=now,status='frozen_before_control_development_predictions',note='All15 unique retention exposure milestones plus equal-total-compute endpoint and both main oracle endpoints. Reuses frozen main development data/predictions. Exact contrast split against retention reference is arithmetic, not additive causal attribution. No new training and no checkpoint selection from outcomes.')
    analysis_path=ROOT/'configs/analysis_retention_development_v1.json';write(analysis_path,analysis)
    release=ROOT/'configs/release_retention_development_v1.json';recipe_path=ROOT/'configs/recipe_retention_development_v1.json'
    bound=[mapping_path,original_path,analysis_path,Path(__file__).resolve(),ROOT/'tools/analyze_evaluations.py',ROOT/'data/released_v1/manifest.json',ROOT/'reports/main_development_seed1_provenance_v1.json']
    for name in ['retention_control_seed1_v1','main_oracle_control_v1']:
        bound += [ROOT/'results'/name/f for f in ['manifest.json','result.json','completed.json']]
    recipe=dict(frozen=True,utc=now,role='confirmation',no_further_training=False,data=analysis['data'],population_manifest=str(ROOT/'data/released_v1/manifest.json'),checkpoint_paths=list(models.values()),output=str(release),decoding=dict(start='full',samples=8,temperature=.8,max_new_tokens=512,seed=20260910),prospective_input_sha256={str(p):sha(p) for p in bound},recipe_note='All retention checkpoints saved prospectively from fixed training trajectory; metadata-only exposure matching. Execute after the main finish pipeline. No optimization is part of this pipeline.')
    write(recipe_path,recipe)
    source=ROOT/'snapshots/confirmation_v1';management=ROOT/'snapshots/finish_management_v1'
    commands=[[PYTHON,str(management/'release_checkpoints.py'),str(recipe_path)]]
    template=json.loads((ROOT/'configs/pipeline_main_development_seed1_v1.json').read_text())['commands'][1]
    for name,cp in models.items():
        c=list(template)
        for flag,value in [('--checkpoint',cp),('--data',analysis['data']),('--release',str(release)),('--output',str(ROOT/'results'/f'confirmation_{name}_v1'))]:c[c.index(flag)+1]=value
        commands.append(c)
    inputs={str(p) for p in bound}|set(models.values())|{str(recipe_path),analysis['data']}
    plan=dict(name='retention_development_v1',question='At every frozen main compute milestone, how does including policy loss change heldout performance relative to the same ordered retention exposure?',expected_seconds=300,timeout_seconds=900,deadline_utc='2026-09-11T02:58:00+00:00',source_snapshots=[str(source),str(management)],inputs=sorted(inputs),commands=commands,status='frozen_ready_after_finish_pipeline',training_updates=0,final_payload_access=False)
    write(ROOT/'configs/pipeline_retention_development_v1.json',plan)
    print(json.dumps({'models_to_evaluate':len(models),'analysis_models':len(analysis['models']),'analysis_contrasts':len(analysis['contrasts'])}))

if __name__=='__main__':main()
