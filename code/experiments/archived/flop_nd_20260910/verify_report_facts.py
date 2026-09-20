"""Reconcile final report claims and saved artifact identities without running inference."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import common

ROOT = common.ROOT
facts=[]
read_paths=set()
def read(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    read_paths.add(p)
    return json.loads(p.read_text())
def equal(label, actual, expected):
    if actual != expected:raise AssertionError((label,actual,expected))
    facts.append({'claim':label,'observed':actual})
def near(label, actual, expected, atol=1e-9):
    if not math.isclose(actual,expected,abs_tol=atol,rel_tol=0):raise AssertionError((label,actual,expected))
    facts.append({'claim':label,'observed':actual})
def analysis(study,population,profile):
    stem=f'{study}_' if study else ''
    return read(f'reports/analysis_{stem}{population}_final_{profile}_v1.json')

release=read('configs/all_lm_endpoints_v1.json')
equal('Frozen LM endpoint count',len(release['checkpoint_paths']),21)
main_names=[f'main_{c}_seed{s}_v1' for c in ('relative_full','relative_reset','ppo_full','ppo_reset') for s in (1,2)]
for profile,overall,longer in [('greedy',[137,149,147,148,94,89,88,94],[37,47,36,42,15,22,15,23]),
                               ('sample4',[157,163,163,167,113,112,109,105],[49,53,45,49,20,27,19,26])]:
    d=analysis('','primary',profile);metric='first' if profile=='greedy' else 'any'
    equal(f'LM {profile} endpoint coverage',len(d['summaries']),21)
    equal(f'LM {profile} all statements',d['summaries']['baseline']['all']['theorems'],768)
    equal(f'LM {profile} primary reference stratum',d['summaries']['baseline']['reference_over6']['theorems'],576)
    equal(f'LM {profile} initialization overall',d['summaries']['baseline']['all'][metric],173 if profile=='greedy' else 201)
    equal(f'LM {profile} initialization primary stratum',d['summaries']['baseline']['reference_over6'][metric],48 if profile=='greedy' else 63)
    for n,x,y in zip(main_names,overall,longer):
        equal(f'{n} {profile} overall',d['summaries'][n]['all'][metric],x)
        equal(f'{n} {profile} primary stratum',d['summaries'][n]['reference_over6'][metric],y)
    s=analysis('','structured',profile)
    equal(f'Structured LM {profile} endpoints',len(s['summaries']),21)
    for n,a in s['summaries'].items():
        equal(f'{n} structured {profile} denominator',a['all']['theorems'],384)
        equal(f'{n} structured {profile} successes',a['all'][metric],0)
    for n,count in zip(['online_on_seed1_v1','online_off_seed1_v1','online_on_seed2_v1','online_off_seed2_v1'],
                       [62,88,67,96] if profile=='greedy' else [73,100,80,111]):
        equal(f'{n} {profile}',d['summaries'][n]['all'][metric],count)
    retention=[a['all'][metric] for n,a in d['summaries'].items() if n.startswith('retention')]
    if profile=='greedy':equal('Retention final greedy range',[min(retention),max(retention)],[132,148])

for study,seeds,pop,profile,expected in [
 ('general',[4301,4302],'primary','greedy',{'bc':[122,111],'full':[157,173],'reset':[176,191]}),
 ('general',[4301,4302],'structured','greedy',{'bc':[15,6],'full':[117,166],'reset':[239,159]}),
 ('general',[4301,4302],'structured','sample4',{'bc':[26,17],'full':[173,207],'reset':[237,178]}),
 ('minimal',[4401,4402],'primary','greedy',{'bc':[69,70],'reset':[93,95]}),
 ('minimal',[4401,4402],'structured','greedy',{'bc':[74,83],'full':[0,0],'reset':[64,79]}),
 ('minimal',[4401,4402],'structured','sample4',{'bc':[121,134],'full':[0,0],'reset':[116,142]})]:
    d=analysis(study,pop,profile);metric='first' if profile=='greedy' else 'any'
    for condition,counts in expected.items():
        for seed,count in zip(seeds,counts):equal(f'{study} {condition} seed{seed} {pop} {profile}',d['summaries'][f'{condition}_s{seed}']['all'][metric],count)
    if study=='general' and pop=='structured':
        for c in ('full','reset'):
            for seed in seeds:equal(f'General {c} seed{seed} scoped {profile}',d['summaries'][f'{c}_s{seed}']['family:scope_projection'][metric],0)
    if study=='minimal' and pop=='structured':
        for n,a in d['summaries'].items():equal(f'Minimal {n} scoped {profile}',a['family:scope_projection'][metric],int(profile=='sample4' and n=='reset_s4402'))

l=read('reports/analysis_length_challenge_v1.json')
equal('Length challenge cells',len(l['records']),27)
equal('Length challenge missing audits',l['missing_complete_audits'],[])
expected_depth={33:{'reset_s4401':3,'reset_s4402':2,'bc_s4401':3,'bc_s4402':4},65:{'reset_s4401':1,'reset_s4402':3,'bc_s4401':0,'bc_s4402':2},129:{'reset_s4401':1,'reset_s4402':2,'bc_s4401':0,'bc_s4402':0}}
for r in l['records']:
    equal(f"Length {r['reference_lines']} {r['policy']} solved",r['summaries']['all']['first'],expected_depth[r['reference_lines']].get(r['policy'],0))
    equal(f"Length {r['reference_lines']} {r['policy']} compound",r['summaries']['family:projection_compound']['first'],0)
w=l['illustrative_maximum_witness']
for k,v in {'raw_lines':130,'pruned_lines':129,'compressed_lines':129,'pruned_tokens':34055}.items():equal('129-line witness '+k,w[k],v)
w=read('reports/witness_minimal_scoped_final_v1.json')
equal('Scoped witness new demonstrated rules',sorted(w['rules_absent_from_demonstrations']),['AS','IMPI'])
equal('Scoped witness compressed rules',w['compressed_rule_counts'],{'ANDE1':3,'ANDE2':4,'AS':1,'IMPI':1})

for n,primary,structured in [('baseline',297,9),('main_relative_full_seed1_v1',260,0),('main_relative_reset_seed1_v1',271,4),('main_ppo_full_seed1_v1',226,0),('main_ppo_reset_seed1_v1',224,0)]:
    for pop,count in [('primary',primary),('structured',structured)]:
        d=read(f'results/legal_lm_{n}_{pop}_final_greedy_v1/result.json');equal(f'Legal decoder {n} {pop}',d['successes'],count)
for pop,n in [('primary',768),('structured',640)]:
    d=read(f'results/classical_{pop}_final_v1/result.json');equal(f'Classical {pop} heuristic',d['metrics']['heuristic_4096']['verified_successes'],n)
    equal(f'Classical {pop} truth-assignment constructor',d['metrics']['truth_assignment_100000']['verified_successes'],n)
    equal(f'Classical {pop} workers remaining',d['owned_worker_processes_remaining'],0)
    if pop=='primary':
        h=d['metrics']['heuristic_4096'];equal('Primary heuristic fits 512 tokens',h['raw_budget_eligible_successes']['512'],768)
        equal('Primary heuristic at most six lines',sum(v for k,v in h['raw_line_histogram'].items() if int(k)<=6),383)
d=read('reports/review_environment_reachability_final_v2.json')
for name,count in [('general_primary',763),('general_structured',384),('minimal_structured',640)]:equal('Constructed replay '+name,d['cohorts'][name]['constructively_reachable'],count)

for n in main_names:
    d=read(f'results/{n}/result.json');equal(f'{n} stopping reason',d['stop_reason'],'flop_proxy_ceiling')
    equal(f'{n} raw audit',read(f'reports/audit_{n}.json')['status'],'passed')
for n in [f'online_{m}_seed{s}_v1' for m in ('on','off') for s in (1,2)]:equal(f'{n} raw audit',read(f'reports/audit_{n}.json')['status'],'passed')
for s in (1,2):equal(f'Retention audit seed{s}',read(f'reports/audit_retention_seed{s}_v1.json')['status'],'passed')
d=read('jobs/final_comparators_v1/result.json');equal('Final CPU comparator cells',len(d['tasks']),84)
equal('Final CPU comparator completed cells',sum(v['status']=='completed' and v['exit_code']==0 for v in d['tasks'].values()),84)
t=read('reports/endpoint_tables.json');equal('Numeric compiler missing final analyses',t['missing'],[])

hashes={}
def digest(p):
    p=Path(p)
    if p not in hashes:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        hashes[p]=h.hexdigest()
    return hashes[p]
identity_claims=0
for name,path in release['checkpoint_paths'].items():
    equal(f'Frozen checkpoint identity {name}',digest(path),release['checkpoint_sha256'][name]);identity_claims+=1
for population in ('primary','structured'):
    for profile in ('greedy','sample4'):
        d=analysis('',population,profile)
        for name,r in d['run_provenance'].items():
            for field,filename in [('attempts_sha256','attempts.jsonl'),('result_sha256','result.json')]:
                p=Path(r['directory'])/filename
                if digest(p)!=r[field]:raise AssertionError(('LM provenance mismatch',str(p)))
                identity_claims+=1
            equal(f'{name} {population} {profile} frozen checkpoint',r['checkpoint_sha256'],release['checkpoint_sha256'][name])
print(json.dumps({'phase':'headline_and_lm_identity_checks','facts':len(facts),'hashed_files':len(hashes)}),flush=True)

# Reconcile each stored absolute input identity in the proof audits and numeric reports.
# This checks artifact integrity; proof verdicts were independently replayed in the cited audits.
input_docs=list((ROOT/'reports').glob('audit_*.json'))+list((ROOT/'reports').glob('analysis_*final*v1.json'))
input_docs += [ROOT/'reports/analysis_length_challenge_v1.json',ROOT/'reports/endpoint_tables.json',ROOT/'reports/witness_minimal_scoped_final_v1.json']
nonpath_input_keys=[]
audit_statuses=collections.Counter()
for p in input_docs:
    d=read(p)
    if p.name.startswith('audit_'):
        state=d.get('status')
        if state is None:
            equal(p.name+' successful-archive rows reconciled',d['counts']['strict_successes_reverified'],d['counts']['rows'])
            state='successful_archive_schema_without_status'
        elif state not in ('complete','completed','passed'):raise AssertionError(('Audit incomplete',str(p),state))
        audit_statuses[state]+=1
    inputs=d.get('input_sha256',{})
    if not isinstance(inputs,dict):continue
    for key,wanted in inputs.items():
        if not isinstance(key,str) or not isinstance(wanted,str):
            nonpath_input_keys.append({'report':str(p),'key':key});continue
        target=Path(key) if Path(key).is_absolute() else ROOT/key
        if not target.is_file():raise AssertionError(('Saved input path missing',str(p),key))
        if digest(target)!=wanted:raise AssertionError(('Saved input hash mismatch',str(p),key))
        identity_claims+=1
print(json.dumps({'phase':'audit_input_identities','hashed_files':len(hashes),'identity_claims':identity_claims}),flush=True)
job_records=[]
for p in sorted((ROOT/'jobs').glob('*/completed.json')):
    d=read(p);equal(p.parent.name+' GPU job completed',d['status'],'completed');equal(p.parent.name+' GPU exit code',d['exit_code'],0)
    job_records.append({'job':p.parent.name,**d})
leased_seconds=sum(d['elapsed_seconds'] for d in job_records)
receipt={'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only reconciliation of report numbers, frozen checkpoint/evaluation identities and saved audit input hashes; no new training or inference. This does not independently replay neural optimization.',
 'facts':facts,'verified_facts':len(facts),'identity_claims_checked':identity_claims,'unique_files_hashed':len(hashes),
 'audit_status_counts':dict(audit_statuses),'nonscalar_input_keys_not_rehashed':nonpath_input_keys,
 'gpu_jobs':job_records,'local_gpu_leased_job_wall_seconds':leased_seconds,'local_gpu_leased_job_wall_hours':leased_seconds/3600,
 'new_paid_spending_usd':0,'code_sha256':common.sha(__file__),
 'source_document_sha256':{str(p):digest(p) for p in sorted(read_paths)}}
common.write_json(ROOT/'reports/final_fact_verification_v1.json',receipt)
print(json.dumps({'status':'passed','facts':len(facts),'unique_files_hashed':len(hashes),'identity_claims':identity_claims,'gpu_leased_hours':leased_seconds/3600,'nonpath_keys':len(nonpath_input_keys)}),flush=True)
