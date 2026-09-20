"""Deterministic illustrative coverage changes, retaining raw proof provenance."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def rows(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','final'],required=True);a=p.parse_args()
    plan_path=ROOT/'configs'/('analysis_main_development_seed1_v1.json' if a.phase=='development' else 'analysis_final_seed1_v1.json')
    plan=json.loads(plan_path.read_text())
    if a.phase=='final':
        closed=json.loads((ROOT/'reports/training_closed_v1.json').read_text())
        if closed.get('no_further_training') is not True: raise ValueError('Final requires closure')
    sources={r['source_id']:r for r in rows(plan['data'])};names=['adapt','selected_full_s1_f40','selected_reset_s1_f40','uniform_full_s1_f40','uniform_reset_s1_f40']
    successes={};hashes={str(plan_path):sha(plan_path),plan['data']:sha(plan['data']),str(Path(__file__).resolve()):sha(__file__)}
    for name in names:
        spec=plan['models'][name];path=Path(spec['attempts']); hashes[str(path)]=sha(path); successes[name]={}
        for i,row in enumerate(rows(path)):
            if row['reward']==1:
                if not row['ok'] or not row['independent_ok'] or row['reset_index']!=0: raise ValueError('Invalid credited proof')
                item={**row,'original_row_index':i}
                old=successes[name].get(row['source_id'])
                key=lambda x:(x['compression']['compressed_length'],x['n_lines'],x['attempt_index'])
                if old is None or key(item)<key(old): successes[name][row['source_id']]=item
    examples=[];base=set(successes['adapt'])
    for name in names[1:]:
        covered=set(successes[name])
        for kind,pool,winner in [('newly_covered_in_eight_draws',covered-base,name),('lost_coverage_in_eight_draws',base-covered,'adapt')]:
            if not pool:continue
            sid=sorted(pool,key=lambda s:(-sources[s]['n_lines'],s))[0]
            examples.append(dict(model=name,kind=kind,source_id=sid,source=sources[sid],credited_model=winner,attempt=successes[winner][sid]))
    output=ROOT/'reports'/f'{a.phase}_proof_examples_v1.json'
    result=dict(phase=a.phase,selection_rule='For each main final arm and direction versus adapted: largest reference total-line count, then lexicographically smallest source ID; shortest observed compressed valid attempt, then raw length and attempt index.',line_definition='Total ND proof lines, including PR premises; QED is a terminator.',interpretation='Illustrative finite eight-draw coverage changes; absence of a sampled solution is not proof of zero success probability, novel skill, or minimum proof difficulty.',examples=examples,input_sha256=hashes)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(output),'examples':[{k:e[k] for k in ['model','kind','source_id']}|{'reference_lines':e['source']['n_lines'],'generated_pruned_lines':e['attempt']['compression']['compressed_length']} for e in examples]}))

if __name__=='__main__':main()
