"""Summarize the shortest observed checked certificate for every covered target."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','final'],required=True);a=p.parse_args()
    plan_path=ROOT/'configs'/('analysis_main_development_seed1_v1.json' if a.phase=='development' else 'analysis_final_seed1_v1.json');plan=json.loads(plan_path.read_text())
    if a.phase=='final' and json.loads((ROOT/'reports/training_closed_v1.json').read_text()).get('no_further_training') is not True:raise ValueError('Final closure required')
    sources=rows(plan['data']);by_id={r['source_id']:r for r in sources};hashes={str(plan_path):sha(plan_path),plan['data']:sha(plan['data']),str(Path(__file__).resolve()):sha(__file__)}
    models={}
    for name,spec in plan['models'].items():
        if not spec.get('attempts'):continue
        path=Path(spec['attempts']);done=json.loads((path.parent/'completed.json').read_text())
        if not done.get('ok') or done['result_sha256']!=sha(path.parent/'result.json'):raise ValueError('Incomplete result '+name)
        attempts=rows(path);hist=Counter();best={};counts=Counter();maxraw=0
        for r in attempts:
            counts[r['source_id']]+=1
            if r['reward']:
                if not r['ok'] or not r['independent_ok'] or r['reset_index']!=0:raise ValueError('Invalid credited proof')
                n=r['compression']['compressed_length'];hist[n]+=1;maxraw=max(maxraw,r['n_lines']);sid=r['source_id'];best[sid]=min(best.get(sid,n),n)
        if set(counts)!=set(by_id) or set(counts.values())!={plan['samples']}:raise ValueError('Sampling denominator differs')
        bins={}
        for label in ['2-6','7-9','10-12']:
            ids={r['source_id'] for r in sources if r['length_bin']==label};solved=ids&best.keys()
            bins[label]={'targets':len(ids),'covered':len(solved),'covered_with_observed_certificate_at_most_six':sum(best[s]<=6 for s in solved),'covered_best_observed_certificate_above_six':sum(best[s]>6 for s in solved),'best_observed_certificate_histogram':dict(sorted(Counter(best[s] for s in solved).items()))}
        long={sid for sid in best if by_id[sid]['n_lines']>6}
        models[name]={'covered':len(best),'by_reference_length_bin':bins,'maximum_solved_reference_lines':max((by_id[s]['n_lines'] for s in best),default=None),'maximum_valid_written_lines':maxraw or None,'maximum_best_observed_pruned_lines':max(best.values(),default=None),'all_successful_attempt_pruned_histogram':dict(sorted(hist.items())),'covered_long_reference_targets':len(long),'long_reference_targets_with_observed_cap6_certificate':sum(best[s]<=6 for s in long),'best_observed_certificate_by_source':best}
        hashes[str(path)]=sha(path)
    result={'phase':a.phase,'line_definition':'Total ND proof body lines, including PR premises and AS assumptions. QED/PRF/FINAL markers do not count.','interpretation':'Minimum across observed successful attempts is an upper bound on minimum proof length. Reference proofs are also witness upper bounds. A longer observed certificate does not prove a longer certificate is required.','models':models,'input_sha256':hashes}
    target=ROOT/'reports'/f'{a.phase}_certificate_lengths_v1.json'
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(target),'models':len(models)}))

if __name__=='__main__':main()
