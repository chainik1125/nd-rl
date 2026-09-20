"""Frozen selection versus fresh final family utility, with target-context sensitivity."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
import analyze_evaluations as A

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plan_path=ROOT/'configs/analysis_final_calibration_v1.json';plan=json.loads(plan_path.read_text())
    final_path=ROOT/'reports/final_calibration_v1.json';final=json.loads(final_path.read_text())
    ranking_path=ROOT/'results/calibration_v1/ranking.json';old=json.loads(ranking_path.read_text())
    initial={r['family']:r for r in old['transfer']};names=sorted(initial)
    rows=A.read_rows(plan['data']);ids=[r['source_id'] for r in rows]
    weights=A.cluster_weights(rows,5000,20260911);ys=[];boots=[];families={}
    for name in names:
        control,_=A.ordered_scores(plan['models'][name+'_control']['nll'],ids)
        family,_=A.ordered_scores(plan['models'][name+'_family']['nll'],ids)
        result,boot=A.summarize(control-family,rows,weights); y=result['balanced']['mean'];ys.append(y);boots.append(boot)
        if abs(y-final['comparisons'][name+'_control_minus_'+name+'_family']['whole_proof_bits']['balanced']['mean'])>1e-10:raise ValueError('Final utility aggregation differs')
        families[name]={'selection_utility_bits':initial[name]['utility_bits'],'final_utility_bits':y,'final_context_bootstrap95':result['balanced']['context_bootstrap95']}
    x=np.array([initial[n]['utility_bits'] for n in names]);y=np.array(ys);matrix=np.stack(boots,axis=1)
    def correlations(values,reference):
        centered=values-values.mean(axis=1,keepdims=True);r=reference-reference.mean();return (centered@r)/(np.linalg.norm(centered,axis=1)*np.linalg.norm(r))
    rho_boot=correlations(rankdata(matrix,axis=1),rankdata(x));r_boot=correlations(matrix,x)
    top=['NEGI','IMPI_flat','NEGE']
    report={'families':families,'frozen_top3':top,'selection_vs_final_pearson':float(pearsonr(x,y).statistic),'selection_vs_final_spearman':float(spearmanr(x,y).statistic),'sign_agreement':int(np.sum(np.sign(x)==np.sign(y))),'families_count':len(names),'pearson_final_context_bootstrap95':A.interval(r_boot),'spearman_final_context_bootstrap95':A.interval(rho_boot),'frozen_top3_mean_final_utility':final['contrasts']['frozen_top3_average_utility']['metrics']['whole_proof_bits']['balanced'],'uncertainty_scope':'Final target-context resampling only, conditional on the twelve fixed family/control interventions, training seed and original selection estimates. No training-seed or new-family uncertainty; no multiplicity correction.','input_sha256':{str(p):sha(p) for p in [plan_path,final_path,ranking_path,Path(__file__).resolve(),ROOT/'tools/analyze_evaluations.py']}}
    out=ROOT/'reports/final_calibration_ranking_v1.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['selection_vs_final_pearson','selection_vs_final_spearman','sign_agreement','spearman_final_context_bootstrap95','frozen_top3_mean_final_utility']}))

if __name__=='__main__':main()
