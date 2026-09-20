"""Verify declared training completion and record global closure without reading final data."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    plan_path=Path(sys.argv[1]); plan=json.loads(plan_path.read_text())
    if plan.get('no_further_training_after_closure') is not True: raise ValueError('Global training closure is required')
    receipts={}
    for directory in plan['declared_training_outputs']:
        p=Path(directory); done=json.loads((p/'completed.json').read_text()); result=json.loads((p/'result.json').read_text())
        if done.get('ok') is not True or done.get('result_sha256') != sha(p/'result.json'): raise ValueError('Training completion mismatch: '+directory)
        if result.get('final_sha256') and result['final_sha256'] != sha(p/'final.pt'): raise ValueError('Final training weights differ')
        receipts[directory]={'completed_sha256':sha(p/'completed.json'),'result_sha256':sha(p/'result.json'),'manifest_sha256':sha(p/'manifest.json')}
    checkpoints={p:sha(p) for p in list(plan['final_evaluation_models'].values())+plan['calibration_score_only']}
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_further_training=True,no_further_selector_changes=True,final_payload_opened=False,plan_sha256=sha(plan_path),training_outputs=receipts,checkpoint_sha256=checkpoints,main_provenance_sha256=sha(plan['main_provenance']),cancelled_unrun=plan['unrun_cancelled_for_this_sprint'])
    with Path(plan['closure_output']).open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps({'training_closed':True,'checkpoints':len(checkpoints),'output':plan['closure_output']}),flush=True)

if __name__ == '__main__': main()
