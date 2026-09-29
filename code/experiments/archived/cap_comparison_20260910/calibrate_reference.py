import hashlib,json,random,sys,time
from pathlib import Path
import importlib.util
import torch
torch.set_num_threads(1)
OLD=Path('/data/users/dmitry/nd-rl/r1-sprint-20260909-1625')
NEW=Path('/data/users/dmitry/nd-rl/cap-comparison-20260910-1958')
sys.path[:0]=[str(OLD/'snapshots/baseline'),str(OLD/'snapshots/baseline/vendor')]
spec=importlib.util.spec_from_file_location('calibration_old',OLD/'snapshots/baseline/run.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
spec=importlib.util.spec_from_file_location('calibration_new',NEW/'code/run.py');new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
sources={};results={};started=time.monotonic()
def capture(p):
 p=Path(p);raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return raw
def jread(p):return json.loads(capture(p))
def jrows(p):return [json.loads(x) for x in capture(p).splitlines() if x.strip()]
def emit():
 value={'status':'historical_training_only_proxy_reconstruction','version':new.COMPUTE_PROXY_VERSION,'note':new.COMPUTE_PROXY_NOTE,'results':results,'source_sha256':sources,'elapsed_seconds':time.monotonic()-started,'no_textbook_development_or_final_access':True}
 Path('/tmp/cap_compute_reference.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
for mode,name in [('cot','fresh8m_cap6_dependency'),('direct','fresh8m_cap6_direct')]:
 sm=jread(OLD/'runs'/f'{name}_sft/manifest.json');sa=sm['args'];events=jrows(OLD/'runs'/f'{name}_sft/events.jsonl');data=jrows(sa['data']);assert sources[sa['data']]==sm['data_sha256']
 tok=core.ScratchTokenizer();m=core.ProofModel(core.ModelConfig(**sm['config']));encoded=[]
 for row in data:
  e=core.encode_training(row,tok,mode,sm['config']['max_seq_len'],sa['max_proof_lines'],sa['scratch_style']);encoded.append((len(e['prompt_ids']),len(e['response_ids'])))
 ledger=new.ComputeLedger(m);rng=random.Random(sa['seed']);targets=0;sft={}
 for step in range(1,sa['steps']+1):
  batch=[rng.choice(encoded) for _ in range(sa['batch_size'])];targets+=sum(x[1] for x in batch)
  for offset in range(0,len(batch),sa['microbatch']):
   chunk=batch[offset:offset+sa['microbatch']];ledger.account('sft',len(chunk),max(a+b-1 for a,b in chunk),backward=True)
  if step in (4000,8000):sft[str(step)]={'total_flop_proxy':ledger.total,'actual_response_tokens':targets,'compute':ledger.snapshot()}
 assert targets==events[-1]['actual_response_tokens']
 assert sft['4000']['actual_response_tokens']==next(x['actual_response_tokens'] for x in events if x['step']==4000)
 results[mode]={'parameters':ledger.parameters,'sft':sft};print(mode,'SFT',[(k,v['total_flop_proxy'],v['actual_response_tokens']) for k,v in sft.items()],flush=True);emit()
 rm=jread(OLD/'runs'/f'{name}_rl/manifest.json');ra=rm['args'];revents=jrows(OLD/'runs'/f'{name}_rl/events.jsonl');assert len(revents)==ra['steps']==512
 pool=jrows(ra['data']);retention=jrows(ra['retention_data']);assert sources[ra['data']]==rm['data_sha256'];assert sources[ra['retention_data']]==rm['retention_data_sha256']
 retained_lengths=[]
 for row in retention:
  e=core.encode_training(row,tok,mode,rm['config']['max_seq_len'],ra['max_proof_lines'],ra['scratch_style']);retained_lengths.append((len(e['prompt_ids']),len(e['response_ids'])))
 ledger=new.ComputeLedger(m);rng=random.Random(ra['seed']);buffer=[];generated=0;trace=OLD/'runs'/f'{name}_rl/rollouts.jsonl';digest=hashlib.sha256();processed=0
 with trace.open('rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);buffer.append(row)
   if len(buffer)<ra['prompts_per_step']*ra['group_size']:continue
   step=processed+1;assert all(r['step']==step for r in buffer)
   selected=[core.target_prompt(rng.choice(pool)) for _ in range(ra['prompts_per_step'])]
   assert [r['prompt'] for r in buffer]==[p for p in selected for _ in range(ra['group_size'])]
   retained=[rng.choice(retained_lengths) for _ in range(ra['retention_batch'])]
   assert sum(b for a,b in retained)*ra['policy_epochs']==revents[processed]['retention_response_tokens']
   order=sorted(range(len(buffer)),key=lambda i:len(buffer[i]['prompt_ids']))
   for offset in range(0,len(order),ra['decode_batch']):
    chunk=[buffer[i] for i in order[offset:offset+ra['decode_batch']]];width=max(len(r['prompt_ids']) for r in chunk);iterations=max(len(r['response_ids']) for r in chunk);ledger.account('generation_prefill',len(chunk),width)
    for call in range(1,iterations):ledger.account('generation_decode',len(chunk),1,width+call)
   for offset in range(0,len(buffer),ra['microbatch']):
    chunk=buffer[offset:offset+ra['microbatch']];width=max(len(r['prompt_ids'])+len(r['response_ids'])-1 for r in chunk);ledger.account('behavior_teacher_audit',len(chunk),width)
    for epoch in range(ra['policy_epochs']):ledger.account('policy',len(chunk),width,backward=True)
   if ra['retention_weight']:
    for epoch in range(ra['policy_epochs']):
     for offset in range(0,len(retained),ra['microbatch']):
      chunk=retained[offset:offset+ra['microbatch']];ledger.account('retention',len(chunk),max(a+b-1 for a,b in chunk),backward=True)
   generated+=sum(len(r['response_ids']) for r in buffer);assert generated==revents[processed]['actual_response_tokens'];processed+=1;buffer=[]
 assert processed==512 and not buffer;sources[str(trace)]=digest.hexdigest();results[mode]['rl512']={'total_flop_proxy':ledger.total,'actual_response_tokens':generated,'compute':ledger.snapshot(),'multiplied_8_budget':8*ledger.total};print(mode,'RL512',ledger.total,'8x',8*ledger.total,'response',generated,flush=True);emit()
for path,digest in sources.items():assert core.sha256(path)==digest
print('COMPLETE /tmp/cap_compute_reference.json',time.monotonic()-started,flush=True)
