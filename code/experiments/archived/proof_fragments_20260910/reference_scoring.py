"""Temperature-explicit reference scoring without changing the frozen sampler."""
import math
import torch
import learning as L


def validate_temperature(value):
    if not math.isfinite(value) or value<=0:
        raise ValueError("Reference-score temperature must be finite and positive")


@torch.no_grad()
def score(model,tok,rows,batch_size=32,ledger=None,precision="fp32",score_temperature=1.):
    validate_temperature(score_temperature)
    if ledger is not None:
        contract={"score_temperature":score_temperature,"unit":"bits",
                  "normalization":"sum over reference action tokens; zero loss on supplied context and padding"}
        if ledger.get("reference_scoring_contract",contract)!=contract:
            raise ValueError("A scoring ledger cannot silently mix temperatures")
        ledger["reference_scoring_contract"]=contract
    if score_temperature==1.:
        result=L.score(model,tok,rows,batch_size,ledger,precision)
    else:
        model.eval();result=[];device=next(model.parameters()).device
        for offset in range(0,len(rows),batch_size):
            chunk=rows[offset:offset+batch_size];batch=L.collate(chunk,tok.pad_id,device)
            chosen=L.logprobs(model,batch,temperature=score_temperature,precision=precision)
            bits=-(chosen*batch["mask"]).sum(-1)/math.log(2)
            if ledger is not None:L.account(ledger,model,batch,"teacher_scoring")
            for row,value in zip(chunk,bits.cpu().tolist()):
                result.append({"source_id":row["source_id"],"family":row["family"],"bits":value,
                               "tokens":len(row["response_ids"]),"bits_per_token":value/len(row["response_ids"]),
                               "reset_index":row["reset_index"],"source_lines":row["source_lines"]})
    for row in result:row["score_temperature"]=score_temperature
    return result
