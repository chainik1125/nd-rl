"""Explicit root-released full-certificate SFT on proposed diagnostic cohorts."""
import argparse
import json
from pathlib import Path
import time
import torch
import common
import learning as L
import nd_environment as E
from nd_model import save_checkpoint
from prepare_prefix_comparison import full_encoding
from retention_only import validate_records
from oracle_sft import bind_population,step_proxy


def validate_release(args,release,proposal_hash,cohort_hash,checkpoint_hash,retention_hash):
    if release.get("frozen") is not True or release.get("approved_for_diagnostic_full_certificate_sft") is not True:
        raise ValueError("Root must explicitly freeze and release diagnostic full-certificate SFT")
    if args.supervision!="full-certificate" or release.get("supervision")!=args.supervision or release.get("full_source_supervision_cap")!=6:
        raise ValueError("Release must explicitly authorize full-certificate cap6 supervision")
    if release.get("proposal_manifest_sha256")!=proposal_hash or cohort_hash not in release.get("cohort_sha256",[]):
        raise ValueError("Proposal/cohort hashes are not released")
    if checkpoint_hash not in release.get("checkpoints",[]) or release.get("retention_sha256")!=retention_hash:
        raise ValueError("Checkpoint/retention identities are not released")
    config=release.get("training_config",{})
    for name in ("steps","seed","lr","batch_size","fixed_pad","retention_fraction","precision"):
        if config.get(name)!=getattr(args,name):raise ValueError(f"Training setting {name} differs from root's common frozen configuration")
    if args.steps<1 or args.batch_size<2 or not 0<args.retention_fraction<1:
        raise ValueError("Positive common training and retention shares are required")


def run(args):
    started=time.time();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError("Diagnostic SFT output must be empty")
    proposal=Path(args.proposal).resolve();proposal_path=proposal/"manifest.json";p=json.loads(proposal_path.read_text())
    if p.get("status")!="proposed_not_released" or p.get("training_authorized") is not False:
        raise ValueError("Expected the preserved proposal; release must be a separate root artifact")
    data=proposal/f"{args.arm}.jsonl";cohort_hash=common.sha(data)
    if p["cohort_sha256"].get(args.arm)!=cohort_hash:raise ValueError("Proposed cohort changed")
    release_path=Path(args.release).resolve();release=json.loads(release_path.read_text())
    retention=bind_population(args.retention,"retention");checkpoint_hash=common.sha(args.checkpoint)
    validate_release(args,release,common.sha(proposal_path),cohort_hash,checkpoint_hash,retention[1])
    if p["input_sha256"].get(str(Path(args.checkpoint).resolve()))!=checkpoint_hash or p["input_sha256"].get(str(retention[0]))!=retention[1]:
        raise ValueError("Training base differs from proposal's common checkpoint/retention")
    # Root's complete release is checked before diagnostic proof payloads.
    records=common.read_jsonl(data);expected_role="diagnostic_failed_rollout" if args.arm=="mined" else "diagnostic_reference_prefix"
    if len(records)!=84 or any(r.get("role")!=expected_role or r.get("comparison_arm")!=args.arm for r in records):
        raise ValueError("Unexpected proposed cohort count/role")
    if [r["paired_index"] for r in records]!=list(range(84)):raise ValueError("Paired source order changed")
    if len({r["source_id"] for r in records})!=84 or len({r["theorem_id"] for r in records})!=84:
        raise ValueError("Duplicate proposed cohort source/endpoint")
    retained=common.read_jsonl(retention[0]);validate_records(retained)
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok,metadata=L.load_model(args.checkpoint,args.device)
    if metadata.get("format")!=E.FORMAT:raise ValueError("Common adapted starting format is required")
    primary=[full_encoding(r,tok) for r in records];retention_rows=L.encode_population(retained,tok,"empty")
    largest=max(len(r["prompt_ids"])+len(r["response_ids"]) for r in primary+retention_rows)
    if not largest<=args.fixed_pad<=model.config.max_seq_len:raise ValueError("Fixed padding cannot fit every full certificate")
    manifest={"command":"diagnostic_certificate_sft","args":vars(args),"format":E.FORMAT,"arm":args.arm,
              "supervision":"full-certificate","supplied_proof_tokens":0,"full_source_supervision_cap":6,
              "original_diagnostic_role_preserved":expected_role,"root_release_is_explicit_training_authority":True,
              "n_primary_sources":84,"n_retention_sources":512,"same_paired_order_as_proposal":True,
              "declared_dense_training_flop_proxy":args.steps*step_proxy(model,args.batch_size,args.fixed_pad),
              "input_sha256":{str(proposal_path):common.sha(proposal_path),str(data):cohort_hash,str(release_path):common.sha(release_path),
                              str(Path(args.checkpoint).resolve()):checkpoint_hash,str(retention[0]):retention[1]},
              "proposal_limitations":p["limitations"],"code_sha256":{str(path):common.sha(path) for path in Path(__file__).parent.glob("*.py")},
              "dependency_sha256":common.dependency_hashes(),"started_unix":started}
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/"manifest.json",manifest)
    common.write_jsonl(output/"source_index.jsonl",[{"source_id":r["source_id"],"theorem_id":r["theorem_id"],"context_id":r["context_id"],
        "original_role":r["role"],"paired_index":r["paired_index"],"n_lines":r["n_lines"],"provenance":r["provenance"]} for r in records])
    ledger={}
    result=L.train_sft(model,tok,primary,retention_rows,steps=args.steps,batch_size=args.batch_size,fixed_pad=args.fixed_pad,
        lr=args.lr,seed=args.seed,ledger=ledger,output=output,retention_fraction=args.retention_fraction,precision=args.precision)
    if L.total_flop_proxy(ledger)!=manifest["declared_dense_training_flop_proxy"]:raise AssertionError("Training compute differs from common fixed padded schedule")
    save_checkpoint(output/"final.pt",model,tok,format=E.FORMAT,control="released_diagnostic_full_certificate_sft",arm=args.arm,
        supervision="full-certificate",full_source_supervision_cap=6,source_checkpoint_sha256=checkpoint_hash,
        proposal_manifest_sha256=common.sha(proposal_path),cohort_sha256=cohort_hash,root_release_sha256=common.sha(release_path),
        original_data_role=expected_role,eligible_evaluation_roles=["confirmation","final_confirmation"])
    result.update({"supervision":"full-certificate","arm":args.arm,"ledger":ledger,"total_flop_proxy":L.total_flop_proxy(ledger),
                   "final_sha256":common.sha(output/"final.pt"),"generation_executed":False,"elapsed_seconds":time.time()-started})
    common.write_json(output/"result.json",result);common.write_json(output/"completed.json",{"ok":True,"result_sha256":common.sha(output/"result.json"),"final_sha256":result["final_sha256"]})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("checkpoint","retention","proposal","release","output"):p.add_argument("--"+flag,required=True)
    p.add_argument("--arm",choices=("mined","reference"),required=True)
    p.add_argument("--supervision",choices=("full-certificate",),required=True)
    p.add_argument("--steps",type=int,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--lr",type=float,default=5e-5);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--fixed-pad",type=int,default=512)
    p.add_argument("--retention-fraction",type=float,default=.5);p.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--device",default="cpu");p.add_argument("--cpu-threads",type=int,default=1)
    run(p.parse_args())
