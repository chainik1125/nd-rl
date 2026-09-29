"""Render portable figures and a report from strictly replayed results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from analyze import read, sha, write

COLORS = {"direct": "#2563a6", "cot": "#ae3f73"}
CAP_COLORS = {6: "#2563a6", 10: "#cd6b18", 12: "#238b69"}


def save(fig, root, name):
    fig.savefig(root / "reports" / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(root / "reports" / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered(results):
    return sorted(results["arms"].items(), key=lambda pair: (pair[1]["mode"] != "direct", pair[1]["cap"]))


def coverage_figures(root, results):
    for cohort, name, subtitle in ((None, "textbook_coverage", "All 58 development problems"),
                                   ("6", "textbook_beyond_six", "Same 43 problems with reference proofs longer than 6 lines"),
                                   ("12", "textbook_long_coverage", "Same 23 problems with reference proofs longer than 12 lines")):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
        all_counts = []
        for ax, procedure, title in zip(axes, ("greedy", "sampled"), ("One greedy attempt", "At least one success in 16 samples")):
            for mode in ("direct", "cot"):
                arms = [arm for _,arm in ordered(results) if arm["mode"] == mode]
                if not arms:
                    continue
                for phase, style in (("sft", "--"), ("rl", "-")):
                    points = [arm["endpoints"][f"{phase}_{procedure}"] for arm in arms]
                    counts = [(p["cohorts_above_reference"][cohort] if cohort else p)["solved"] for p in points]
                    all_counts.extend(counts)
                    ax.plot([arm["cap"] for arm in arms], counts, marker="s" if mode=="direct" else "o", linestyle=style,
                            markersize=9 if phase=="sft" else 5,
                            markerfacecolor="none" if phase=="sft" else COLORS[mode],
                            color=COLORS[mode], label=f"{'Direct' if mode == 'direct' else 'CoT'} · {phase.upper()}")
                    for arm, count in zip(arms, counts):
                        ax.annotate(str(count), (arm["cap"], count), xytext=(-9 if mode == "direct" else 9, 7 if phase == "rl" else -13),
                                    textcoords="offset points", ha="center", color=COLORS[mode], fontsize=8)
            ax.set(title=title, xlabel="Initial supervised proof cap (lines)", xticks=[6,10,12], xlim=(5.5,12.5))
            ax.grid(alpha=.2)
        maximum = max(all_counts, default=0)
        axes[0].set_ylim(-.5, max(1, maximum) + max(1, maximum * .12))
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylabel("Textbook problems solved")
        axes[1].legend(fontsize=8)
        fig.suptitle(subtitle + (" — PARTIAL" if not results["complete"] else ""))
        fig.text(.5, -.01, "One seed per run. Squares: Direct; circles: CoT. Open/dashed: SFT; filled/solid: full-budget RL. No smoothing.", ha="center", fontsize=9)
        fig.tight_layout()
        save(fig, root, name)


def length_figure(root, results):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for row, (metric, label) in enumerate((("max_solved_reference", "Longest solved reference (lines)"),
                                          ("max_shortest_compressed_witness", "Longest shortest observed\ncompressed witness (lines)"))):
        all_lengths = [arm["endpoints"][phase+"_"+procedure][metric] for _,arm in ordered(results)
                       for phase in ("sft","rl") for procedure in ("greedy","sampled")]
        maximum = max((n for n in all_lengths if n is not None), default=0)
        for col, (procedure, title) in enumerate((("greedy", "One greedy attempt"), ("sampled", "Up to 16 samples"))):
            ax=axes[row,col]
            for mode in ("direct", "cot"):
                arms=[arm for _,arm in ordered(results) if arm["mode"]==mode]
                if not arms:continue
                for phase, style in (("sft","--"),("rl","-")):
                    values=[arm["endpoints"][phase+"_"+procedure][metric] for arm in arms]
                    ax.plot([arm["cap"] for arm in arms],values,marker="s" if mode=="direct" else "o",
                            markersize=9 if phase=="sft" else 5,
                            markerfacecolor="none" if phase=="sft" else COLORS[mode],linestyle=style,color=COLORS[mode],
                            label=f"{'Direct' if mode=='direct' else 'CoT'} · {phase.upper()}")
            ax.set(xlabel="Initial supervised proof cap (lines)",xticks=[6,10,12],xlim=(5.5,12.5))
            ax.set_ylim(0, max(1, maximum) + max(1, maximum * .12))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            if row==0:ax.set_title(title)
            if col==0:ax.set_ylabel(label)
    axes[0,1].legend(fontsize=8)
    fig.suptitle("Textbook proof-length reach"+(" — PARTIAL" if not results["complete"] else ""))
    fig.text(.5,-.01,"Reference and generated lengths are witness upper bounds, not minimum difficulty. Failed proofs never count.",ha="center",fontsize=9)
    fig.tight_layout()
    save(fig,root,"textbook_proof_lengths")


def probe_records(root, results):
    protocol = read(root / "plans/protocol.json")
    records = []
    for name, arm in ordered(results):
        spec = read(root / "plans" / f"{name}.json")
        base = root / "runs" / name
        expected_positions = {
            "sft": {"checkpoint_sha256": arm["training"]["sft"]["checkpoint_sha256"], "step": 0, "total_flop_proxy": 0},
            "rl": {"checkpoint_sha256": arm["training"]["rl"]["checkpoint_sha256"],
                   "step": arm["training"]["rl"]["actual_steps"], "total_flop_proxy": arm["training"]["rl"]["actual_flop_proxy"]}}
        for index, point in enumerate(arm["training"]["rl"]["milestones"]):
            if point["requested_flop_proxy"] < arm["training"]["rl"]["target_flop_proxy"]:
                expected_positions[f"rl_m{index:02d}"] = {"checkpoint_sha256": point["checkpoint_sha256"],
                    "step": point["step"], "total_flop_proxy": point["actual_flop_proxy"]}
        for split in ("pretraining", "validation_synthetic"):
            expected_n = 512 if split == "pretraining" else 256
            expected_sha = protocol["input_sha256"][spec["data"][split]]
            for phase in ("sft", "rl_m00", "rl_m01", "rl_m02", "rl"):
                directory = base / "evaluations" / f"{phase}_{split}"
                summary, manifest, position = [read(directory / f) for f in ("summary.json", "manifest.json", "checkpoint_position.json")]
                if (summary["data_sha256"] != expected_sha or manifest["data_sha256"] != expected_sha
                        or summary["problems"] != expected_n or summary["samples_per_problem"] != 1
                        or manifest["args"]["temperature"] != 0
                        or position != expected_positions[phase]
                        or summary["checkpoint_sha256"] != position["checkpoint_sha256"]):
                    raise ValueError("Probe dataset/checkpoint/procedure identity differs")
                records.append({"arm": name, "cap": arm["cap"], "mode": arm["mode"], "phase": phase,
                                "split": split, **position, "solved": summary["solved"], "problems": expected_n,
                                "accuracy": summary["solved"] / expected_n,
                                "data_sha256": expected_sha})
    return records


def probe_figure(root, results, records):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, split, title in zip(axes, ("pretraining", "validation_synthetic"),
                                ("Common short SFT probe (512; 2–6 lines)", "Short synthetic validation (256; 2–6 lines)")):
        for name, arm in ordered(results):
            points = sorted([p for p in records if p["arm"] == name and p["split"] == split], key=lambda p:p["total_flop_proxy"])
            ax.plot([p["total_flop_proxy"] / 1e15 for p in points], [100*p["accuracy"] for p in points],
                    marker="s" if arm["mode"] == "direct" else "o", markersize=4, color=CAP_COLORS[arm["cap"]], linestyle="-" if arm["mode"] == "direct" else "--",
                    label=f"{'Direct' if arm['mode'] == 'direct' else 'CoT'} cap {arm['cap']}")
        ax.set(title=title, xlabel="RL compute proxy since SFT (10¹⁵ FLOPs)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Greedy proof accuracy (%)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.text(.5, -.01, "Each point evaluates a saved checkpoint. Lines connect only the same run. SFT work is additional (25.81 × 10¹⁵ proxy FLOPs).", ha="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, root, "training_accuracy")


def inference_figure(root, results):
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax, phase in zip(axes,("sft","rl")):
        for _,arm in ordered(results):
            points = arm["endpoints"][phase+"_sampled"]["prefix_tokens"]
            budgets = sorted(map(int,points))
            ax.plot(budgets,[points[str(b)]["solved"] for b in budgets],marker="s" if arm["mode"] == "direct" else "o",markersize=3,
                    color=CAP_COLORS[arm["cap"]],linestyle="-" if arm["mode"]=="direct" else "--",
                    label=f"{'Direct' if arm['mode']=='direct' else 'CoT'} cap {arm['cap']}")
        ax.set(xscale="log",title=phase.upper(),xlabel="Generated-token budget per textbook problem")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Textbook problems solved (out of 58)")
    axes[1].legend(fontsize=8,ncol=2)
    fig.text(.5,-.01,"Saved samples in fixed order, up to 16. All scratch/failure tokens charged; unfinished attempts do not count as successes.",ha="center",fontsize=8.5)
    fig.tight_layout()
    save(fig,root,"inference_budget")


def family_figure(root, results):
    arms = ordered(results)
    records = {}
    for name, arm in arms:
        record = read(root / "reports" / f"{name}_training_information.json")
        if record["rl_checkpoint_sha256"] != arm["training"]["rl"]["checkpoint_sha256"]:
            raise ValueError("Family diagnostic checkpoint differs")
        records[name] = record["families"]
    families = sorted({family for record in records.values() for family in record if family.startswith("long/")})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, metric, title in zip(axes, ("success", "mixed"), ("Successful training samples (%)", "Groups with both success and failure (%)")):
        values = []
        for family in families:
            row = []
            for name, _ in arms:
                entry = records[name].get(family)
                row.append(float("nan") if entry is None else 100 * (entry["correct_responses"] / entry["responses"]
                           if metric == "success" else entry["mixed_group_fraction"]))
            values.append(row)
        plot = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.1f}" if value == value else "—", ha="center", va="center",
                        color="white" if value > 55 else "#15253b", fontsize=9)
        labels = [f"{'Direct' if arm['mode']=='direct' else 'CoT'}\ncap {arm['cap']}" for _, arm in arms]
        ax.set(title=title, xticks=range(len(arms)), xticklabels=labels,
               yticks=range(len(families)), yticklabels=[family.split("/", 1)[1].replace("_", " ") for family in families])
        fig.colorbar(plot, ax=ax, shrink=.85, ticks=[0,25,50,75,100])
    fig.suptitle("Long synthetic proof families — training diagnostics")
    fig.text(.5, -.01, "Training samples only. Each column is a separate completed run; family names identify synthetic generators.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, root, "training_families")


def report(root, results, probes):
    title = "Proof cap comparison" + (" — partial results" if not results["complete"] else "")
    lines = [f"# {title}", "", f"Generated {datetime.now(timezone.utc).isoformat()}. Branch `experiment/cap-comparison`.", "",
             f"Completed configurations: {len(results['arms'])}/6. One seed (20260910) per configuration. "
             "The benchmark is the existing **58-problem adaptive textbook development set**. No old final-test problems were evaluated for this comparison.", "",
             "## Textbook results", "", "Every textbook count below was reproduced by strict replay of the saved final proof against its original theorem. "
             "SFT and RL use the same evaluation procedure; each textbook evaluation started only after that arm completed its full RL budget.", "",
             "| Model | Cap | Greedy SFT → RL (/58) | Sample16 SFT → RL (/58) | RL greedy max reference | RL sample16 max reference |",
             "|---|---:|---:|---:|---:|---:|"]
    interpretation_path = root / "reports/interpretation.json"
    if interpretation_path.exists():
        interpretation = read(interpretation_path)
        if not results["complete"] or interpretation["analysis_results_sha256"] != sha(root / "reports/analysis_results.json"):
            raise ValueError("Authored interpretation requires its reviewed, complete analysis")
        position = lines.index("## Textbook results")
        narrative = [line for paragraph in interpretation["paragraphs"] for line in (paragraph, "")]
        lines[position:position] = narrative
    for _, arm in ordered(results):
        e = arm["endpoints"]
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {e['sft_greedy']['solved']} → {e['rl_greedy']['solved']} | {e['sft_sampled']['solved']} → {e['rl_sampled']['solved']} | {e['rl_greedy']['max_solved_reference']} | {e['rl_sampled']['max_solved_reference']} |")
    lines += ["", "![Textbook coverage](textbook_coverage.png)", "", "![Textbook proof-length reach](textbook_proof_lengths.png)", "", "## Longer textbook problems", "",
              "Reference lengths are supplied witness lengths, not minimum proof difficulty. Every configuration is compared on the same cohorts: 43 problems above six lines and 23 above twelve. The common >10 cohort and own-cap counts are also retained in `analysis_results.json`.", "",
              "| Model | Cap | Common >6, greedy SFT → RL (/43) | Common >6, sample16 SFT → RL (/43) | Common >12, greedy SFT → RL (/23) | Common >12, sample16 SFT → RL (/23) |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, arm in ordered(results):
        e = arm["endpoints"]
        six = [f"{e['sft_'+p]['cohorts_above_reference']['6']['solved']} → {e['rl_'+p]['cohorts_above_reference']['6']['solved']}" for p in ("greedy","sampled")]
        common = [f"{e['sft_'+p]['cohorts_above_reference']['12']['solved']} → {e['rl_'+p]['cohorts_above_reference']['12']['solved']}" for p in ("greedy","sampled")]
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {six[0]} | {six[1]} | {common[0]} | {common[1]} |")
    lines += ["", "![Coverage on the same problems above six lines](textbook_beyond_six.png)", "",
              "![Coverage on the same problems above twelve lines](textbook_long_coverage.png)", "",
              "| Model | Cap | RL greedy max shortest raw / compressed witness | RL sample16 max shortest raw / compressed witness |",
              "|---|---:|---:|---:|"]
    for _,arm in ordered(results):
        values = [arm["endpoints"]["rl_"+p] for p in ("greedy","sampled")]
        counts = [f"{v['max_shortest_raw_witness']} / {v['max_shortest_compressed_witness']}" for v in values]
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {counts[0]} | {counts[1]} |")
    lines += ["", "For each solved problem, take its shortest observed accepted witness, then report the maximum across solved problems. "
              "Compressed certificates are reverified; compression cannot rescue a failed final proof. These remain upper bounds, not minimum lengths.", "",
              "## Paired RL changes", "", "| Model | Cap | Greedy Δ solved; 95% interval (percentage points) | Sample16 Δ solved; 95% interval (percentage points) |", "|---|---:|---:|---:|"]
    for _,arm in ordered(results):
        cells=[]
        for p in ("greedy","sampled"):
            c=arm["contrasts"][p]["full"]; low,high=c["paired_target_bootstrap95"]
            cells.append(f"{c['delta_solved']:+d}; [{low*100:+.1f}, {high*100:+.1f}]")
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {cells[0]} | {cells[1]} |")
    lines += ["", "Intervals use 5,000 paired target bootstrap resamples, conditional on the saved attempts and one training seed. "
              "They do not measure seed variation. Prespecified K=1/4/16 and token-budget contrasts are retained in `analysis_results.json`.", "",
              "![Inference budget comparison](inference_budget.png)", "",
              "| Model | Cap | RL greedy token-limit attempts | RL sampled token-limit attempts |",
              "|---|---:|---:|---:|"]
    for _,arm in ordered(results):
        g,s=[arm["endpoints"]["rl_"+p] for p in ("greedy","sampled")]
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {g['truncated_attempts']}/{g['strict_replayed_attempts']} | {s['truncated_attempts']}/{s['strict_replayed_attempts']} |")
    lines += ["", "The 1,536-token limit applies to each attempt, including scratch. Larger total token budgets in the graph mean more saved independent attempts; they do not extend the maximum length of an individual attempt.", "",
              "## Training accuracy and compute", "",
              "![Fixed dataset accuracy by RL compute](training_accuracy.png)", "",
              "| Model | Cap | SFT updates | RL updates | SFT response tokens (M) | RL generated tokens (M) | SFT work (10¹⁵ proxy FLOPs) | RL work (10¹⁵ proxy FLOPs) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,arm in ordered(results):
        s,r=arm["training"]["sft"],arm["training"]["rl"]
        tokens = []
        for phase, record in (("sft",s),("rl",r)):
            completion = root / "runs" / name / phase / "completed.json"
            if sha(completion) != record["completed_sha256"]:
                raise ValueError("Token-count source differs from completed training")
            tokens.append(read(completion)["actual_response_tokens"])
        lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {s['actual_steps']:,} | {r['actual_steps']:,} | {tokens[0]/1e6:.2f} | {tokens[1]/1e6:.2f} | {s['actual_flop_proxy']/1e15:.4f} | {r['actual_flop_proxy']/1e15:.4f} |")
    lines += ["", "Every arm uses the same SFT target (25,807,392,746,127,360 proxy FLOPs) and RL target "
              "(14,663,886,875,197,440). Training stops after the first completed update reaching each target. "
              "The proxy charges actual padded model forwards, backward passes, generation, likelihood checks and retention; it is not measured hardware FLOPs. "
              "SFT token counts measure response-label exposure; RL token counts include generated scratch and failed attempts. They exclude prompt/padding positions and additional retention-label exposure, which are charged in the compute proxy. "
              "SFT work is additional to the RL x-axis. Validation and pretraining points use the identical fixed 256/512 theorem cohorts, respectively. Both contain reference proofs of 2–6 lines, so these curves measure short-proof competence; they do not test longer-proof transfer. The pretraining probe covers common short examples, not the full SFT pool. See `fixed_probe_length_audit.json`.", "",
              "## Observed RL learning signal", "",
              "A group has eight responses to the same training theorem. With leave-one-out advantages, groups that all pass or all fail supply no policy reward contrast. "
              "The table separates training sources; these are training diagnostics, not validation scores. Retention updates can still occur when policy advantages are zero. "
              "An aggregate pass rate can mix theorems that always succeed with theorems that always fail; it does not establish that individual theorems produce useful reward contrasts.", "",
              "| Model | Cap | Training source | Sample success | Mixed groups | Tokens with nonzero advantage |", "|---|---:|---|---:|---:|---:|"]
    for name,arm in ordered(results):
        path=root / "reports" / f"{name}_training_information.json"
        if not path.exists():continue
        information=read(path)
        if information["rl_checkpoint_sha256"] != arm["training"]["rl"]["checkpoint_sha256"]:
            raise ValueError("Training information checkpoint differs")
        for bucket,c in sorted(information["buckets"].items()):
            source={"book":"14 permitted textbook training prompts","long":"Long synthetic RL prompts","short":"Short synthetic RL prompts"}.get(bucket,bucket)
            lines.append(f"| {arm['mode'].upper()} | {arm['cap']} | {source} | {100*c['correct_responses']/c['responses']:.1f}% | {100*c['mixed_group_fraction']:.1f}% | {100*c['signal_token_fraction']:.1f}% |")
    lines += ["", "![Learning signal by long synthetic generator family](training_families.png)", "",
              "These family statistics cover the observed training rollouts over the full RL run. They are not held-out accuracy, fragment transfer utility, or an EDL estimate. No dedicated held-out long synthetic probe is included in this comparison."]
    growth_rows=[]
    for name, arm in ordered(results):
        path=root / "reports" / f"{name}_training_growth.json"
        if not path.exists():continue
        growth=read(path)
        if growth["rl_checkpoint_sha256"]!=arm["training"]["rl"]["checkpoint_sha256"]:
            raise ValueError("Training growth checkpoint differs")
        buckets=growth["unique_theorems_by_sampling_bucket"]
        example=growth["largest_compressed_above_cap_example_by_bucket"].get("long")
        growth_rows.append(f"| [{'Direct' if arm['mode']=='direct' else 'CoT'} cap {arm['cap']}]({path.name}) | {growth['maximum_raw_training_witness']} | {buckets.get('long',0)} | {example['compressed_length'] if example else '—'} | {buckets.get('short',0)} |")
    if growth_rows:
        lines += ["", "## Training certificate lengths", "",
                  "This posthoc audit replays and compresses accepted training rollouts whose raw certificates exceed their own initial cap. RL-only synthetic prompts are disjoint from SFT but are repeatedly used during RL; they are not held-out problems.", "",
                  "| Configuration / audit | Raw training maximum | RL-only synthetic theorems with compressed certificate above cap | Longest such compressed certificate | SFT-exposed short theorems also producing above-cap certificates |",
                  "|---|---:|---:|---:|---:|", *growth_rows, "",
                  "These count observed certificates, not the minimum lines required by their theorems. A [strictly verified counterexample](training_length_counterexample.json) shows a seven-line compressed model certificate with a six-line alternative: citing an existing assumption directly removes an unnecessary derivation. The textbook coverage and witness results remain the primary transfer measurements."]
    lines += ["", "## Inspect generated proofs", "",
              "The examples below show saved final proofs and, for CoT models, their symbolic scratchpads. They are posthoc illustrations selected by a fixed rule, not additional evaluations or evidence replacing the aggregate counts.", ""]
    for name, arm in ordered(results):
        filename=f"{name}_examples.md"
        if (root / "reports" / filename).exists():
            lines.append(f"- [{'Direct' if arm['mode']=='direct' else 'CoT'} cap {arm['cap']} examples]({filename})")
    paired_example=root / "reports/cap10_direct_long_textbook_example.md"
    if paired_example.exists():
        lines.append("- [Cap-10 Direct's seven-line disjunction-commutativity proof, already present after SFT](cap10_direct_long_textbook_example.md)")
    diagnostic=root / "reports/failure_diagnostics.json"
    if diagnostic.exists() and read(diagnostic)["analysis_results_sha256"]==sha(root / "reports/analysis_results.json"):
        lines += ["", "[Failure diagnostics](failure_diagnostics.json) describe verifier errors, termination reasons and observed proof-text diversity for every completed run."]
    lines += ["", "## Execution cost", "",
              "| Configuration | Host / GPU | Pipeline wall minutes (SFT + RL + evaluations) |", "|---|---|---:|"]
    state_path=root / "reports/recovery_state.json"
    state=read(state_path) if state_path.exists() else {"assignments":{}}
    for name,arm in ordered(results):
        assignment=state["assignments"].get(name,{})
        progress=read(root / "runs" / name / "progress.json")
        lines.append(f"| {arm['mode'].upper()} cap {arm['cap']} | {assignment.get('host','unknown')} / {assignment.get('gpu','unknown')} | {progress['elapsed_seconds']/60:.1f} |")
    receipt=read(root / "reports/recovery_protocol.json")
    interrupted_seconds=sum(item["result"]["seconds"] for item in receipt["interrupted"].values())
    successful_seconds=sum(read(root / "runs" / name / "progress.json")["elapsed_seconds"] for name in results["arms"])
    lines += ["", f"The {len(receipt['interrupted'])} interrupted attempts consumed {interrupted_seconds/3600:.2f} GPU-hours of pipeline occupancy. "
              f"Completed comparison arms consumed {successful_seconds/3600:.2f} GPU-hours. These measure wall time holding one GPU, not full GPU utilization. "
              "Queue time is excluded; the separate implementation smoke test is excluded. The planned six configurations use one seed each; infrastructure retries are not additional seeds.", "",
              "## Scope and limitations", "",
              "- The architecture, initialization, seed, RL prompt pool, decoding limit and compute budgets are shared. The 8.87M-parameter models use an 8,192-token context and 1,536 generated-token limit for RL/evaluation.",
              "- Initial caps constrain supplied SFT/retention certificates. The relative-reference tokenizer and verifier accept generated proofs beyond that cap; there is no six-line generation or acceptance cutoff. The per-attempt token budget still applies.",
              "- CoT is a learned symbolic dependency-plan scratchpad followed by a final natural-deduction proof. This is an R1-inspired scratchpad/verified-reward experiment, not a reproduction of DeepSeek-R1. Only the final proof is rewarded; evaluation samples are independent and do not receive verifier feedback while generating.",
              "- Cap 6 has 60,000 unique SFT examples; caps 10/12 have 63,372/64,969 with weighted long-proof mixtures. Raising the cap also expands family support. This is a comparison of training distributions, not an isolated causal effect of length.",
              "- The common RL sampling mixture is 50% from 512 SFT-exposed short prompts, 45% from 1,303 long synthetic prompts disjoint from every SFT pool, and 5% from the 14 permitted textbook training prompts. The long synthetic source witnesses span 7–14 lines, including 72 above 12; RL receives no proof or length labels. See `rl_pool_length_audit.json`.",
              "- All 4,969 added long certificates retain their lengths under the implemented dependency/reiteration compression, and every premise line participates in the final line's citation graph. The strict source audit and `long_source_dependency_audit.json` document these checks. Shorter alternative proofs can still exist.",
              "- The 14 previously permitted textbook training classes contribute prompt-only RL exposure; no textbook proof labels or additional classes enter training. The 58 development problems remain disjoint from training. The historical final set was already released and is excluded here.",
              f"- {len(receipt['interrupted'])} remote attempts were interrupted when other users' multi-GPU jobs appeared. The watchdog released only our own process groups. Partial attempts were archived on both hosts and excluded; retries used the same initialization and complete budgets. See `recovery_protocol.json`.",
              "- All six configurations are retained in the inventory. One training seed limits conclusions about robustness; no averaging across configurations or procedures is used.", "",
              "The cap 6 CoT same-seed infrastructure restart reproduced the SFT update count and compute exactly, but not the checkpoint or probe scores (training-pool 463→453/512; synthetic validation 225→224/256). "
              "Deterministic kernels were not enforced. The restart was required before textbook evaluation and was not selected by score; see `restart_reproducibility.json`. Small configuration differences require replication.", "",
              "## Reproducibility", "", f"Training protocol SHA256: `{results['training_protocol_sha256']}`.", "",
              f"Analysis protocol SHA256: `{results['analysis_protocol_sha256']}`. Results bind every evaluated checkpoint and saved-attempt file. "
              "Analysis tests cover strict replay in both modes, proof compression, theorem/reward tampering, missing/duplicate attempts, and whole-attempt inference budgets.", "",
              "`code/analyze.py` generates verified results; `code/report.py` generates this document and PNG/SVG figures. "
              "Dashboard project: `cap-comparison-v2`, with smoothing disabled and `compute/rl_budget_fraction` available as the x-axis."]
    if results["complete"]:
        count = sum(endpoint["strict_replayed_attempts"] for arm in results["arms"].values()
                    for endpoint in arm["endpoints"].values())
        lines += ["", f"The completed comparison contains **{count:,} distinct textbook attempts**, all strictly replayed. "
                  "The [completion audit](completion_audit.json) checks the frozen inputs and complete artifact inventory."]
        if (root / "reports/dashboard_browser_check.json").exists():
            lines += ["", "The [final dashboard snapshot](dashboard.png), [72-point accuracy audit](dashboard_accuracy_audit.json) "
                      "and [browser check](dashboard_browser_check.json) document separate run labels, checkpoint identities, "
                      "compute axes and disabled smoothing. The dashboard shows greedy accuracy; sampled coverage is in this report."]
    pending=[name for name,done in results["inventory"].items() if not done]
    if pending: lines += ["", "Incomplete configurations: " + ", ".join(pending) + ". No full comparison conclusion is available."]
    (root / "reports/summary.md").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    args=ap.parse_args(); root=args.root.resolve()
    results=read(root / "reports/analysis_results.json")
    if (results["training_protocol_sha256"] != sha(root / "plans/protocol.json")
            or results["analysis_protocol_sha256"] != sha(root / "reports/analysis_protocol.json")):
        raise ValueError("Report input protocol differs")
    if not results["arms"]:
        raise ValueError("No completed results to report")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    coverage_figures(root,results)
    length_figure(root,results)
    records=probe_records(root,results)
    write(root / "reports/fixed_probe_results.json",{"analysis_results_sha256":sha(root / "reports/analysis_results.json"),"records":records})
    probe_figure(root,results,records)
    inference_figure(root,results)
    family_figure(root,results)
    report(root,results,records)
    print(json.dumps({"completed_arms":len(results["arms"]),"summary":str(root / "reports/summary.md")}),flush=True)


if __name__ == "__main__":
    main()
