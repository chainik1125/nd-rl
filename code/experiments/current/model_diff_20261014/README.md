---
written_on: 2026-10-14
written_by: agent:claude
---

# Model diffing the GRPO wall: what RL changed to get depth-3 boxes

Robbie asked (2026-10-14, overnight): look at the GRPO model that suddenly got box depth 3; train SAEs and crosscoders;
find anything interesting. Context: [long-RL findings](../../../../.claude/notes/long-rl-what-models-learn.md) (GRPO v3
s0's "wall" over oracle calls 84-102 = learning depth-3 nested assumption boxes; s1 never did). Summary for humans:
[experiment-summaries/2026-10-14-grpo-wall-model-diff](../../../../experiment-summaries/2026-10-14-grpo-wall-model-diff/README.md).

Status: done for the night (2026-10-15 ~00:30 PDT). Everything below is teacher-forced on Lean-accepted proofs unless it
says "free generation". Proof length = body lines.

## One-paragraph answer

The wall was not about opening a third box; every model opens it fine. It was about **closing**: after a run of
consecutive box closes, the next `exact` must cite the `have` that opened the box just closed. Every pretrained model
(8 seeds) does this with a shortcut that is exact on its <= 6-line pretraining data: "depth after the r-th consecutive
close = max(0, 2 - r)". On a depth-3 proof that pops one level too far, cites the wrong `have`, and ends the proof with
a box still open; in free generation this is the first error in 93-95% of failed depth-3 samples. GRPO s0 and expert
iteration learned to use the depth at which the close run started. Mechanistically the fix is a small, distributed
change to blocks 0-2 at the closing tokens; the downstream "pointer" heads that copy the opener's name already existed
and simply land on the right box header afterwards. Crosscoders find **no model-exclusive features** (not for GRPO, not
for EI, and not even between two different pretraining seeds); what changes is what shared features write. A diff-SAE
on h_fin - h_pre is the best tool here: its top 20 latents (of 4,096), added to the pretrained model at the closing
tokens, take held-out k1 r2 pops from 27% to 60%, and they read as "first close of a run that started at depth >= 2".
First-order attribution patching gets even the sign wrong; only causal tests are trustworthy here.

## Models

All untied (run 198 trunk: 6 Peri-LN blocks, d 384, 8 heads of which 0-3 ALiBi and 4-7 NoPE), from HF
`robbiethompson2018/nd-rl-checkpoints`, downloaded to `ckpt/` (not committed).

| tag | run | what |
| --- | --- | --- |
| g0_pre / g0_fin | `long-rl-20261014-v3_untied-grpo_s0` pretrain.pt / final.pt | GRPO, crossed the wall |
| g1_pre / g1_fin | `..._v3_untied-grpo_s1` | GRPO, same recipe, never crossed |
| e0_*, e1_* | `..._v2_untied-ei_s0`, `_s1` | expert iteration, learned depth 3 gradually |
| v4 GRPO s0/s1, v4 EI s0/s1 | `..._v4_*` (pretrain, final, rl_round_NNN) | trajectories (checkpoints every 5 rounds); neither v4 GRPO run hit a wall |

Every run has its own pretraining seed (all state dicts differ). No intermediate checkpoints exist for v3, so "before the
wall" = the pretrained model. Relative L2 weight change pretrain → final: GRPO s0 9.0%, GRPO s1 8.7%, EI s0 72%.

## Terms

A **pop** is the `exact` right after a box closes (`... exact nE ) ; exact nX`): nX must be the `have` whose `:=` opened
the just-closed box. **k** = depth returned to; **r** = index of this close in a run of consecutive box closes;
**D_start** = k + r = depth before the run. A pop is **hard** if k != max(0, 2 - r). Spine-3 targets = the 1,805 ladder
targets whose goal is a 3-deep implication `A → (B → (C → D))`.

## Findings

### Behaviour

1. **The blocker is closing, not opening** (`localize.py`, `drill.py`, `closes.py`). Opening the third box costs ~0
   nats for every model. On spine-3 proofs (EI s1's texts), every pretrained model and g1_fin pop one level too far:
   the k=2 pop cites the depth-1 opener (93-98%), the k=1 pop cites the top-level opener (78-85%), and the `)` that
   closes box 1 is predicted as `<eos>` (85%). g0_fin: 92-100% correct; EI finals: 100%.
2. **The pretrained rule is "depth after the r-th consecutive close = max(0, 2 - r)"** (`runpos.py`, 3,498 pops):

   | g0_pre | r=1 | r=2 | r=3 |
   | --- | --- | --- | --- |
   | k=0 | 21%* | 91% | 89% |
   | k=1 | 72% | 27% | 0% |
   | k=2 | 21% | 0% | |

   *k=0 r=1 is mostly the second branch of an `Or.elim` in long gold proofs: the "nested case split" failure class, a
   different mechanism; identical in every model.
3. **Same rule in all 8 pretrained seeds; only runs that took off fix it** (`hardpop.py`, 1,000 records, k >= 1;
   `out/hardpop.json`, fig. f1, f3):

   | checkpoint | easy | hard | k1 r2 | k2 r1 | k1 r3 |
   | --- | --- | --- | --- | --- | --- |
   | 8 pretrained models (v2/v3/v4, all seeds) | 65-79% | 21-24% | 25-30% | 21-26% | 0% |
   | GRPO without a wall: v3 s1, v4 s0, v4 s1 finals | 71-86% | 25-33% | 29-30% | 27-44% | 0% |
   | GRPO v3 s0 final (the wall) | 80% | 69% | 72% | 78% | 41% |
   | EI v2 s0, v2 s1, v4 s0 finals | 94-99% | 81-87% | 85-96% | 89-93% | 44-47% |
   | EI v4 s1 final (dev 186, the weak seed) | 92% | 52% | 33% | 76% | 0% |

   D_start = 4 runs (k1 r3, k2 r2) stay at ~40% even in the best models: the fix covers depth 3, half of depth 4. v4 EI
   s0's k1 r2 goes 29% → 58% → 80% between rounds 10, 15 and 20 (abrupt under EI too); v4 EI s1 learned k2 r1 but never
   k1 r2. v4 GRPO s1 has a transient at round 100 (hard 36%) that is gone by round 120.
4. **It is the cause of death in free generation** (`gen_check.py`, 60 spine-3 targets x 4 samples, T 0.8, Lean):

   | model | accepted | rejected samples reaching depth 3 | first error = off-by-one pop |
   | --- | --- | --- | --- |
   | g0_pre | 2/240 | 194 | 185 (95%) |
   | g1_fin | 28/240 (all depth <= 2) | 197 | 184 (93%) |
   | g0_fin | 180/240 (152 at depth 3) | 45 | 9 |

5. **The wall is the moment depth-3 proofs start to succeed** (found_targets of all four GRPO runs, fig. f2). Every GRPO
   run solves ~240 spine-3 targets in the first 20 rounds, by proofs that avoid the third box (2-3% reach depth 3, 6%
   contain a hard pop). v3 s0's 2,089 proofs in calls 80-102: 99% reach depth 3, 99% contain a hard pop. Rounds 20-79
   held the seeds: v3 s0 found 11 hard-pop proofs, v3 s1 0, v4 s1 11 (and still never took off), so the seed count
   alone does not decide the takeoff.

### Mechanism

6. **The information is linearly present before RL but not used** (`probe_depth.py`, `probe_close.py`). Box depth is
   ~95% linearly decodable in every model from block 4 on; at pop tokens a ridge probe for k is right on 90% of g0_pre's
   hard pops (matching the heuristic on 5%). The rotated crosscoder decoders (below) are orthogonal to these probe
   directions (`align.py`, |cos| <= 0.06, inside the random null): the probe's direction is not the one the circuit reads.
7. **The fix lives in blocks 0-2, jointly, at the closing tokens** (`transplant.py`, `patch_resid.py`, `hardpop.py` on
   `ckpt/runs/long-rl-20261014-mix/`).
   - Single components moved from g0_fin into g0_pre (one block's attention or MLP, q/k or v/o, embeddings, head): no
     effect (7%/4%). Reverting single early components of g0_fin hurts k=1 pops (L0 attn 62%, L0 MLP 50%, L1 attn 44%,
     L2 attn 45%; clean 92%); blocks 3-5 don't matter.
   - Whole blocks of g0_fin in g0_pre (hard / k1 r2 / k2 r1): blocks 0-1 25/30/28%; **blocks 0-2 46/63/42%**; 0-3
     63/68/70%; 3-5 26/29/30%; 1-5 46/**30**/68% (block 0 is necessary for k1 r2; k2 r1 rides on blocks 1-3). g0_fin
     69/72/78%.
   - Residual patching: g0_fin's residual after 3 blocks written into g0_pre **only at the closing tokens**
     (`exact nE )`, `;`, pop `exact`) takes k=1 pops 7% → 88%; box-header tokens do nothing. g0_pre's residual after one
     block written into g0_fin at the closing tokens breaks them (92% → 11%).
8. **The readout: "pointer" heads that attend to a box header** (`attn_where.py`, `pointer.py`, `headscan.py`). From
   the pop `exact`, heads in layers 3-5 put most of their weight on one box header's `( fun (` tokens; the argmax equals
   that header's opener 70-97% of the time. In pretrained models and g1_fin these heads point at the wrong header on
   hard pops (right 18-34%); in g0_fin, e0_fin, e1_fin they point right (71-86%). Each pretrained model already had such
   heads (g0_pre L3.H2 split 0.34 right / 0.42 wrong; e1_pre L5.H2 0.61 on the wrong one; g1_pre's weak). **No single
   head is causal**: swapping one head's query, keys or whole output between models changes nothing (`patch_head.py`).
9. **Early attention at the closing tokens is local in every model** (`early_attn.py`): ALiBi heads attend to the
   current close run (the r count). The one visible recruit is g0_fin's NoPE head L3.H4, which looks up the header of
   the box being closed (0.35) and, at r=2, the box closed one step earlier (0.55): a content lookup of where the run
   started. Not necessary for k1 r2 (blocks 0-2 suffice), so the main change in blocks 0-2 is in what is written, not
   where heads look.

### Crosscoders, SAEs, diff-SAEs

`cc.py`: BatchTopK crosscoder (Lindsey et al. 2024; BatchTopK and decoder-norm-weighted selection as in Minder et al.
2025), 8,192 latents, k 32, AuxK, per-model scalar normalisation, 40M proof tokens with spine-3 records oversampled 3x;
`--diff` trains an SAE on h_fin - h_pre (4,096 latents, k 16). All splices preserve behaviour (e.g. g0 L3: g0_fin pop
accuracy 75.8% clean vs 75.0% with the layer replaced by its reconstruction), so the dictionaries capture the circuit.

10. **No model-exclusive latents, anywhere** (fig. f4):

    | crosscoder | layer | exclusive (r<0.1 or >0.9) | decoder cos < 0.5 |
    | --- | --- | --- | --- |
    | g0 pre/fin (GRPO, wall) | 1 / 2 / 3 | 0 / 0 / 0 | 4% / - / 8% |
    | g1 pre/fin (GRPO, no wall) | 3 | 0 | 4% |
    | e1 pre/fin (EI) | 3 / 5 | 0 / 2 | 92% / 94% |
    | g0_pre vs g1_pre (two pretraining seeds: positive control) | 3 | 28 (latent scaling: partly shared) | 100% |

    The positive control says the decoder-norm ratio cannot even detect a different model here: both seeds compute the
    same structural features in unrelated bases, so the crosscoder pairs them. Decoder cosine is the informative
    statistic, and for EI (72% weight change) it is swamped by basis drift. So for this task "no exclusive latents"
    means "no new token-level feature", not "nothing changed".
11. **Where the GRPO change sits in the crosscoder** (g0 L3; `analyze_cc.py`, `ablate_cc.py`, `interpret_top.py`).
    The latents most enriched on pops / closes are shared with lowered decoder cosine (#4540 −0.06, #9 −0.01, #2922
    0.29, #2149 0.53). They read as a run-position code: each fires on one token role at one r; a few are D_start
    sensitive (#6125 first close only when D_start >= 2; #9 the second close of a D_start = 3 run). Causal tests at
    the closing tokens, steering g0_pre by f_j (d_j^fin − d_j^pre):

    | latents steered | k1 r2 | k1 r3 |
    | --- | --- | --- |
    | none (g0_pre) | 27% | 0% |
    | 10-40 most rotated / most pop-enriched / 607 with cos < 0.5 | 27-31% | 0-4% |
    | top 20 by Σ f·‖Δd‖ at closing tokens | 51% | 33% |
    | top 100 | 69% | 43% |
    | all 7,214 | 71% | 43% |

    Ablating #9 + #2922 alone in g0_fin is specific: k1 r2 74% → 50%, k1 r3 41% → 12%, other cells unchanged;
    frequency-matched random latents do nothing. But adding 4 neighbouring run-position latents to that ablation removes
    the drop again (non-additive).
12. **The diff-SAE gives the sparsest handle** (`diff_analyze.py --mag`, rank on 250 spine-3 records, evaluate on
    1,000 other records). g0_pre + reconstructed diff at layer 3 (all positions): pop accuracy 47.9% → 65.3% vs 66.5%
    for the full diff. Steering at closing tokens only, top-N latents by change magnitude:

    | diff latents | k1 r2 | k1 r3 | k2 r1 |
    | --- | --- | --- | --- |
    | none | 27% | 0% | 26% |
    | top 5 | 29% | 0% | 26% |
    | **top 20 (of 4,096)** | **60%** | **43%** | 26% |
    | top 100 | 72% | 57% | 29% |

    (k2 r1 needs header positions too, as in finding 7.) The top diff latents: #327 fires on the first close of a run
    only when D_start >= 2 (96-98% at D_start 3, 66-73% at 2, ~2% at 1); #1873 on the first close and following pop
    of D_start >= 2 runs; #37/#712 on later pops of deep runs; #2128/#979/#2018 on name tokens of r >= 2 closes. That is
    the missing ingredient in readable form: D_start-aware run-position features. GRPO s1's diff is small by
    comparison (full diff at L3: g1_pre 46.8% → 53.1%).
13. **First-order attribution fails here** (`attrib.py`, `diff_analyze.py` without `--mag`). Metric logit(correct) −
    logit(enclosing box's opener), gradient at g0_pre's closing-token residuals: the first-order total of the change is
    **negative** (−226 over 465 pops) while the actual patch raises accuracy 27% → 73%; latents ranked by attribution
    (top 2 carry most of it) do nothing when steered. The readout is strongly nonlinear (attention softmax over box
    headers), so attribution patching should not be used to pick latents for this kind of change.

## Interpretation

- Pretraining (<= 6 body lines, depth 3 rare with trivial interiors) taught a cheap local rule at the closing tokens:
  count closes in the current run (ALiBi heads do this) and assume the run started at depth 2. True depth was decodable
  but the circuit did not read it.
- RL that succeeds (GRPO s0, EI) re-weighted blocks 0-2 so the closing tokens carry D_start-aware run-position
  information; the existing pointer heads then attend to the right box header and copy its opener. Nothing new was
  added to the feature vocabulary; the same features now write slightly different things. By the group's framing this
  is closer to elicitation of an already-represented quantity than to a new feature, but it needed a real circuit change
  that 3 of 4 GRPO runs never made.
- Why GRPO s0 and not the others: unresolved. All pretrained seeds are equally bad on hard pops; seeds (hard-pop proofs
  found in rounds 20-79) existed in v3 s0 and v4 s1 alike. Takeoff looks stochastic, with positive feedback once
  depth-3 proofs start entering the found-proof pool. EI's much larger updates (72% vs 9% weight change) make the same
  transition reliably.

## Limitations

- Teacher-forced on EI s1's (and corpus) proofs; free generation checked only for the spine-3 failure (finding 4).
- One crosscoder seed per configuration; dictionary size and k untuned (8,192 / 32; diff 4,096 / 16).
- Interpretation of latents is from firing profiles over a hand-built role labelling (`roles_lib.py`), not from
  automated labelling of top contexts.
- The diff-SAE / crosscoder steering uses both models' activations to compute the codes, so it localises the change; it
  is not a deployable steering vector for g0_pre.
- No v3 GRPO intermediate checkpoints: the wall itself (calls 84-102) is not observed from the inside.

## Next steps (not run)

1. Re-run GRPO from g0_pre and g1_pre with new RL seeds (4-6 runs, ~$2-3 each on A40) to separate "pretraining luck"
   from "RL luck"; save checkpoints every 5 rounds so the takeoff can be watched with `hardpop.py`.
2. Train-time fix in the spirit of the long-RL note: pretraining proofs with depth-3/4 boxes whose closing runs are
   preceded by non-trivial interiors (so max(0, 2 - r) is wrong on them); predict hard-pop accuracy ~80% before RL and
   no GRPO wall.
3. A deployable test of the diff-SAE finding: learn a small adapter on blocks 0-2 of g0_pre that reproduces the top-20
   diff latents' writes from g0_pre's own activations; check spine-3 solve rate in free generation.

## Files

| file | what |
| --- | --- |
| `nd.py` | loader on the real model code, tokenisation, structural labels |
| `tiny.py` | standalone re-implementation of the untied model (max logit diff 0.0 vs the real one), used on the pod |
| `corpus.py` | the shared corpus: 42,524 records, 13.6M tokens (pretraining, gold, EI s1 / GRPO s0 / GRPO s1 found proofs, spine-3 ladder) with per-token labels -> `out/corpus.pt` (not committed) |
| `localize.py`, `drill.py`, `closes.py`, `runpos.py`, `hardpop.py`, `gen_check.py` | behaviour |
| `probe_depth.py`, `probe_close.py`, `align.py` | linear probes |
| `transplant.py`, `patch_resid.py`, `patch_head.py` | causal localisation |
| `attn_pop.py`, `attn_where.py`, `pointer.py`, `headscan.py`, `early_attn.py` | attention |
| `cc.py`, `analyze_cc.py`, `ablate_cc.py`, `interpret_top.py`, `attrib.py`, `diff_analyze.py`, `diff_interpret.py`, `roles_lib.py` | crosscoders, SAEs, diff-SAEs |
| `figures.py` -> `figs/` | figures |
| `podctl.py`, `jobs/` | RunPod A40 plumbing and job scripts |
| `out/` | json outputs; `runs/` (crosscoder weights and stats, ~50 MB each) is not committed |
