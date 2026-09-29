This is a proposed comparison, not a data release. Neither arm has been trained.
The existing 85-item online-mining export and 306-endpoint reference export
remain unchanged. The proposed arms each contain 84 distinct endpoints and are
disjoint from one another. The length-6 mined certificate is excluded because
the reference-only control has no corresponding support.

The common line histogram is 10 / 36 / 27 / 11 examples of lengths 2 / 3 / 4 / 5.
Matching used no model outcome: exact line length, response-token distance,
prompt-token distance, then a seeded tie rank. Mean absolute response-token
distance is 0.595 tokens, with maximum 5 and 50 exactly matched pairs. Total
response tokens are 4,068 for mined versus 4,086 for reference. Context counts
are 51 versus 70; context effective counts are 40.55 versus 56.00. The family
distribution total-variation distance is 0.381. These remaining differences
prevent interpreting this as an isolated acquisition-method effect.

Every mined row preserves its acquired-reference overlap report and the
independent ancestry-review hits. Among the 84 eligible mined rows, 14 theorem
endpoints and 12 exact proof prefixes occur in baseline/adaptation supervised
training-source inventories. Separately, 47 endpoints and 33 proof prefixes
occur in acquired candidate/selection references. These flags overlap and
must not be added. Retention-inventory matches are zero. Historical individual
ancestor draws have not been reconstructed here. Reference-control ancestry
beyond its acquired-candidate origin remains unmeasured in this proposal.

Root can promote this proposal with the following explicit procedure:

1. Review the matching receipt and exposure limitations. Freeze the current
   implementation in a new snapshot containing `diagnostic_certificate_sft.py`
   and `prepare_prefix_comparison.py` plus their pinned dependencies.
2. Choose one common training schedule before evaluation. A concrete initial
   diagnostic schedule is 100 updates, seed 20260910, learning rate 0.00005,
   batch size 32, fixed pad 512, retention fraction 0.5, and fp32. This yields
   1,600 primary and 1,600 retained proof draws per arm. Paired pool ordering
   ensures the same primary index stream matches line length at every draw;
   the independent retention stream is identical.
3. Copy `release_template.json` to a new root-owned config. Fill `steps` and
   `seed`, set `frozen` and `approved_for_diagnostic_full_certificate_sft` to
   true, and preserve the proposal/cohort/checkpoint/retention hashes. The
   proposal and original diagnostic roles stay unchanged. The current
   template is deliberately rejected by the trainer.
4. Launch both explicit full-certificate arms from the common adapted weights,
   through the existing root-controlled GPU launcher. For each arm, the worker
   command has this form (substitute the frozen snapshot, root release and
   fresh output):

   ```
   python FROZEN_SNAPSHOT/diagnostic_certificate_sft.py \
     --checkpoint results/adapt_v1/final.pt \
     --retention data/released_v1/retention.jsonl \
     --proposal code/diagnostics/mined_reference_comparison_proposal_v1 \
     --release ROOT_RELEASE_JSON --arm mined --supervision full-certificate \
     --steps 100 --seed 20260910 --lr 0.00005 --batch-size 32 --fixed-pad 512 \
     --retention-fraction 0.5 --precision fp32 --device cuda --output NEW_OUTPUT
   ```

   Use `--arm reference` with another fresh output for the paired arm.
5. Require equal dense training proxy and retention-schedule hashes before
   comparing frozen endpoints. Use the already established unaided evaluation
   protocol with a separate root checkpoint/data release. Training-set local
   reference losses are not held-out evidence. No final data has been opened
   by this preparation path.

The new trainer supplies no proof prefix: its supervised response contains
FINAL, every line of the local certificate, and QED. Full proof tokenization
and round trips have been checked. The existing fragment intervention and its
cap6 diagnostic-role rejection remain unchanged. This is an explicit approved
use of diagnostic data only after root creates the separate release.
