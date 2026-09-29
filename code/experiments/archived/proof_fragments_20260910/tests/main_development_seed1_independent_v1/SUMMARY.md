Main seed-one development audit

All 52,224 saved attempts from 16 fixed main checkpoints plus the adapted baseline passed independent token decoding, source-state reconstruction and strict proof replay. Release identities, checkpoint/code hashes, completion counts and whole-reference means reconcile.

Coverage at eight samples, out of 384 development sources:

| Arm | 0.5e15 | 1e15 | 2e15 | 4e15 |
| --- | ---: | ---: | ---: | ---: |
| selected / full | 141 | 141 | 139 | 139 |
| selected / reset | 144 | 134 | 153 | 136 |
| uniform / full | 150 | 152 | 140 | 130 |
| uniform / reset | 143 | 145 | 140 | 141 |

Adapted baseline: 157 / 384 covered. Every final main arm has a lower coverage point estimate and higher whole-reference loss than this baseline.

At the fixed 4e15 endpoint:

| Contrast | Coverage change (percentage points) | Paired context bootstrap range | Whole-proof bits change |
| --- | ---: | --- | ---: |
| selection_main_effect_f40 | 0.52 | [-2.57, 4.13] | -11.62 |
| reset_main_effect_f40 | 1.04 | [-1.71, 3.69] | -10.14 |
| interaction_f40 | -3.65 | [-9.21, 1.55] | -0.16 |

Negative bits changes favor the named intervention. Selection and resets lower relative reference loss, while their coverage ranges include zero. These comparisons concern reused development contexts at one fixed online-training and decoding seed; the ranges do not estimate seed-level uncertainty and are descriptive without multiplicity correction.

No checkpoint was chosen from these outcomes. All four fixed horizons and the original 4e15 primary endpoint remain in the proposed final set. Retention-only controls are pending and are needed to interpret continued supervised retention during the many updates with zero policy advantage.

Limit: saved reference NLL and behavior likelihood audits were hash-bound and aggregated, not recomputed with model forward passes. All credited raw and compressed proofs were independently checked. No final population was accessed.
