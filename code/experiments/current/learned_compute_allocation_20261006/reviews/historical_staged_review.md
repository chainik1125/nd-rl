---
written_on: 2026-10-06
written_by: agent:codex
---

# Independent historical frozen-pool staged audit

Accepted the six original root CPU analyses following explicit historical release. Both execution ledgers are complete, all six child returncodes are0, and their combined argv exactly equals the immutable prepared six-command plan. No fitting, policy changes or neural inference were performed in this review.

Evidence: [feature replay](audit_historical_feature_staging.py) / [results](historical_feature_staging_independent_audit.json); [BCE replay](audit_historical_bce_staging.py) / [results](historical_bce_staging_independent_audit.json); [aggregate replay](audit_historical_staged_aggregate.py) / [results](historical_staged_aggregate_independent_audit.json).

The feature replay reconstructs frozen public-feature scores and3312 exact uniform-order policy rows, including required likelihood replay costs and fixed1 exemption. Its24054 scalar checks differ by at most5.55e−17. The three BCE critics'9936 policy rows independently reconstruct subset/order expectations and required critic input work, including oversized-neutral handling; maximum relative error4.86e−16. All MAIN-DEV policy choices remain byte-bound and unchanged.

The aggregate's9216 normalized rows, three-seed within-theorem averages, all policy means and all5000-draw paired role-stratified intervals reproduce exactly. There are72 historical theorems:58 textbook_dev and14 rl_train. Bootstrap draws preserve those observed counts, not equal role weights. These roles encode known prior reuse/exposure, so this remains a descriptive historical benchmark rather than fresh structural generalization.

At cost0.03, the feature adaptive policy has accuracy .55150463, expected attempts4.80094246, required logical work20145.50635, and reward .43747636. Its unchanged MAIN-DEV work mixture has accuracy .50288841 and work9121.23883. The adaptive accuracy difference is +.04861622 [.00566354,.09272620], but work increases11024.26752 [7455.30276,14953.65965]. Reward difference is −.01518555 [−.06466991,.03227860].

The three-seed BCE mean adaptive policy has accuracy .55226521, attempts4.42572751, work11623.19956, and reward .44949339. Against its own frozen work mixture, accuracy increases .06138103 [.01975748,.10568453], while work increases6326.86482 [4184.12507,8663.40302]. Reward difference is .00778350 [−.04050956,.05784956].

These are substantial TEST budget mismatches under distribution shift. Accuracy gains do not establish equal-compute superiority; both reward-difference intervals cross zero. No weights were refitted to historical outcomes. Logical work excludes padding and full deployment overhead, and feature replay remains charged. Feature fixed8 accuracy .55555556 independently reproduces. This review does not substitute a new threshold or redefine the historical role estimand.

All three helpers pass Ruff and syntax checks. The numerical checks were executed; no historical selection-successor output is included in this note yet.
