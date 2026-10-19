**For the current unrestricted search at `floor=20`, I found no cost-increasing flaw in transformations (1)–(5). I believe their substantive arguments hold.** I did reproduce a false certificate at the direct API boundary, and found a small exception missing from the tower proof’s wording.

1. **Memoization and budget reuse: sound under the current defaults.**  
   [The implementation](fastmin.c#L367) distinguishes exact minima from “no proof through budget `b`.” Successful searches minimize within their budget, so their answers are exact; failed searches justify reuse only at smaller or equal budgets. The stricter C branch bounds discard ties, not possible improvements.

   I checked record identity comparisons, eviction, resizing, and cached-pointer invalidation through `epoch`. I found no stale-record path. Eviction can lose work without manufacturing a bound.

   Crucially, full search and “no top lemma” search have **separate memo slots**. These costs need not agree: with
   ```
   A = {P, P→Q, Q→R}
   ```
   full search derives `R` in two lines, while reduced search without a top lemma cannot do so within two. Sharing those lower bounds would be unsound; the code correctly avoids that.

   **Depth and hypothesis provenance are safely omitted only because this wrapper uses unrestricted defaults.** I reproduced these contrasting reference-search answers:

   - With `max_depth=1`, `A={Q}`, goal `P→P`, budget 2: failure at depth 1, success at depth 0.
   - With `ANDE_NEGI_HYP`, `A={P∧Q}`, goal `P`, budget 1: failure when the conjunction is tagged as an enclosing NEGI hypothesis, success without that tag.

   Reusing the first failure in either second context would miss a shorter proof. Those restrictions are absent here. Also, the byte encoding assumes budgets/costs below 128; `floor=20` fits, but the API should enforce that limit.

2. **Semantic pruning: sound.**  
   `A` includes all currently citable formulas, including open hypotheses. Every search constructor preserves classical entailment from that context. The truth-table implementation handles inconsistent contexts correctly: they entail every goal. The wrapper enforces the ten-atom limit required by the fixed arrays.

3. **Dropping towers: I believe the normalization argument works, including its difficult NEGI case.**  
   [Lemma B and the main induction](fastmin.c#L30) use the essential condition
   ```
   z ∈ sub, but ~z ∉ sub
   ```
   before replacing available `~~z` with `z`. That condition ensures `~~~z` is unavailable in the restricted universe, and `~~z` cannot be a component of another formula there. Its relevant uses reduce to DN and contradiction with `~z`, both handled without increased cost.

   The concrete counterexample to **weakening** that condition is:
   ```
   {~~P, ~~~P} ⊢ F     one line
   {P,   ~~~P} ⊢ F     two lines
   ```
   I reproduced both costs. The actual condition excludes this counterexample.

   The outside-S NEGI polarity reversal is handled by Lemma B; simply claiming that `phi` commutes with negation would have been wrong.

   **Small proof-wording defect:** F4, stated universally, fails for `f=F` when `F∉sub`: `phi(F)=F`, `~F∉S`, but `F` is not `~~y`. F4 needs an explicit bottom exception. The subsequent induction separately handles the relevant odd-bottom hypothesis through `DROP`; I do **not** see this wording defect yielding a missed proof.

4. **Reduced rules: sound after tower elimination.**  
   [The reduction](fastmin.c#L434) can select the first unavailable input in evaluation order, derive it as a lemma at the **same box level**, and continue with it available. For multiple inputs, repeat this transformation. No extra proof line is charged for the lemma constructor.

   This preserves the sharing already present in ANDI/IMPE/NEGE. For ORE, only the disjunction’s derivation moves into a preceding lemma; branch derivations retain their hypotheses. Self-goal inputs cannot improve an optimum: the derivation has already obtained its conclusion before the final rule.

5. **Lemma hoisting: sound as an existential normalization, not an equality for every lemma.**  
   The transformation is:
   ```
   derive L using an initial lemma M; continue to G
       →
   derive M; derive L with M available; continue to G
   ```
   Everything remains in the same context. Additional available formulas cannot increase continuation cost, and choosing the cheapest first lemma gives the strict descent used in the header. If `M=G`, its derivation already beats the supposed optimal whole proof.

   The implementation correctly leaves **full search inside box bodies and continuations**. It does not try to pull hypothesis-dependent lemmas out of boxes. The separate memo kinds above are necessary for this argument.

**Reproduced API bug:** [the entry point skips searching when the conclusion is a premise](fastmin.c#L868), leaving the return value meaning “exhausted.” Consequently:
```python
certify_fast("THM P SEQ P PRF", 1, floor=20)
# min_body=None, cert_ge=20, timeout=False
```
Yet `Search.best` finds the one-line reiteration immediately. Python `Search.run` has the same skip, so this is inherited behavior, not a C divergence. `C.check` rejects such inputs in the CLI pipeline; direct API callers are exposed. Rejecting them explicitly would prevent the false certificate.

**The certification scope also matters beyond formula restrictions.** I verified this two-line proof:
```
N1 | F     : AS
N2   ~F    : NEGI N1 N1
```
Both searches report three body lines for `⊢ ~F`, because neither implements the hypothesis-only NEGI box. Thus even matching minlen exactly does not establish a lower bound over all verifier-valid proofs using S.

Validation: 160 random valid sequents matched Python minlen, default C, `FULLRULES`, `NOHOIST`, and `TOWERS` through six body lines; tiny-cache runs also matched while forcing 151 evictions. Nine targeted cases matched across C variants through eleven body lines, including negation boundaries, classical proofs, ORE, and a ten-atom implication chain. No source changes.