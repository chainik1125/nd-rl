# Second-reader check: candidate_v0 (104)

Plan gate T4: 20 random problems checked against the printed source by someone other than the transcriber. Sources are LaTeX files at pinned commits (links go to the exact line) and the Carnap Book pages.

**How to check.** Open the source at the location given. Confirm the prompt is the exercise as printed, allowing only: letters renamed to P–S as the notes say; ↔ written as (A → B) ∧ (B → A); the book's own bracket conventions. Mark each row OK, or write what differs. Return the file to Claude; any mismatch widens the sample.

Sample: 20 of 104 rows, `random.Random(20260930).sample`, rows numbered from 0 in `problems.jsonl`.

## 1. Row 2: `textbook_95041444f8cb17626d64`

- **Source:** forall x: Calgary (Thomas-Bolduc & Zach, after Magnus and Button) — [forallx-yyc-prooftfl.tex:793 (ch. Basic rules for TFL, block at line 785, item 5)](https://github.com/rzach/forallx-yyc/blob/c5cb41544eb61302fa2d06b8a86c76ed3ba52a7c/forallx-yyc-prooftfl.tex#L793); item 5
- **As printed:** `\item $(C\eand D)\eor E \therefore E\eor D$`
- **Transcribed as:** ((P ∧ Q) ∨ R) ⊢ (R ∨ Q)
- **Prompt:** `THM ( ( P & Q ) v R ) SEQ ( R v Q ) PRF`
- **Encoding notes:** sentence letters renamed: C→P, D→Q, E→R
- **Verdict:** [ ] OK   [ ] differs: 

## 2. Row 6: `textbook_12d1e9ba77154cecdaa4`

- **Source:** forall x: Calgary (Thomas-Bolduc & Zach, after Magnus and Button) — [forallx-yyc-prooftfl.tex:797 (ch. Basic rules for TFL, block at line 785, item 9)](https://github.com/rzach/forallx-yyc/blob/c5cb41544eb61302fa2d06b8a86c76ed3ba52a7c/forallx-yyc-prooftfl.tex#L797); item 9
- **As printed:** `\item $P \eand (Q\eor R), P\eif \enot R \therefore Q\eor E$`
- **Transcribed as:** (P ∧ (Q ∨ R)), (P → (¬R)) ⊢ (Q ∨ S)
- **Prompt:** `THM ( P & ( Q v R ) ) , ( P > ( ~ R ) ) SEQ ( Q v S ) PRF`
- **Encoding notes:** sentence letters renamed: P→P, Q→Q, R→R, E→S
- **Verdict:** [ ] OK   [ ] differs: 

## 3. Row 28: `textbook_fcf1f314b60800e5aff4`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:357 (sec. Additional assumptions, block at line 351, item 3)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L357); item 3
- **As printed:** `\item $P \eand Q, (Q \eand P) \eif R \proves R$`
- **Transcribed as:** (P ∧ Q), ((Q ∧ P) → R) ⊢ R
- **Prompt:** `THM ( P & Q ) , ( ( Q & P ) > R ) SEQ R PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 4. Row 35: `textbook_7c718c1db52fdc350567`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:453 (sec. Theorems and reiterating, block at line 448, item 2)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L453); item 2
- **As printed:** `\item $\vdash (A\eif B)\eif ((B\eif C)\eif (A\eif C))$`
- **Transcribed as:** ⊢ ((P → Q) → ((Q → R) → (P → R)))
- **Prompt:** `THM SEQ ( ( P > Q ) > ( ( Q > R ) > ( P > R ) ) ) PRF`
- **Encoding notes:** sentence letters renamed: A→P, B→Q, C→R
- **Verdict:** [ ] OK   [ ] differs: 

## 5. Row 39: `textbook_87a23fa584426b3d9328`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:530 (sec. Biconditional rules, block at line 521, item 6)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L530); item 6
- **As printed:** `\item $P \rightarrow R,  \ R\rightarrow (S \land P) \vdash P \leftrightarrow R$`
- **Transcribed as:** (P → Q), (Q → (R ∧ P)) ⊢ ((P → Q) ∧ (Q → P))
- **Prompt:** `THM ( P > Q ) , ( Q > ( R & P ) ) SEQ ( ( P > Q ) & ( Q > P ) ) PRF`
- **Encoding notes:** sentence letters renamed: P→P, R→Q, S→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 6. Row 51: `textbook_06ccd6ec0200aeb51862`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:812 (sec. Disjunction rules, block at line 799, item 10)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L812); item 10
- **As printed:** `\item $(A \eor B) \eor C, \ C \eiff B \proves C \eor A$`
- **Transcribed as:** ((P ∨ Q) ∨ R), ((R → Q) ∧ (Q → R)) ⊢ (R ∨ P)
- **Prompt:** `THM ( ( P v Q ) v R ) , ( ( R > Q ) & ( Q > R ) ) SEQ ( R v P ) PRF`
- **Encoding notes:** sentence letters renamed: A→P, B→Q, C→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 7. Row 53: `textbook_911ca699a25f56477d3e`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:815 (sec. Disjunction rules, block at line 799, item 13)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L815); item 13
- **As printed:** `\item $J \eif I, \ L \rightarrow (L \rightarrow K)  \proves (L \eor J) \eif (I \eor K)$`
- **Transcribed as:** (P → Q), (R → (R → S)) ⊢ ((R ∨ P) → (Q ∨ S))
- **Prompt:** `THM ( P > Q ) , ( R > ( R > S ) ) SEQ ( ( R v P ) > ( Q v S ) ) PRF`
- **Encoding notes:** sentence letters renamed: J→P, I→Q, L→R, K→S
- **Verdict:** [ ] OK   [ ] differs: 

## 8. Row 59: `textbook_b7132b7858d742201ec0`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:940 (sec. Proof strategies, block at line 931, item 7)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L940); item 7; right to left
- **As printed:** `\item $P \eiff \enot Q \pequiv \enot(P \eiff Q)$`
- **Transcribed as:** (¬((P → Q) ∧ (Q → P))) ⊢ ((P → (¬Q)) ∧ ((¬Q) → P))
- **Prompt:** `THM ( ~ ( ( P > Q ) & ( Q > P ) ) ) SEQ ( ( P > ( ~ Q ) ) & ( ( ~ Q ) > P ) ) PRF`
- **Encoding notes:** ↔ written as (A → B) ∧ (B → A); interderivable pair, right to left
- **Verdict:** [ ] OK   [ ] differs: 

## 9. Row 64: `textbook_17db74b2c27db844c116`

- **Source:** forall x: Syracuse (Rieppel, after Button's Cambridge edition) — [forallx-04-tfl-prf.tex:1140 (sec. Derived rules, block at line 1130, item 8)](https://github.com/mrieppel/forallxsyr/blob/46841216b602e7e9d0fa4069593dfb47e1af0a91/forallx-04-tfl-prf.tex#L1140); item 8
- **As printed:** `\item $(B \eand C) \eor \enot (A \eif \enot D), \enot C \proves A \eand D$`
- **Transcribed as:** ((P ∧ Q) ∨ (¬(R → (¬S)))), (¬Q) ⊢ (R ∧ S)
- **Prompt:** `THM ( ( P & Q ) v ( ~ ( R > ( ~ S ) ) ) ) , ( ~ Q ) SEQ ( R & S ) PRF`
- **Encoding notes:** sentence letters renamed: B→P, C→Q, A→R, D→S
- **Verdict:** [ ] OK   [ ] differs: 

## 10. Row 72: `textbook_c1d05f6df5cf79a1edb7`

- **Source:** forall x: Mississippi State (Johnson, after Calgary) — [forallx-msu-part4a--proof_tfl.tex:1257 (ch. Rules of derivation, block at line 1235, item 9)](https://github.com/loighic/forallx-msu/blob/1c4200da893a75bac85d37f97f904a9e2e058a00/forallx-msu-part4a--proof_tfl.tex#L1257); item 9
- **As printed:** `\item $S \eif T, Q \eand \enot R, \enot R \eiff (T \eif S) \proves S \eiff T $\smallskip`
- **Transcribed as:** (P → Q), (R ∧ (¬S)), (((¬S) → (Q → P)) ∧ ((Q → P) → (¬S))) ⊢ ((P → Q) ∧ (Q → P))
- **Prompt:** `THM ( P > Q ) , ( R & ( ~ S ) ) , ( ( ( ~ S ) > ( Q > P ) ) & ( ( Q > P ) > ( ~ S ) ) ) SEQ ( ( P > Q ) & ( Q > P ) ) PRF`
- **Encoding notes:** sentence letters renamed: S→P, T→Q, Q→R, R→S; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 11. Row 74: `textbook_e2c593f0543fc6234b2d`

- **Source:** forall x: Mississippi State (Johnson, after Calgary) — [forallx-msu-part4a--proof_tfl.tex:1260 (ch. Rules of derivation, block at line 1235, item 12)](https://github.com/loighic/forallx-msu/blob/1c4200da893a75bac85d37f97f904a9e2e058a00/forallx-msu-part4a--proof_tfl.tex#L1260); item 12
- **As printed:** `\item $S \eif T, T \eif S, T \eiff (T \eiff S) \proves S$\smallskip`
- **Transcribed as:** (P → Q), (Q → P), ((Q → ((Q → P) ∧ (P → Q))) ∧ (((Q → P) ∧ (P → Q)) → Q)) ⊢ P
- **Prompt:** `THM ( P > Q ) , ( Q > P ) , ( ( Q > ( ( Q > P ) & ( P > Q ) ) ) & ( ( ( Q > P ) & ( P > Q ) ) > Q ) ) SEQ P PRF`
- **Encoding notes:** sentence letters renamed: S→P, T→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 12. Row 80: `textbook_5c8618f27cf57fd6c99b`

- **Source:** forall x: Mississippi State (Johnson, after Calgary) — [forallx-msu-part4b-sound_complete.tex:123 (ch. Soundness, block at line 111, item 7)](https://github.com/loighic/forallx-msu/blob/1c4200da893a75bac85d37f97f904a9e2e058a00/forallx-msu-part4b-sound_complete.tex#L123); item 7
- **As printed:** `\item $F \eand (K \eand R) \proves F \eiff (K \eiff R)$`
- **Transcribed as:** (P ∧ (Q ∧ R)) ⊢ ((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P))
- **Prompt:** `THM ( P & ( Q & R ) ) SEQ ( ( P > ( ( Q > R ) & ( R > Q ) ) ) & ( ( ( Q > R ) & ( R > Q ) ) > P ) ) PRF`
- **Encoding notes:** exercise: prove if valid; sentence letters renamed: F→P, K→Q, R→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 13. Row 81: `textbook_61c2a7da79562d1ee741`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/3, exercise 4.1](https://carnap.io/book/3); item 4.1
- **As printed:** `P:|-: P`
- **Transcribed as:** P ⊢ P
- **Prompt:** `THM P SEQ P PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 14. Row 82: `textbook_5f07170d9486281717b7`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/3, exercise 4.4](https://carnap.io/book/3); item 4.4
- **As printed:** `~P, ~Q->P :|-: Q`
- **Transcribed as:** (¬P), ((¬Q) → P) ⊢ Q
- **Prompt:** `THM ( ~ P ) , ( ( ~ Q ) > P ) SEQ Q PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 15. Row 86: `textbook_70f616e852eadc25705c`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/5, exercise 6.5](https://carnap.io/book/5); item 6.5
- **As printed:** `(S->R)->Q, R :|-: Q`
- **Transcribed as:** ((P → Q) → R), Q ⊢ R
- **Prompt:** `THM ( ( P > Q ) > R ) , Q SEQ R PRF`
- **Encoding notes:** sentence letters renamed: S→P, R→Q, Q→R
- **Verdict:** [ ] OK   [ ] differs: 

## 16. Row 90: `textbook_95d404da37f07dfd2b34`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/6, exercise 7.7](https://carnap.io/book/6); item 7.7
- **As printed:** `:|-: (-P->P) -> --P`
- **Transcribed as:** ⊢ (((¬P) → P) → (¬(¬P)))
- **Prompt:** `THM SEQ ( ( ( ~ P ) > P ) > ( ~ ( ~ P ) ) ) PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 17. Row 91: `textbook_d9dc2ee1269d002c72d2`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/6, exercise 7.8](https://carnap.io/book/6); item 7.8
- **As printed:** `R->(Q->P), Q->-P :|-: R->-Q`
- **Transcribed as:** (P → (Q → R)), (Q → (¬R)) ⊢ (P → (¬Q))
- **Prompt:** `THM ( P > ( Q > R ) ) , ( Q > ( ~ R ) ) SEQ ( P > ( ~ Q ) ) PRF`
- **Encoding notes:** sentence letters renamed: R→P, Q→Q, P→R
- **Verdict:** [ ] OK   [ ] differs: 

## 18. Row 92: `textbook_c367f2ad3a68caa103ce`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/6, exercise 7.9](https://carnap.io/book/6); item 7.9
- **As printed:** `P->Q, Q->R, R->S :|-: P->-(P->-S)`
- **Transcribed as:** (P → Q), (Q → R), (R → S) ⊢ (P → (¬(P → (¬S))))
- **Prompt:** `THM ( P > Q ) , ( Q > R ) , ( R > S ) SEQ ( P > ( ~ ( P > ( ~ S ) ) ) ) PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 19. Row 93: `textbook_d66e5d2ec8cab38d5b68`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/6, exercise 7.10](https://carnap.io/book/6); item 7.10
- **As printed:** `P->Q, (R->Q)->-P :|-: -P`
- **Transcribed as:** (P → Q), ((R → Q) → (¬P)) ⊢ (¬P)
- **Prompt:** `THM ( P > Q ) , ( ( R > Q ) > ( ~ P ) ) SEQ ( ~ P ) PRF`
- **Encoding notes:** none
- **Verdict:** [ ] OK   [ ] differs: 

## 20. Row 96: `textbook_43e30ad215c242c5605c`

- **Source:** The Carnap Book (Leach-Krouse & Ehrlich) — [carnap.io/book/7, exercise 8.7](https://carnap.io/book/7); item 8.7
- **As printed:** `P -> R, R -> -Q, S-> P, Q->S :|-: -Q`
- **Transcribed as:** (P → Q), (Q → (¬R)), (S → P), (R → S) ⊢ (¬R)
- **Prompt:** `THM ( P > Q ) , ( Q > ( ~ R ) ) , ( S > P ) , ( R > S ) SEQ ( ~ R ) PRF`
- **Encoding notes:** sentence letters renamed: P→P, R→Q, Q→R, S→S
- **Verdict:** [ ] OK   [ ] differs: 
