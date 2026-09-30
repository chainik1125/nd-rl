# Second-reader check: candidate_v1 (80)

Plan gate T4: 20 random problems checked against the printed source by someone other than the transcriber. Sources: Roy's `SymLogFullTxt.pdf` and Carr's `NDpack.pdf` in `data/textbook_problem/raw/` (gitignored; URLs in `candidate_v1/README.md`). Roy's PDF page numbers are given; his printed page number is 12 less.

**How to check.** Open the source at the location given. Confirm the prompt is the exercise as printed, allowing only: letters renamed to P–S as the notes say; ↔ written as (A → B) ∧ (B → A); the book's own bracket conventions. Mark each row OK, or write what differs. Return the file to Claude; any mismatch widens the sample.

Sample: 20 of 80 rows, `random.Random(20260930).sample`, rows numbered from 0 in `problems.jsonl`.

## 1. Row 2: `textbook_158a629091d4e5e68845`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 227, sec. 6.2.1 (→ and ∧), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.7f
- **As printed:** `A, (A ∧ B) → (C ∧ D) ⊢ B → C`
- **Transcribed as:** P, ((P ∧ Q) → (R ∧ S)) ⊢ (Q → R)
- **Prompt:** `THM P , ( ( P & Q ) > ( R & S ) ) SEQ ( Q > R ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q, C→R, D→S
- **Verdict:** [ ] OK   [ ] differs: 

## 2. Row 3: `textbook_d6b908b9f5a04da5eb8e`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 227, sec. 6.2.1 (→ and ∧), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.7g
- **As printed:** `C → A, C → (A → B) ⊢ C → (A ∧ B)`
- **Transcribed as:** (P → Q), (P → (Q → R)) ⊢ (P → (Q ∧ R))
- **Prompt:** `THM ( P > Q ) , ( P > ( Q > R ) ) SEQ ( P > ( Q & R ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: C→P, A→Q, B→R
- **Verdict:** [ ] OK   [ ] differs: 

## 3. Row 4: `textbook_5d3bc34b8343293008e6`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 227, sec. 6.2.1 (→ and ∧), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.7h
- **As printed:** `A → B, B → C ⊢ (A ∧ K) → C`
- **Transcribed as:** (P → Q), (Q → R) ⊢ ((P ∧ S) → R)
- **Prompt:** `THM ( P > Q ) , ( Q > R ) SEQ ( ( P & S ) > R ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R, K→S
- **Verdict:** [ ] OK   [ ] differs: 

## 4. Row 5: `textbook_ee78aa5facb31dea0f2b`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 227, sec. 6.2.1 (→ and ∧), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.7j
- **As printed:** `D ∧ E, (D → F ) ∧ (E → G) ⊢ F ∧ G`
- **Transcribed as:** (P ∧ Q), ((P → R) ∧ (Q → S)) ⊢ (R ∧ S)
- **Prompt:** `THM ( P & Q ) , ( ( P > R ) & ( Q > S ) ) SEQ ( R & S ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: D→P, E→Q, F→R, G→S
- **Verdict:** [ ] OK   [ ] differs: 

## 5. Row 6: `textbook_a65326c83332278d644e`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 227, sec. 6.2.1 (→ and ∧), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.7k
- **As printed:** `O → B, B → S , S → L ⊢ O → L`
- **Transcribed as:** (P → Q), (Q → R), (R → S) ⊢ (P → S)
- **Prompt:** `THM ( P > Q ) , ( Q > R ) , ( R > S ) SEQ ( P > S ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: O→P, B→Q, S→R, L→S
- **Verdict:** [ ] OK   [ ] differs: 

## 6. Row 10: `textbook_2c11961d305c0c9b6759`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 235, sec. 6.2.2 (∼ and ∨), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.10f
- **As printed:** `A ∧ B ⊢ (R ↔ S) ∨ B`
- **Transcribed as:** (P ∧ Q) ⊢ (((R → S) ∧ (S → R)) ∨ Q)
- **Prompt:** `THM ( P & Q ) SEQ ( ( ( R > S ) & ( S > R ) ) v Q ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q, R→R, S→S; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 7. Row 22: `textbook_7d7325c3e090d7880e32`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 242, sec. 6.2.3 (↔), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.13n
- **As printed:** `¬A → B, A → ¬B ⊢ ¬A ↔ B`
- **Transcribed as:** ((¬P) → Q), (P → (¬Q)) ⊢ (((¬P) → Q) ∧ (Q → (¬P)))
- **Prompt:** `THM ( ( ~ P ) > Q ) , ( P > ( ~ Q ) ) SEQ ( ( ( ~ P ) > Q ) & ( Q > ( ~ P ) ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 8. Row 24: `textbook_db6d3b54eef1e05a7c29`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, theorems T6.1–T6.20 (E6.16), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item T6.3
- **As printed:** `⊢ (¬Q → ¬P ) → ((¬Q → P ) → Q)`
- **Transcribed as:** ⊢ (((¬P) → (¬Q)) → (((¬P) → Q) → P))
- **Prompt:** `THM SEQ ( ( ( ~ P ) > ( ~ Q ) ) > ( ( ( ~ P ) > Q ) > P ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: Q→P, P→Q
- **Verdict:** [ ] OK   [ ] differs: 

## 9. Row 28: `textbook_f293718d83c7943f5f2e`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, theorems T6.1–T6.20 (E6.16), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item T6.13
- **As printed:** `⊢ (A ∨ B) ↔ (B ∨ A)`
- **Transcribed as:** ⊢ (((P ∨ Q) → (Q ∨ P)) ∧ ((Q ∨ P) → (P ∨ Q)))
- **Prompt:** `THM SEQ ( ( ( P v Q ) > ( Q v P ) ) & ( ( Q v P ) > ( P v Q ) ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 10. Row 35: `textbook_f31c1b9612d470888486`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, theorems T6.1–T6.20 (E6.16), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item T6.20
- **As printed:** `⊢ [A ∨ (B ∨ C)] ↔ [(A ∨ B) ∨ C]`
- **Transcribed as:** ⊢ (((P ∨ (Q ∨ R)) → ((P ∨ Q) ∨ R)) ∧ (((P ∨ Q) ∨ R) → (P ∨ (Q ∨ R))))
- **Prompt:** `THM SEQ ( ( ( P v ( Q v R ) ) > ( ( P v Q ) v R ) ) & ( ( ( P v Q ) v R ) > ( P v ( Q v R ) ) ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q, C→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 11. Row 39: `textbook_6c3a988514b8a2c9259f`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, sec. 6.2.4 Strategy, NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.15f
- **As printed:** `(A ∧ B) → (C ∧ D) ⊢ [(A ∧ B) → C ] ∧ [(A ∧ B) → D]`
- **Transcribed as:** ((P ∧ Q) → (R ∧ S)) ⊢ (((P ∧ Q) → R) ∧ ((P ∧ Q) → S))
- **Prompt:** `THM ( ( P & Q ) > ( R & S ) ) SEQ ( ( ( P & Q ) > R ) & ( ( P & Q ) > S ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R, D→S
- **Verdict:** [ ] OK   [ ] differs: 

## 12. Row 44: `textbook_260e73207a2aecc2bbac`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, sec. 6.2.4 Strategy, NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.15o
- **As printed:** `¬A ↔ ¬B ⊢ A ↔ B`
- **Transcribed as:** (((¬P) → (¬Q)) ∧ ((¬Q) → (¬P))) ⊢ ((P → Q) ∧ (Q → P))
- **Prompt:** `THM ( ( ( ~ P ) > ( ~ Q ) ) & ( ( ~ Q ) > ( ~ P ) ) ) SEQ ( ( P > Q ) & ( Q > P ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 13. Row 51: `textbook_c5bd6ea079ef7455b1e2`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 251, sec. 6.2.4 Strategy, NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.15x
- **As printed:** `⊢ ¬A → [(B ∧ A) → C ]`
- **Transcribed as:** ⊢ ((¬P) → ((Q ∧ P) → R))
- **Prompt:** `THM SEQ ( ( ~ P ) > ( ( Q & P ) > R ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R
- **Verdict:** [ ] OK   [ ] differs: 

## 14. Row 53: `textbook_1ba13cd6b075e3368060`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 263, theorems T6.21–T6.30 (E6.19, E6.20e), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item T6.23
- **As printed:** `⊢ ¬(A ∨ B) ↔ (¬A ∧ ¬B)`
- **Transcribed as:** ⊢ (((¬(P ∨ Q)) → ((¬P) ∧ (¬Q))) ∧ (((¬P) ∧ (¬Q)) → (¬(P ∨ Q))))
- **Prompt:** `THM SEQ ( ( ( ~ ( P v Q ) ) > ( ( ~ P ) & ( ~ Q ) ) ) & ( ( ( ~ P ) & ( ~ Q ) ) > ( ~ ( P v Q ) ) ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 15. Row 55: `textbook_2e2c1312ef21c3a68875`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 263, theorems T6.21–T6.30 (E6.19, E6.20e), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item T6.25
- **As printed:** `⊢ (A → B) ↔ (¬A ∨ B)`
- **Transcribed as:** ⊢ (((P → Q) → ((¬P) ∨ Q)) ∧ (((¬P) ∨ Q) → (P → Q)))
- **Prompt:** `THM SEQ ( ( ( P > Q ) > ( ( ~ P ) v Q ) ) & ( ( ( ~ P ) v Q ) > ( P > Q ) ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; sentence letters renamed: A→P, B→Q; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 16. Row 59: `textbook_5de54e13acab762c32cc`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 264, sec. 6.2.4 Strategy (contradiction), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.18a
- **As printed:** `A → ¬(B ∧ C ), B → C ⊢ A → ¬B`
- **Transcribed as:** (P → (¬(Q ∧ R))), (Q → R) ⊢ (P → (¬Q))
- **Prompt:** `THM ( P > ( ~ ( Q & R ) ) ) , ( Q > R ) SEQ ( P > ( ~ Q ) ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R
- **Verdict:** [ ] OK   [ ] differs: 

## 17. Row 63: `textbook_8040875d110cc42408af`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 264, sec. 6.2.4 Strategy (contradiction), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.18h
- **As printed:** `A ↔ (¬B ∨ C ), B → C ⊢ A`
- **Transcribed as:** ((P → ((¬Q) ∨ R)) ∧ (((¬Q) ∨ R) → P)), (Q → R) ⊢ P
- **Prompt:** `THM ( ( P > ( ( ~ Q ) v R ) ) & ( ( ( ~ Q ) v R ) > P ) ) , ( Q > R ) SEQ P PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 18. Row 64: `textbook_a9434bd440f14643825c`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 264, sec. 6.2.4 Strategy (contradiction), NDs — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.18j
- **As printed:** `A ↔ ¬(B ↔ ¬C ), ¬(A ∨ B) ⊢ C`
- **Transcribed as:** ((P → (¬((Q → (¬R)) ∧ ((¬R) → Q)))) ∧ ((¬((Q → (¬R)) ∧ ((¬R) → Q))) → P)), (¬(P ∨ Q)) ⊢ R
- **Prompt:** `THM ( ( P > ( ~ ( ( Q > ( ~ R ) ) & ( ( ~ R ) > Q ) ) ) ) & ( ( ~ ( ( Q > ( ~ R ) ) & ( ( ~ R ) > Q ) ) ) > P ) ) , ( ~ ( P v Q ) ) SEQ R PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: A→P, B→Q, C→R; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 

## 19. Row 70: `textbook_d337b29b8e27a24990b3`

- **Source:** Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), 2023 print edition, third revision July 2025 — PDF p. 273, sec. 6.2.5 The system NDs+ (derived rules) — SymLogFullTxt.pdf (PDF page = printed page + 12); item E6.21a
- **As printed:** `(H ∧ G) → (L ∨ K), G ∧ H ⊢ K ∨ L`
- **Transcribed as:** ((P ∧ Q) → (R ∨ S)), (Q ∧ P) ⊢ (S ∨ R)
- **Prompt:** `THM ( ( P & Q ) > ( R v S ) ) , ( Q & P ) SEQ ( S v R ) PRF`
- **Encoding notes:** read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢; starred: the author's answer is published in Answers to Selected Exercises; sentence letters renamed: H→P, G→Q, L→R, K→S
- **Verdict:** [ ] OK   [ ] differs: 

## 20. Row 79: `textbook_66cb15032d000a4c5f35`

- **Source:** The Natural Deduction Pack (Carr), March 2018, for Halbach's The Logic Manual — PDF p. 17, 4.6 Negation — NDpack.pdf; item 14
- **As printed:** `P∨Q,P ↔Q,¬(P∧Q)⊣R`
- **Transcribed as:** (P ∨ Q), ((P → Q) ∧ (Q → P)), (¬(P ∧ Q)) ⊢ R
- **Prompt:** `THM ( P v Q ) , ( ( P > Q ) & ( Q > P ) ) , ( ~ ( P & Q ) ) SEQ R PRF`
- **Encoding notes:** read from the PDF text (pypdf); ⊣ is the pack's turnstile; author's solution length: 6 steps; under the heading 'Ex falso quodlibet'; ↔ written as (A → B) ∧ (B → A)
- **Verdict:** [ ] OK   [ ] differs: 
