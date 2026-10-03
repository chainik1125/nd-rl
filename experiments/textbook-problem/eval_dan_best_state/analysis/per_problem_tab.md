# Per-problem results

Non-sealed sets only (the 72 are counts in the main tab). Rates = fraction of 32 samples accepted: pt = Robbie's pretrain d384 (one seed); rl s0 / s1 = Dan's best-state r8 seeds. ref = reference proof lines. Category: both / RL only / neither (no problem was pretrain-only).


## v0

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| carnap-book 4.1 | P ⊢ P | 1 | 0.062 | 1.0 | 0.969 | both |
| forallx-calgary 2 | ¬¬¬¬P ⊢ P | 3 | 0.531 | 0.844 | 0.844 | both |
| forallx-msu 5 | ⊢ ((P → Q) ∧ (Q → P)) → (Q → P) | 3 | 0.906 | 0.969 | 0.938 | both |
| forallx-calgary 1 | (P → Q) ∧ (Q → P) ⊢ (Q → P) ∧ (P → Q) | 4 | 0.0 | 0.469 | 0.156 | rl only |
| forallx-calgary 5 | ((P → P) ∧ (P → P)) ∧ (Q ∧ ¬Q) ⊢ ⊥ | 5 | 0.0 | 0.156 | 0.438 | rl only |
| forallx-syracuse 2 | (P ∨ Q) → R ⊢ Q → R | 5 | 0.969 | 1.0 | 0.906 | both |
| forallx-msu 4 | P ∧ Q, (Q ∨ R) → S ⊢ S | 5 | 0.938 | 0.688 | 0.031 | both |
| carnap-book 9.3 | ¬P ⊢ ¬(Q ∧ P) | 5 | 0.906 | 1.0 | 1.0 | both |
| forallx-calgary 2 | P → (P ∧ ¬P) ⊢ ¬P | 6 | 0.0 | 0.938 | 0.875 | rl only |
| forallx-calgary 9 | ¬(P ∨ Q), R, R → P ⊢ ⊥ | 6 | 0.656 | 0.875 | 0.531 | both |
| forallx-syracuse 2 | P ∧ Q, R, (Q ∧ R) → S ⊢ S | 6 | 0.781 | 0.594 | 0.281 | both |
| forallx-syracuse 3 | P ∧ Q, (Q ∧ P) → R ⊢ R | 6 | 0.0 | 0.0 | 0.094 | neither |
| forallx-syracuse 16 | ⊢ ¬P → (P → Q) | 6 | 0.25 | 1.0 | 1.0 | both |
| forallx-syracuse 3 | (P ∧ Q) ∨ (P ∧ R) ⊢ P | 6 | 0.219 | 0.375 | 0.844 | both |
| forallx-msu 3 | P ∧ Q, (Q → R) ∧ (R → Q) ⊢ R ∨ S | 6 | 0.0 | 0.0 | 0.094 | neither |
| forallx-msu 2 | P → Q ⊢ (P ∧ R) → (Q ∨ S) | 6 | 0.844 | 0.938 | 0.281 | both |
| forallx-msu 3 | ¬(P ∨ Q) ⊢ Q → P | 6 | 0.938 | 0.812 | 0.969 | both |
| carnap-book 6.5 | (P → Q) → R, Q ⊢ R | 6 | 0.688 | 0.625 | 0.344 | both |
| carnap-book 7.7 | ⊢ (¬P → P) → ¬¬P | 6 | 0.0 | 0.906 | 0.844 | rl only |
| forallx-calgary 5 | (P ∧ Q) ∨ R ⊢ R ∨ Q | 7 | 0.0 | 0.781 | 0.812 | rl only |
| forallx-calgary 4 | P → Q ⊢ ¬(P ∧ ¬Q) | 7 | 0.062 | 0.688 | 0.656 | both |
| forallx-calgary 7 | ¬¬P → ¬Q ⊢ P → ¬Q | 7 | 0.094 | 0.969 | 0.969 | both |
| forallx-syracuse 1 | P ∧ (Q ∧ R) ⊢ (P ∧ Q) ∧ R | 7 | 0.0 | 0.062 | 0.094 | rl only |
| forallx-syracuse 4 | P ∧ Q, Q → (P → R) ⊢ R ∧ Q | 7 | 0.031 | 0.031 | 0.094 | both |
| forallx-syracuse 5 | P → ((Q ∧ P) → R), P ∧ Q ⊢ R | 7 | 0.0 | 0.0 | 0.031 | neither |
| forallx-syracuse 3 | P → Q, Q → R ⊢ P → (Q ∧ R) | 7 | 0.969 | 1.0 | 0.969 | both |
| forallx-syracuse 4 | (P ∧ Q) → R, P → Q ⊢ P → R | 7 | 0.0 | 0.031 | 0.0 | rl only |
| forallx-syracuse 5 | P → Q ⊢ (P ∧ R) → (Q ∧ R) | 7 | 0.594 | 0.875 | 0.188 | both |
| forallx-syracuse 6 | P → Q, Q → (R ∧ P) ⊢ (P → Q) ∧ (Q → P) | 7 | 0.0 | 0.062 | 0.156 | rl only |
| forallx-syracuse 2 | P → ¬Q ⊢ Q → ¬P | 7 | 0.094 | 0.938 | 0.969 | both |
| forallx-syracuse 7 | P → Q, Q → ¬P ⊢ ¬P | 7 | 0.0 | 0.25 | 0.375 | rl only |
| forallx-syracuse 18 | ⊢ (¬P → P) → P | 7 | 0.0 | 0.031 | 0.406 | rl only |
| forallx-syracuse 8 | P ∨ (¬P ∧ Q) ⊢ P ∨ Q | 7 | 0.031 | 0.375 | 0.5 | both |
| forallx-syracuse 1 | (P ∨ Q) → R, ¬R ⊢ ¬P | 7 | 0.0 | 0.625 | 0.594 | rl only |
| forallx-msu 9 | P → Q, R ∧ ¬S, (¬S → (Q → P)) ∧ ((Q → P) → ¬S) ⊢ (P → Q) ∧ (Q → P) | 7 | 0.0 | 0.094 | 0.094 | rl only |
| forallx-msu 12 | P → Q, Q → P, (Q → ((Q → P) ∧ (P → Q))) ∧ (((Q → P) ∧ (P → Q)) → Q) ⊢ P | 7 | 0.0 | 0.0 | 0.0 | neither |
| forallx-msu 2 | ¬(P ∨ Q) ⊢ R → ¬P | 7 | 0.781 | 0.938 | 0.969 | both |
| carnap-book 4.4 | ¬P, ¬Q → P ⊢ Q | 7 | 0.0 | 0.469 | 0.656 | rl only |
| carnap-book 4.5 | P, ¬Q → ¬P ⊢ Q | 7 | 0.0 | 0.469 | 0.5 | rl only |
| carnap-book 6.2 | P → Q ⊢ (Q → R) → (P → R) | 7 | 0.188 | 0.031 | 0.406 | both |
| carnap-book 8.2 | P → (P → Q), (P → Q) → Q ⊢ Q | 7 | 0.0 | 0.188 | 0.312 | rl only |
| forallx-calgary 4 | P ∧ Q ⊢ (P → Q) ∧ (Q → P) | 8 | 0.0 | 0.688 | 0.688 | rl only |
| forallx-calgary 8 | (P ∧ Q) ∨ (Q ∧ R), Q → S ⊢ S | 8 | 0.0 | 0.344 | 0.062 | rl only |
| forallx-syracuse 1 | P → (Q ∧ R), P → S ⊢ P → (S ∧ R) | 8 | 0.0 | 0.781 | 0.75 | rl only |
| forallx-syracuse 2 | ⊢ (P → Q) → ((Q → R) → (P → R)) | 8 | 0.0 | 0.031 | 0.312 | rl only |
| forallx-syracuse 3 | ⊢ (P → (Q → R)) → (Q → (P → R)) | 8 | 0.0 | 0.875 | 0.719 | rl only |
| forallx-syracuse 7 | (P → (Q ∧ R)) ∧ ((Q ∧ R) → P), R → P ⊢ (P → R) ∧ (R → P) | 8 | 0.0 | 0.031 | 0.5 | rl only |
| forallx-syracuse 8 | (P ∧ Q) → ¬P ⊢ P → ¬Q | 8 | 0.0 | 0.812 | 0.938 | rl only |
| forallx-syracuse 12 | P → Q ⊢ (P ∨ (R ∧ Q)) → Q | 8 | 0.0 | 0.188 | 0.5 | rl only |
| forallx-syracuse 15 | P ∨ Q, Q → R ⊢ P ∨ R | 8 | 0.0 | 0.562 | 0.094 | rl only |
| forallx-msu 1 | ¬P → (Q ∨ P), ¬P ⊢ Q | 8 | 0.0 | 0.969 | 0.969 | rl only |
| forallx-msu 2 | P → (Q ∨ ¬P), P ⊢ Q | 8 | 0.0 | 1.0 | 0.938 | rl only |
| forallx-msu 10 | (P ∨ Q) → R, (S → R) ∧ (R → S), P ⊢ P ∧ S | 8 | 0.0 | 0.156 | 0.0 | rl only |
| forallx-calgary 4 | (P → ¬P) ∧ (¬P → P) ⊢ ⊥ | 9 | 0.0 | 0.219 | 0.031 | rl only |
| forallx-syracuse 10 | (P ∧ ¬Q) → Q ⊢ P → Q | 9 | 0.0 | 0.0 | 0.219 | neither |
| carnap-book 5.10 | P → ¬Q, R → Q ⊢ P → ¬R | 9 | 0.0 | 0.594 | 0.844 | rl only |
| carnap-book 7.10 | P → Q, (R → Q) → ¬P ⊢ ¬P | 9 | 0.0 | 0.062 | 0.0 | rl only |
| forallx-calgary 2 | P ∨ (Q → P) ⊢ ¬P → ¬Q | 10 | 0.0 | 0.875 | 0.656 | rl only |
| forallx-calgary 2 | P ∧ (¬Q → ¬P) ⊢ (Q ∧ P) ∨ ¬P | 10 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 3 | P ∨ Q ⊢ ¬(¬P ∧ ¬Q) | 10 | 0.0 | 0.375 | 0.969 | rl only |
| forallx-msu 6 | P ∧ (Q ∨ R), P → ¬R ⊢ Q | 10 | 0.0 | 0.844 | 0.594 | rl only |
| carnap-book 7.8 | P → (Q → R), Q → ¬R ⊢ P → ¬Q | 10 | 0.0 | 0.469 | 0.188 | rl only |
| carnap-book 9.8 | (P ∧ Q) → R ⊢ (P ∧ ¬R) → ¬Q | 10 | 0.0 | 0.219 | 0.031 | rl only |
| forallx-calgary 9 | P ∧ (Q ∨ R), P → ¬R ⊢ Q ∨ S | 11 | 0.0 | 0.531 | 0.531 | rl only |
| forallx-calgary 3 | ⊢ ((P → Q) ∧ (R → Q)) → ((P ∨ R) → Q) | 11 | 0.0 | 0.094 | 0.625 | rl only |
| forallx-calgary 3 | P → Q ⊢ ¬P ∨ Q | 11 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 14 | P → Q, R → (S ∧ Q) ⊢ (P ∨ R) → (S ∨ Q) | 11 | 0.0 | 0.0 | 0.125 | neither |
| forallx-syracuse 4 | ¬P ∨ (Q → R) ⊢ (P ∧ Q) → R | 11 | 0.0 | 0.281 | 0.344 | rl only |
| forallx-syracuse 3 | ¬(P → (Q ∨ ¬R)) ⊢ (Q ∨ R) → P | 11 | 0.0 | 0.0 | 0.0 | neither |
| forallx-msu 4 | ⊢ ¬((P ∨ Q) ∧ (¬P ∧ ¬Q)) | 11 | 0.0 | 0.281 | 0.25 | rl only |
| carnap-book 7.4 | P → Q, Q → R, R → S, P ⊢ ¬(P → ¬S) | 11 | 0.0 | 0.0 | 0.0 | neither |
| carnap-book 8.5 | ⊢ P → ((¬Q → ¬R) → (R → Q)) | 11 | 0.0 | 0.5 | 0.188 | rl only |
| carnap-book 8.7 | P → Q, Q → ¬R, S → P, R → S ⊢ ¬R | 11 | 0.0 | 0.0 | 0.0 | neither |
| carnap-book 9.6 | ⊢ ((P ∧ Q) → (Q ∧ P)) ∧ ((Q ∧ P) → (P ∧ Q)) | 11 | 0.0 | 0.062 | 0.031 | rl only |
| carnap-book 9.9 | (P ∧ ¬Q) → ¬R ⊢ (P ∧ R) → Q | 11 | 0.0 | 0.094 | 0.156 | rl only |
| forallx-calgary 11 | ¬¬((P → ¬P) ∧ (¬P → P)), (Q ∨ P) ∨ Q ⊢ (Q → P) ∧ Q | 12 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 8 | P ∨ (Q ∨ R) ⊢ (P ∨ Q) ∨ R | 12 | 0.0 | 0.031 | 0.0 | rl only |
| forallx-syracuse 13 | P → Q, R → (R → S) ⊢ (R ∨ P) → (Q ∨ S) | 12 | 0.0 | 0.031 | 0.0 | rl only |
| forallx-msu 7 | ((P ∨ ¬Q) → R) ∧ (R → (P ∨ ¬Q)), R ∧ Q ⊢ P ∧ R | 12 | 0.0 | 0.031 | 0.0 | rl only |
| forallx-msu 8 | (P ∧ Q) → R, P ∨ ¬S, S ∧ Q ⊢ R | 12 | 0.0 | 0.0 | 0.5 | neither |
| carnap-book 7.3 | ¬P → Q, R → ¬Q, ¬R → P ⊢ ¬¬P | 12 | 0.0 | 0.25 | 0.0 | rl only |
| carnap-book 7.5 | (P → Q) → R, S → (P → Q), ¬S → R ⊢ ¬¬R | 12 | 0.0 | 0.0 | 0.0 | neither |
| carnap-book 7.9 | P → Q, Q → R, R → S ⊢ P → ¬(P → ¬S) | 12 | 0.0 | 0.0 | 0.031 | neither |
| carnap-book 9.7 | P ∨ ¬Q, Q ∨ R, ¬P ⊢ R | 12 | 0.0 | 0.094 | 0.0 | rl only |
| forallx-calgary 7 | ¬P → Q, P → R ⊢ Q ∨ R | 13 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 4 | ⊢ ¬(P → Q) → (P ∧ ¬Q) | 13 | 0.0 | 0.062 | 0.0 | rl only |
| forallx-calgary 3 | ⊢ (P → (P ∨ (Q ∧ ¬Q))) ∧ ((P ∨ (Q ∧ ¬Q)) → P) | 13 | 0.0 | 0.156 | 0.719 | rl only |
| forallx-syracuse 10 | (P ∨ Q) ∨ R, (R → Q) ∧ (Q → R) ⊢ R ∨ P | 13 | 0.0 | 0.312 | 0.0 | rl only |
| carnap-book 8.8 | ¬P → ¬¬Q, ¬¬R → ¬P, ¬R → Q ⊢ Q | 13 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 1 | P ∨ Q, Q ∨ R, ¬Q ⊢ P ∧ R | 14 | 0.0 | 0.031 | 0.188 | rl only |
| forallx-syracuse 3 | ¬(¬P ∧ ¬Q) ⊢ P ∨ Q | 14 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 6 | (P → Q) ∧ (Q → P), (Q → R) ∧ (R → Q) ⊢ (P → R) ∧ (R → P) | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 4 | ⊢ ¬(P ∧ Q) → (¬P ∨ ¬Q) | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 6 | (¬P → Q) ∧ (Q → ¬P) ⊢ ¬((P → Q) ∧ (Q → P)) | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 11 | (P → ¬Q) ∧ (¬Q → P) ⊢ ¬((P → Q) ∧ (Q → P)) | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 8 | P ∨ Q ⊢ P ∨ (¬P ∧ Q) | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-msu 7 | P ∧ (Q ∧ R) ⊢ (P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P) | 15 | 0.0 | 0.375 | 0.781 | rl only |
| carnap-book 8.9 | P → ¬P, Q → P, R → Q, ¬R → Q ⊢ S | 15 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 3 | (P → Q) ∧ (Q → P) ⊢ ((P ∧ R) → (Q ∧ R)) ∧ ((Q ∧ R) → (P ∧ R)) | 16 | 0.0 | 0.0 | 0.0 | neither |
| forallx-syracuse 4 | ((P ∧ Q) → (P ∧ R)) ∧ ((P ∧ R) → (P ∧ Q)) ⊢ P → ((Q → R) ∧ (R → Q)) | 16 | 0.0 | 0.0 | 0.0 | neither |
| forallx-calgary 3 | (P ∨ Q) ∧ (R ∨ S), Q → S, ¬S ⊢ P ∧ R | 17 | 0.0 | 0.031 | 0.094 | rl only |
| forallx-syracuse 8 | (P ∧ Q) ∨ ¬(R → ¬S), ¬Q ⊢ R ∧ S | 17 | 0.0 | 0.0 | 0.062 | neither |
| forallx-calgary 4 | (P ∨ Q) ∨ (R ∨ S), Q → R, ¬S ⊢ P ∨ R | 18 | 0.0 | 0.062 | 0.0 | rl only |
| forallx-syracuse 7 | ¬((P → Q) ∧ (Q → P)) ⊢ (P → ¬Q) ∧ (¬Q → P) | 29 | 0.0 | 0.0 | 0.0 | neither |

## v1

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| roy E6.10f | P ∧ Q ⊢ ((R → S) ∧ (S → R)) ∨ Q | 3 | 0.938 | 0.969 | 1.0 | both |
| roy E6.15s | ⊢ ((P → Q) ∧ (Q → P)) → (P → Q) | 3 | 0.844 | 0.594 | 1.0 | both |
| roy T6.28 | ⊢ (((P → Q) ∧ (Q → P)) → ((P → Q) ∧ (Q → P))) ∧ (((P → Q) ∧ (Q → P)) → ((P → Q) ∧ (Q → P))) | 3 | 0.0 | 0.875 | 1.0 | rl only |
| roy E6.7c | P ∧ (P → (P ∧ Q)) ⊢ Q | 5 | 0.406 | 1.0 | 0.781 | both |
| roy E6.13l | ⊢ ((P → Q) ∧ (Q → P)) → ((Q → P) ∧ (P → Q)) | 5 | 0.0 | 0.0 | 0.156 | neither |
| roy E6.15c | ¬(P ∧ Q), (¬(P ∧ Q) → R) ∧ (R → ¬(P ∧ Q)) ⊢ P ∨ R | 5 | 0.156 | 0.531 | 0.875 | both |
| carr 9 | ⊢ (P ∧ Q) → ((P → Q) → P) | 5 | 1.0 | 1.0 | 1.0 | both |
| roy E6.13b | (P → (P ∨ Q)) ∧ ((P ∨ Q) → P) ⊢ Q → P | 6 | 0.031 | 0.812 | 0.719 | both |
| roy E6.15a | (P → (P → Q)) ∧ ((P → Q) → P) ⊢ P → Q | 6 | 0.0 | 0.969 | 1.0 | rl only |
| roy E6.15t | ⊢ (P ∧ ¬P) → (Q ∧ ¬Q) | 6 | 0.0 | 0.812 | 0.625 | rl only |
| roy E6.18b | ⊢ ¬(P → P) → P | 6 | 0.0 | 0.781 | 0.938 | rl only |
| roy E6.7d | P ∧ Q, Q → (R ∧ S) ⊢ P ∧ S | 7 | 0.0 | 0.531 | 0.688 | rl only |
| roy E6.7f | P, (P ∧ Q) → (R ∧ S) ⊢ Q → R | 7 | 0.0 | 0.875 | 0.75 | rl only |
| roy E6.7h | P → Q, Q → R ⊢ (P ∧ S) → R | 7 | 0.0 | 0.312 | 0.219 | rl only |
| roy E6.7l | P → Q ⊢ (R → P) → (R → Q) | 7 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.10h | P, (Q ∨ R) → ¬P ⊢ ¬Q | 7 | 0.094 | 0.719 | 0.188 | both |
| roy E6.10n | P ∨ ¬¬Q ⊢ P ∨ Q | 7 | 0.344 | 0.438 | 0.594 | both |
| roy E6.10o | P ∨ Q, ¬Q ⊢ P | 7 | 0.0 | 0.906 | 0.781 | rl only |
| roy T6.9 | (P → Q) ∧ (Q → P), ¬P ⊢ ¬Q | 7 | 0.0 | 0.531 | 0.625 | rl only |
| roy T6.10 | (P → Q) ∧ (Q → P), ¬Q ⊢ ¬P | 7 | 0.0 | 0.25 | 0.562 | rl only |
| roy T6.18 | ⊢ (P → (P ∧ P)) ∧ ((P ∧ P) → P) | 7 | 0.062 | 0.875 | 0.938 | both |
| roy E6.15b | (P ∨ Q) → ((Q → R) ∧ (R → Q)), Q ⊢ Q ∧ R | 7 | 0.0 | 0.156 | 0.062 | rl only |
| roy E6.15w | ⊢ P → (Q → (P → Q)) | 7 | 0.062 | 0.656 | 0.875 | both |
| roy E6.15x | ⊢ ¬P → ((Q ∧ P) → R) | 7 | 0.0 | 0.969 | 0.875 | rl only |
| roy E6.7g | P → Q, P → (Q → R) ⊢ P → (Q ∧ R) | 8 | 0.0 | 0.562 | 0.5 | rl only |
| roy E6.7k | P → Q, Q → R, R → S ⊢ P → S | 8 | 0.0 | 0.062 | 0.062 | rl only |
| roy E6.10e | ¬P → Q, Q → P ⊢ P | 8 | 0.0 | 0.031 | 0.219 | rl only |
| roy E6.15i | P → (Q ∧ R), ¬R ⊢ ¬(P ∧ S) | 8 | 0.0 | 0.656 | 0.156 | rl only |
| roy E6.15u | ⊢ (P → Q) → ((R → P) → (R → Q)) | 8 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.15v | ⊢ ((P → Q) ∧ ¬Q) → ¬P | 8 | 0.0 | 0.656 | 0.625 | rl only |
| roy E6.7j | P ∧ Q, (P → R) ∧ (Q → S) ⊢ R ∧ S | 9 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.7n | P → (Q → R), S → Q ⊢ P → (S → R) | 9 | 0.0 | 0.188 | 0.0 | rl only |
| roy E6.10i | P ∨ Q, P → Q, Q → P ⊢ P ∧ Q | 9 | 0.0 | 1.0 | 0.969 | rl only |
| roy E6.13k | ⊢ (¬¬P → P) ∧ (P → ¬¬P) | 9 | 0.031 | 0.969 | 0.969 | both |
| roy E6.13n | ¬P → Q, P → ¬Q ⊢ (¬P → Q) ∧ (Q → ¬P) | 9 | 0.0 | 0.875 | 0.656 | rl only |
| roy T6.17 | ⊢ (P → ¬¬P) ∧ (¬¬P → P) | 9 | 0.0 | 0.906 | 0.969 | rl only |
| roy T6.19 | ⊢ (P → (P ∨ P)) ∧ ((P ∨ P) → P) | 9 | 0.0 | 0.875 | 1.0 | rl only |
| roy E6.15l | (P ∧ ¬Q) → ¬P ⊢ P → Q | 9 | 0.0 | 0.062 | 0.25 | rl only |
| roy E6.15y | ⊢ (P → Q) → (¬Q → ¬(P ∧ R)) | 9 | 0.0 | 0.25 | 0.0 | rl only |
| carr 7 | P → Q ⊢ ((P ∧ Q) → P) ∧ (P → (P ∧ Q)) | 9 | 0.0 | 0.812 | 0.938 | rl only |
| roy E6.10j | P → Q, (Q ∨ R) → S, S → ¬P ⊢ ¬P | 10 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.3 | ⊢ (¬P → ¬Q) → ((¬P → Q) → P) | 10 | 0.0 | 0.531 | 0.281 | rl only |
| roy E6.15f | (P ∧ Q) → (R ∧ S) ⊢ ((P ∧ Q) → R) ∧ ((P ∧ Q) → S) | 10 | 0.0 | 0.938 | 1.0 | rl only |
| roy E6.15k | P → (Q → R) ⊢ ¬R → ¬(P ∧ Q) | 10 | 0.0 | 0.25 | 0.031 | rl only |
| roy E6.15n | (¬P → Q) ∧ (Q → ¬P), R → P, Q ∧ R ⊢ ¬S | 10 | 0.0 | 0.031 | 0.0 | rl only |
| roy E6.18a | P → ¬(Q ∧ R), Q → R ⊢ P → ¬Q | 10 | 0.0 | 0.656 | 0.594 | rl only |
| roy E6.18d | ¬(P ∧ Q), ¬(P ∧ ¬Q) ⊢ ¬P | 10 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18g | P ∨ ¬Q, ¬P ∨ ¬Q ⊢ ¬Q | 10 | 0.0 | 0.312 | 0.031 | rl only |
| roy E6.13o | P, Q ⊢ (¬P → ¬Q) ∧ (¬Q → ¬P) | 11 | 0.0 | 0.719 | 0.875 | rl only |
| roy T6.12 | ⊢ (((P → Q) ∧ (Q → P)) → ((Q → P) ∧ (P → Q))) ∧ (((Q → P) ∧ (P → Q)) → ((P → Q) ∧ (Q → P))) | 11 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18l | ¬(P → Q), ¬(Q → R) ⊢ ¬S | 11 | 0.0 | 0.156 | 0.312 | rl only |
| roy E6.21a | (P ∧ Q) → (R ∨ S), Q ∧ P ⊢ S ∨ R | 11 | 0.0 | 0.0 | 0.0 | neither |
| carr 6 | P → Q ⊢ ((P ∨ Q) → Q) ∧ (Q → (P ∨ Q)) | 11 | 0.0 | 0.5 | 0.312 | rl only |
| roy E6.18p | P ∨ Q, ¬Q ∨ R, ¬R ⊢ P | 12 | 0.0 | 0.0 | 0.281 | neither |
| roy E6.13c | (P → Q) ∧ (Q → P), (Q → R) ∧ (R → Q), (R → S) ∧ (S → R), ¬P ⊢ ¬S | 13 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.13e | (P → (Q ∧ R)) ∧ ((Q ∧ R) → P), Q ⊢ (P → R) ∧ (R → P) | 13 | 0.0 | 0.344 | 0.219 | rl only |
| roy E6.15p | (P ∨ Q) ∨ R, (Q → R) ∧ (R → Q) ⊢ R ∨ P | 13 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18m | P → ¬Q, ¬(R ∧ P) ⊢ (Q ∨ R) → ¬P | 13 | 0.0 | 0.344 | 0.656 | rl only |
| carr 14 | P ∨ Q, (P → Q) ∧ (Q → P), ¬(P ∧ Q) ⊢ R | 13 | 0.0 | 0.0 | 0.062 | neither |
| roy E6.18h | (P → (¬Q ∨ R)) ∧ ((¬Q ∨ R) → P), Q → R ⊢ P | 14 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21b | ⊢ ((P ∧ Q) → (Q ∧ P)) ∧ (¬(P ∧ Q) → ¬(Q ∧ P)) | 14 | 0.0 | 0.0 | 0.625 | neither |
| roy T6.13 | ⊢ ((P ∨ Q) → (Q ∨ P)) ∧ ((Q ∨ P) → (P ∨ Q)) | 15 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18j | (P → ¬((Q → ¬R) ∧ (¬R → Q))) ∧ (¬((Q → ¬R) ∧ (¬R → Q)) → P), ¬(P ∨ Q) ⊢ R | 15 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.20a | (P ∨ Q) → (P ∨ R) ⊢ P ∨ (Q → R) | 16 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.15 | ⊢ ((P → (Q → R)) → ((P ∧ Q) → R)) ∧ (((P ∧ Q) → R) → (P → (Q → R))) | 17 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.16 | ⊢ ((P ∧ (Q ∧ R)) → ((P ∧ Q) ∧ R)) ∧ (((P ∧ Q) ∧ R) → (P ∧ (Q ∧ R))) | 17 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18q | (¬P ∨ Q) ∨ R, R → ¬S ⊢ (P ∧ S) → Q | 17 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.14 | ⊢ ((P → Q) → (¬Q → ¬P)) ∧ ((¬Q → ¬P) → (P → Q)) | 18 | 0.0 | 0.219 | 0.094 | rl only |
| roy E6.15o | (¬P → ¬Q) ∧ (¬Q → ¬P) ⊢ (P → Q) ∧ (Q → P) | 18 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21f | P → (Q → R), ¬S → (P ∨ R), P → Q ⊢ S ∨ R | 19 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21h | ¬(P ∨ ¬Q) ∨ (¬Q → ¬R), Q → P, P → R ⊢ (Q → R) ∧ (R → Q) | 20 | 0.0 | 0.031 | 0.0 | rl only |
| roy E6.13h | (P → Q) ∧ (Q → P), (R → S) ∧ (S → R) ⊢ ((P ∧ R) → (Q ∧ S)) ∧ ((Q ∧ S) → (P ∧ R)) | 21 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.24 | ⊢ ((¬P → Q) → (P ∨ Q)) ∧ ((P ∨ Q) → (¬P → Q)) | 22 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.25 | ⊢ ((P → Q) → (¬P ∨ Q)) ∧ ((¬P ∨ Q) → (P → Q)) | 22 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21k | ⊢ ((P ∨ (Q → (P → Q))) → (P ∨ ((¬P ∨ ¬Q) ∨ Q))) ∧ ((P ∨ ((¬P ∨ ¬Q) ∨ Q)) → (P ∨ (Q → (P → Q)))) | 22 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.23 | ⊢ (¬(P ∨ Q) → (¬P ∧ ¬Q)) ∧ ((¬P ∧ ¬Q) → ¬(P ∨ Q)) | 23 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.26 | ⊢ ((P ∧ (Q ∨ R)) → ((P ∧ Q) ∨ (P ∧ R))) ∧ (((P ∧ Q) ∨ (P ∧ R)) → (P ∧ (Q ∨ R))) | 25 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.20 | ⊢ ((P ∨ (Q ∨ R)) → ((P ∨ Q) ∨ R)) ∧ (((P ∨ Q) ∨ R) → (P ∨ (Q ∨ R))) | 27 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21i | ⊢ ((P ∨ (Q ∨ R)) → (R ∨ (Q ∨ P))) ∧ ((R ∨ (Q ∨ P)) → (P ∨ (Q ∨ R))) | 27 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.27 | ⊢ ((P ∨ (Q ∧ R)) → ((P ∨ Q) ∧ (P ∨ R))) ∧ (((P ∨ Q) ∧ (P ∨ R)) → (P ∨ (Q ∧ R))) | 28 | 0.0 | 0.0 | 0.031 | neither |

## hand14

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| roy E6.21l | ⊢ (¬P → (¬Q → R)) → ((P ∨ Q) ∨ (¬¬Q ∨ R)) | 20 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18o | (P → Q) ∧ (Q → P), (Q → ¬R) ∧ (¬R → Q) ⊢ ¬((P → R) ∧ (R → P)) | 23 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P9 | ⊢ (((P ∨ Q) ∧ (¬P ∨ Q)) ∧ (P ∨ ¬Q)) → ¬(¬P ∨ ¬Q) | 26 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21m | ⊢ (((¬P → ¬P) ∧ (¬P → ¬P)) → ((¬(¬P → P) → (P → ¬P)) ∧ ((P → ¬P) → ¬(¬P → P)))) ∧ (((¬(¬P → P) → (P → ¬P)) ∧ ((P → ¬P) → ¬(¬P → P))) → ((¬P → ¬P) ∧ (¬P → ¬P))) | 29 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.18i | (P → Q) ∧ (Q → P) ⊢ (((R → P) ∧ (P → R)) → ((R → Q) ∧ (Q → R))) ∧ (((R → Q) ∧ (Q → R)) → ((R → P) ∧ (P → R))) | 30 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.20c | (((P → Q) ∧ (Q → P)) → ((R → S) ∧ (S → R))) ∧ (((R → S) ∧ (S → R)) → ((P → Q) ∧ (Q → P))) ⊢ ((P → R) ∧ (R → P)) → (Q → S) | 43 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P14 | ⊢ (((P → Q) ∧ (Q → P)) → ((Q ∨ ¬P) ∧ (¬Q ∨ P))) ∧ (((Q ∨ ¬P) ∧ (¬Q ∨ P)) → ((P → Q) ∧ (Q → P))) | 45 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.29 | ⊢ (((P → Q) ∧ (Q → P)) → ((P ∧ Q) ∨ (¬P ∧ ¬Q))) ∧ (((P ∧ Q) ∨ (¬P ∧ ¬Q)) → ((P → Q) ∧ (Q → P))) | 46 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.21j | ⊢ ((P → ((Q → R) ∧ (R → Q))) → (P → ((¬Q ∨ R) ∧ (¬R ∨ Q)))) ∧ ((P → ((¬Q ∨ R) ∧ (¬R ∨ Q))) → (P → ((Q → R) ∧ (R → Q)))) | 51 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.20e | (P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P) ⊢ (((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P))) | 51 | 0.0 | 0.0 | 0.0 | neither |
| roy E6.20d | ¬((P → Q) ∧ (Q → P)), ¬((Q → R) ∧ (R → Q)), ¬((R → P) ∧ (P → R)) ⊢ ¬S | 71 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P17 | ⊢ (((P ∧ (Q → R)) → S) → (((¬P ∨ Q) ∨ S) ∧ ((¬P ∨ ¬R) ∨ S))) ∧ ((((¬P ∨ Q) ∨ S) ∧ ((¬P ∨ ¬R) ∨ S)) → ((P ∧ (Q → R)) → S)) | 73 | 0.0 | 0.0 | 0.0 | neither |
| roy T6.30 | ⊢ (((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P)) → ((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P))))) ∧ (((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P)))) → ((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P))) | 102 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P12 | ⊢ (((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P)))) → ((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P))) ∧ (((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P)) → ((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P))))) | 102 | 0.0 | 0.0 | 0.0 | neither |

## batch3

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| oxford iv | ⊢ (P → ¬P) → ¬P | 6 | 0.0 | 0.906 | 0.875 | rl only |
| forallx-r3 3 | P → (Q → R), P ∧ Q ⊢ R | 6 | 0.0 | 0.625 | 0.25 | rl only |
| forallx-r3 4 | P ∧ Q, P → R, Q → S ⊢ R ∧ S | 8 | 0.0 | 0.0 | 0.031 | neither |
| pelletier P3 | ⊢ ¬(P → Q) → (Q → P) | 9 | 0.0 | 0.969 | 0.938 | rl only |
| oxford ii | ¬((P → ¬Q) ∨ ¬R) ⊢ Q | 9 | 0.0 | 0.0 | 0.188 | neither |
| forallx-r3 7 | ¬P → Q, ¬P ∨ R, ¬R ⊢ Q | 9 | 0.0 | 0.625 | 0.875 | rl only |
| pelletier P7 | ⊢ P ∨ ¬¬¬P | 10 | 0.0 | 0.0 | 0.0 | neither |
| forallx-r3 8 | P ∧ (¬P ∨ Q), Q → R ⊢ R | 10 | 0.0 | 0.375 | 0.5 | rl only |
| forallx-r3 7 | ¬P ∨ Q, P → R, P ⊢ Q ∧ R | 10 | 0.0 | 0.875 | 0.969 | rl only |
| forallx-r3 12 | P → Q, ¬Q ∨ R, ¬R ⊢ ¬P | 11 | 0.0 | 0.812 | 0.875 | rl only |
| forallx-r3 10 | P → Q, P ∨ R, R → S, (Q ∨ S) → ¬R ⊢ P | 12 | 0.0 | 0.0 | 0.0 | neither |
| forallx-r3 5 | (P ∨ Q) ∨ R, ¬R, ¬Q ⊢ P ∨ S | 14 | 0.0 | 0.312 | 0.562 | rl only |
| oxford iii | (P ∨ ¬Q) → R ⊢ ¬R → (¬P ∧ Q) | 15 | 0.0 | 0.719 | 0.75 | rl only |
| pelletier P5 | ⊢ ((P ∨ Q) → (P ∨ R)) → (P ∨ (Q → R)) | 17 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P10 | P → Q, Q → (R ∧ P), R → (P ∨ Q) ⊢ (R → P) ∧ (P → R) | 17 | 0.0 | 0.031 | 0.062 | rl only |
| forallx-r3 5 | (P ∧ Q) ∨ R, ¬R, ¬(¬P ∨ ¬Q) → S ⊢ S | 17 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P4 | ⊢ ((¬P → Q) → (¬Q → P)) ∧ ((¬Q → P) → (¬P → Q)) | 19 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P9 | ⊢ (((P ∨ Q) ∧ (¬P ∨ Q)) ∧ (P ∨ ¬Q)) → ¬(¬P ∨ ¬Q) | 26 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P14 | ⊢ (((P → Q) ∧ (Q → P)) → ((Q ∨ ¬P) ∧ (¬Q ∨ P))) ∧ (((Q ∨ ¬P) ∧ (¬Q ∨ P)) → ((P → Q) ∧ (Q → P))) | 45 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P17 | ⊢ (((P ∧ (Q → R)) → S) → (((¬P ∨ Q) ∨ S) ∧ ((¬P ∨ ¬R) ∨ S))) ∧ ((((¬P ∨ Q) ∨ S) ∧ ((¬P ∨ ¬R) ∨ S)) → ((P ∧ (Q → R)) → S)) | 73 | 0.0 | 0.0 | 0.0 | neither |
| pelletier P12 | ⊢ (((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P)))) → ((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P))) ∧ (((P → ((Q → R) ∧ (R → Q))) ∧ (((Q → R) ∧ (R → Q)) → P)) → ((((P → Q) ∧ (Q → P)) → R) ∧ (R → ((P → Q) ∧ (Q → P))))) | 102 | 0.0 | 0.0 | 0.0 | neither |

## lean_v2

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| lean4game-logic lean_lean4game-logic_010 | (P → Q) ∧ (Q → P), (¬(((R → (S ∨ ¬Q)) ∧ ((Q ∨ R) → ¬S)) → (Q → S)) → (R ∧ (S ∧ ¬Q))) ∧ ((R ∧ (S ∧ ¬Q)) → ¬(((R → (S ∨ ¬Q)) ∧ ((Q ∨ R) → ¬S)) → (Q → S))) ⊢ (¬(((R → (S ∨ ¬P)) ∧ ((P ∨ R) → ¬S)) → (P → S)) → (R ∧ (S ∧ ¬P))) ∧ ((R ∧ (S ∧ ¬P)) → ¬(((R → (S ∨ ¬P)) ∧ ((P ∨ R) → ¬S)) → (P → S))) |  | 0.0 | 0.0 | 0.0 | neither |
| lean4game-logic lean_lean4game-logic_012 | ⊢ ((((P ∧ Q) → (R ∧ Q)) ∧ ((R ∧ Q) → (P ∧ Q))) → (Q → ((P → R) ∧ (R → P)))) ∧ ((Q → ((P → R) ∧ (R → P))) → (((P ∧ Q) → (R ∧ Q)) ∧ ((R ∧ Q) → (P ∧ Q)))) |  | 0.0 | 0.0 | 0.0 | neither |
| fm2024 lean_fm2024_025 | ⊢ (((P → (Q → Q)) → ((P → Q) → Q)) → R) → (((((P → P) → Q) → (P → (P → Q))) → R) → ((((P → (P → Q)) → ((P → P) → Q)) → R) → R)) |  | 0.0 | 0.0 | 0.0 | neither |
| lean4game-logic lean_lean4game-logic_006 | P → Q, Q → P ⊢ (P → Q) ∧ (Q → P) | 3 | 1.0 | 0.875 | 0.906 | both |
| fm2024 lean_fm2024_031 | ⊢ ⊥ → (P ∧ ⊥) | 3 | 0.969 | 0.906 | 0.969 | both |
| tpil4 lean_tpil4_044 | P ∧ Q ⊢ Q ∧ (P ∧ Q) | 3 | 1.0 | 0.781 | 1.0 | both |
| lean4game-logic lean_lean4game-logic_002 | P ∧ Q, R ∧ S ⊢ P ∧ S | 5 | 0.344 | 0.844 | 0.812 | both |
| lean4game-logic lean_lean4game-logic_014 | P ⊢ ¬(¬P ∧ Q) | 5 | 0.344 | 0.969 | 0.938 | both |
| logic_and_proof lean_logic_and_proof_059 | ⊢ (P ∧ ¬Q) → (¬Q ∧ P) | 5 | 0.312 | 0.969 | 0.875 | both |
| love lean_love_064 | P, Q, P → Q, Q → R ⊢ R | 5 | 0.0 | 0.0 | 0.0 | neither |
| love lean_love_070 | P ∧ (Q ∧ R) ⊢ P ∧ R | 5 | 0.156 | 0.719 | 0.344 | both |
| lean4game-logic lean_lean4game-logic_003 | (P ∧ (((P ∧ Q) ∧ P) ∧ (P ∧ (P ∧ P)))) ∧ ((P ∧ P) ∧ P) ⊢ Q | 6 | 0.0 | 0.0 | 0.0 | neither |
| logic_and_proof lean_logic_and_proof_060 | ⊢ (P ∧ ((Q → P) ∧ (P → Q))) → Q | 6 | 0.0 | 0.125 | 0.031 | rl only |
| love lean_love_069 | ⊢ ¬¬(¬¬P → P) | 6 | 0.188 | 0.938 | 1.0 | both |
| love lean_love_071 | P ∧ ((Q ∧ R) ∧ S) ⊢ Q ∧ S | 6 | 0.031 | 0.531 | 0.031 | both |
| lean4game-logic lean_lean4game-logic_001 | P, Q, R, S ⊢ (P ∧ Q) ∧ (R ∧ S) | 7 | 0.0 | 0.0 | 0.0 | neither |
| lean4game-logic lean_lean4game-logic_008 | (P → Q) ∧ (Q → P), R → P ⊢ R → Q | 7 | 0.0 | 0.5 | 0.219 | rl only |
| lean4game-logic lean_lean4game-logic_009 | (P → Q) ∧ (Q → P), P → R ⊢ Q → R | 7 | 0.0 | 0.219 | 0.094 | rl only |
| lean4game-logic lean_lean4game-logic_015 | P → ¬Q ⊢ ¬(P ∧ Q) | 7 | 0.0 | 0.812 | 0.75 | rl only |
| fm2024 lean_fm2024_032 | ⊢ (P ∧ Q) → ((Q ∧ R) → (P ∧ R)) | 7 | 0.031 | 0.844 | 0.812 | both |
| fm2024 lean_fm2024_036 | ⊢ (⊥ → (P ∧ ⊥)) ∧ ((P ∧ ⊥) → ⊥) | 7 | 0.0 | 0.875 | 0.75 | rl only |
| tpil4 lean_tpil4_050 | ⊢ ((P ∧ ⊥) → ⊥) ∧ (⊥ → (P ∧ ⊥)) | 7 | 0.0 | 0.875 | 0.656 | rl only |
| logic_and_proof lean_logic_and_proof_058 | ¬(P ∧ Q), P ⊢ ¬P ∨ ¬Q | 7 | 0.0 | 0.438 | 0.094 | rl only |
| lean4game-logic lean_lean4game-logic_005 | ⊢ ¬¬(P ∨ ¬P) | 8 | 0.0 | 0.5 | 0.156 | rl only |
| lean4game-logic lean_lean4game-logic_016 | ¬(P → Q) ⊢ ¬¬P | 8 | 0.0 | 0.25 | 0.906 | rl only |
| lean4game-logic lean_lean4game-logic_017 | P → Q, R → Q, P ∨ R ⊢ Q | 8 | 0.031 | 0.219 | 0.531 | both |
| lean4game-logic lean_lean4game-logic_018 | P → Q, P ∨ R ⊢ Q ∨ R | 8 | 0.0 | 0.656 | 0.719 | rl only |
| fm2024 lean_fm2024_024 | ⊢ (((P → Q) → Q) → Q) → (P → Q) | 8 | 0.0 | 0.094 | 0.0 | rl only |
| tpil4 lean_tpil4_056 | P ⊢ ((P ∧ Q) → Q) ∧ (Q → (P ∧ Q)) | 8 | 0.0 | 0.969 | 0.938 | rl only |
| tpil4 lean_tpil4_057 | P ⊢ (P ∨ (Q ∨ R)) ∧ ((Q ∨ (P ∨ R)) ∧ (Q ∨ (R ∨ P))) | 8 | 0.0 | 0.0 | 0.0 | neither |
| m40001 lean_m40001_075 | ¬P ⊢ (P → ⊥) ∧ (⊥ → P) | 8 | 0.0 | 0.812 | 0.75 | rl only |
| lean4game-logic lean_lean4game-logic_013 | ⊢ P → ((Q → P) ∧ (¬Q → P)) | 9 | 0.0 | 0.312 | 0.75 | rl only |
| lean4game-logic lean_lean4game-logic_019 | P → Q ⊢ (P ∨ R) → (R ∨ Q) | 9 | 0.0 | 0.781 | 0.656 | rl only |
| tpil4 lean_tpil4_055 | P, Q, R ⊢ P ∧ (((P ∧ Q) ∧ R) ∧ (Q ∧ (R ∧ P))) | 9 | 0.0 | 0.5 | 0.406 | rl only |
| love lean_love_067 | ⊢ (P → Q) → ((P → R) → (P → (Q → R))) | 9 | 0.0 | 0.688 | 0.938 | rl only |
| love lean_love_072 | P ∧ ((Q ∧ R) ∧ S) ⊢ Q ∧ (P ∧ (R ∧ Q)) | 9 | 0.0 | 0.0 | 0.0 | neither |
| m40001 lean_m40001_082 | ⊢ (((⊥ → ⊥) → ⊥) → ⊥) ∧ (⊥ → ((⊥ → ⊥) → ⊥)) | 9 | 0.0 | 0.031 | 0.0 | rl only |
| tpil4 lean_tpil4_049 | ⊢ ((P ∨ ⊥) → P) ∧ (P → (P ∨ ⊥)) | 10 | 0.0 | 0.969 | 0.906 | rl only |
| m40001 lean_m40001_074 | P → Q, ¬P → ¬R ⊢ R → Q | 10 | 0.0 | 0.031 | 0.031 | rl only |
| fm2024 lean_fm2024_023 | ⊢ ((P → Q) → Q) → ((P → R) → ((R → Q) → Q)) | 11 | 0.0 | 0.0 | 0.0 | neither |
| love lean_love_066 | ⊢ ((P → Q) → (R → S)) → (R → (Q → S)) | 11 | 0.0 | 0.812 | 0.906 | rl only |
| love lean_love_068 | ⊢ ((((P → Q) → P) → P) → Q) → Q | 12 | 0.0 | 0.0 | 0.0 | neither |
| fm2024 lean_fm2024_022 | ⊢ ((P → Q) → R) → (((Q → R) → P) → (((R → P) → Q) → P)) | 13 | 0.0 | 0.0 | 0.0 | neither |
| lean4game-logic lean_lean4game-logic_011 | (P ∨ (Q ∨ R)) → ¬(P ∧ (Q ∧ R)) ⊢ ((P ∨ Q) ∨ R) → ¬((P ∧ Q) ∧ R) | 14 | 0.0 | 0.0 | 0.0 | neither |
| fm2024 lean_fm2024_035 | ⊢ (((P ∧ Q) ∧ R) → (P ∧ (Q ∧ R))) ∧ ((P ∧ (Q ∧ R)) → ((P ∧ Q) ∧ R)) | 17 | 0.0 | 0.0 | 0.0 | neither |
| lean4game-logic lean_lean4game-logic_000 | (P ∧ Q) ∨ (¬P ∧ ¬Q) ⊢ (P → Q) ∧ (Q → P) | 20 | 0.0 | 0.188 | 0.25 | rl only |
| tpil4 lean_tpil4_046 | ⊢ (((P ∨ Q) → R) → ((P → R) ∧ (Q → R))) ∧ (((P → R) ∧ (Q → R)) → ((P ∨ Q) → R)) | 23 | 0.0 | 0.0 | 0.0 | neither |
| logic_and_proof lean_logic_and_proof_063 | ⊢ (((P ∧ ¬Q) ∨ Q) → (P ∨ Q)) ∧ ((P ∨ Q) → ((P ∧ ¬Q) ∨ Q)) | 25 | 0.0 | 0.0 | 0.0 | neither |
| fm2024 lean_fm2024_039 | ⊢ (((P ∨ Q) ∨ R) → (P ∨ (Q ∨ R))) ∧ ((P ∨ (Q ∨ R)) → ((P ∨ Q) ∨ R)) | 27 | 0.0 | 0.0 | 0.0 | neither |
| m40001 lean_m40001_079 | ⊢ ((P ∨ (Q ∨ R)) → (Q ∨ (P ∨ R))) ∧ ((Q ∨ (P ∨ R)) → (P ∨ (Q ∨ R))) | 27 | 0.0 | 0.0 | 0.0 | neither |

## derived

| source · no. | sequent | ref | pt | rl s0 | rl s1 | category |
| --- | --- | --- | --- | --- | --- | --- |
| curry | ⊢ (P ∧ Q) → (((R → S) ∧ (S → R)) ∨ Q) | 4 | 0.781 | 0.969 | 0.938 | both |
| curry | ⊢ ¬¬¬¬P → P | 4 | 0.0 | 0.094 | 0.938 | rl only |
| forward_control | (P → Q) ∧ (Q → P) ⊢ (Q → P) ∧ (P → Q) | 4 | 0.031 | 0.531 | 0.156 | both |
| forward_control | P ∧ (P → (P ∧ Q)) ⊢ Q | 5 | 0.375 | 0.906 | 0.625 | both |
| permutation | ¬Q ⊢ ¬(P ∧ Q) | 5 | 0.938 | 0.938 | 1.0 | both |
| substitution | ⊢ (((P ∨ Q) → Q) ∧ (Q → (P ∨ Q))) → ((Q → (P ∨ Q)) ∧ ((P ∨ Q) → Q)) | 5 | 0.0 | 0.531 | 0.875 | rl only |
| forward_control | P ∧ Q, (Q ∧ P) → R ⊢ R | 6 | 0.0 | 0.0 | 0.094 | neither |
| forward_control | ¬(P ∨ Q), R, R → P ⊢ ⊥ | 6 | 0.75 | 0.75 | 0.5 | both |
| forward_control | P ∧ Q, (Q → R) ∧ (R → Q) ⊢ R ∨ S | 6 | 0.0 | 0.0 | 0.281 | neither |
| near_miss | (P ∨ Q) → R, R → (P ∨ Q) ⊢ Q → R | 6 | 0.031 | 0.469 | 0.656 | both |
| demorgan | ⊢ ¬¬(P → P) ∨ P | 6 | 0.031 | 0.906 | 0.969 | both |
| no_dn | P → (P ∧ ¬P) ⊢ ¬P | 6 | 0.0 | 0.844 | 0.938 | rl only |
| curry | ⊢ ((P ∧ Q) ∨ (P ∧ R)) → P | 7 | 0.0 | 0.312 | 0.625 | rl only |
| curry | (P → Q) → R ⊢ Q → R | 7 | 0.0 | 0.969 | 0.938 | rl only |
| curry | ⊢ ((P → (P → Q)) ∧ ((P → Q) → P)) → (P → Q) | 7 | 0.0 | 0.156 | 0.688 | rl only |
| curry | ⊢ ¬(P ∧ Q) → (((¬(P ∧ Q) → R) ∧ (R → ¬(P ∧ Q))) → (P ∨ R)) | 7 | 0.0 | 0.125 | 0.594 | rl only |
| forward_control | (P ∨ Q) → ((Q → R) ∧ (R → Q)), Q ⊢ Q ∧ R | 7 | 0.0 | 0.062 | 0.0 | rl only |
| forward_control | P → ((Q ∧ P) → R), P ∧ Q ⊢ R | 7 | 0.0 | 0.0 | 0.031 | neither |
| forward_control | P ∧ Q, Q → (R ∧ S) ⊢ P ∧ S | 7 | 0.0 | 0.5 | 0.719 | rl only |
| wrappers | ¬¬¬(P ∨ Q) ⊢ Q → P | 7 | 0.0 | 0.375 | 0.938 | rl only |
| wrappers | ⊢ ((P ∧ ¬P) → (Q ∧ ¬Q)) ∨ R | 7 | 0.0 | 0.844 | 0.562 | rl only |
| wrappers | ⊢ (¬P → (P → Q)) ∨ R | 7 | 0.0 | 0.969 | 1.0 | rl only |
| permutation | (R ∧ Q) ∨ P ⊢ P ∨ Q | 7 | 0.406 | 0.781 | 0.844 | both |
| substitution | ⊢ ¬(P ∨ Q) → ((Q ∧ (P ∨ Q)) → R) | 7 | 0.0 | 0.969 | 0.938 | rl only |
| demorgan | ¬(¬P ∧ ¬Q), ¬Q ⊢ P | 7 | 0.0 | 0.219 | 0.25 | rl only |
| line_budget | ⊢ (P → (P ∧ P)) ∧ ((P ∧ P) → P) | 7 | 0.031 | 0.844 | 1.0 | both |
| line_budget | (P → Q) ∧ (Q → P), ¬P ⊢ ¬Q | 7 | 0.0 | 0.594 | 0.562 | rl only |
| no_dn | ¬(P ∨ Q) ⊢ R → ¬P | 7 | 0.656 | 0.938 | 1.0 | both |
| no_dn | (P → Q) ∧ (Q → P), ¬Q ⊢ ¬P | 7 | 0.0 | 0.125 | 0.469 | rl only |
| curry | ⊢ ¬(P ∨ Q) → (R → ¬P) | 8 | 0.0 | 0.875 | 0.938 | rl only |
| forward_control | (P ∨ Q) → R, (S → R) ∧ (R → S), P ⊢ P ∧ S | 8 | 0.0 | 0.156 | 0.0 | rl only |
| near_miss | P, (Q ∨ R) → ¬P, ¬P → (Q ∨ R) ⊢ ¬Q | 8 | 0.0 | 0.094 | 0.0 | rl only |
| near_miss | ¬¬P → ¬Q, ¬Q → ¬¬P ⊢ P → ¬Q | 8 | 0.0 | 0.531 | 0.875 | rl only |
| permutation | ⊢ (P → R) → ((R → Q) → (P → Q)) | 8 | 0.0 | 0.0 | 0.031 | neither |
| permutation | ⊢ ((Q → P) ∧ ¬P) → ¬Q | 8 | 0.0 | 0.562 | 0.688 | rl only |
| distractor_scaling | P ∨ (¬P ∧ Q), P → (P ∧ ¬P) ⊢ P ∨ Q | 8 | 0.0 | 0.188 | 0.531 | rl only |
| distractor_scaling | P, ¬Q → ¬P, P ∧ Q ⊢ Q | 8 | 0.0 | 0.875 | 0.781 | rl only |
| no_dn | (P ∧ Q) → ¬P ⊢ P → ¬Q | 8 | 0.0 | 1.0 | 0.969 | rl only |
| force_ore | P → (Q ∨ ¬P), P ⊢ Q | 8 | 0.0 | 0.969 | 0.875 | rl only |
| forward_control | P ∧ Q, (P → R) ∧ (Q → S) ⊢ R ∧ S | 9 | 0.0 | 0.0 | 0.0 | neither |
| wrappers | ⊢ ((P → (Q → R)) → (Q → (P → R))) ∨ S | 9 | 0.0 | 0.781 | 0.781 | rl only |
| wrappers | ⊢ ((P → Q) → ((R → P) → (R → Q))) ∨ S | 9 | 0.0 | 0.0 | 0.0 | neither |
| near_miss | ¬P → Q, Q → P, Q → ¬P ⊢ P | 9 | 0.0 | 0.0 | 0.062 | neither |
| near_miss | P ∨ Q, Q → R, R → Q ⊢ P ∨ R | 9 | 0.0 | 0.5 | 0.219 | rl only |
| distractor_scaling | P ∨ (¬P ∧ Q), P → (P ∧ ¬P), (P ∧ Q) ∨ R ⊢ P ∨ Q | 9 | 0.0 | 0.156 | 0.594 | rl only |
| distractor_scaling | P, ¬Q → ¬P, P ∧ Q, (P ∧ Q) ∨ R ⊢ Q | 9 | 0.0 | 0.0 | 0.0 | neither |
| demorgan | P ∧ Q, ¬(Q ∨ R) ∨ S ⊢ S | 9 | 0.0 | 0.375 | 0.125 | rl only |
| line_budget | (P → ¬P) ∧ (¬P → P) ⊢ ⊥ | 9 | 0.0 | 0.406 | 0.031 | rl only |
| line_budget | P → Q ⊢ ((P ∧ Q) → P) ∧ (P → (P ∧ Q)) | 9 | 0.0 | 0.875 | 0.875 | rl only |
| line_budget | ⊢ (¬¬P → P) ∧ (P → ¬¬P) | 9 | 0.0 | 0.906 | 1.0 | rl only |
| force_ore | ⊢ (P → (P ∨ P)) ∧ ((P ∨ P) → P) | 9 | 0.0 | 0.938 | 1.0 | rl only |
| curry | P → (Q → R) ⊢ (S → Q) → (P → (S → R)) | 10 | 0.0 | 0.125 | 0.0 | rl only |
| negated_conjunction | ⊢ ¬(((P → (P ∨ Q)) ∧ ((P ∨ Q) → P)) ∧ ¬(Q → P)) | 10 | 0.0 | 0.062 | 0.031 | rl only |
| wrappers | ⊢ ((P → Q) → (¬Q → ¬(P ∧ R))) ∨ S | 10 | 0.0 | 0.094 | 0.125 | rl only |
| substitution | (Q → P) → (Q → R), Q → ¬R ⊢ (Q → P) → ¬Q | 10 | 0.0 | 0.531 | 0.594 | rl only |
| substitution | (Q → P) ∧ (¬Q → ¬(Q → P)) ⊢ (Q ∧ (Q → P)) ∨ ¬(Q → P) | 10 | 0.0 | 0.0 | 0.0 | neither |
| substitution | ⊢ (¬(P ∧ Q) → ¬Q) → ((¬(P ∧ Q) → Q) → (P ∧ Q)) | 10 | 0.0 | 0.594 | 0.094 | rl only |
| demorgan | ⊢ ¬((P → Q) ∧ (Q → P)) ∨ (P → Q) | 10 | 0.0 | 0.0 | 0.0 | neither |
| demorgan | ⊢ ¬((P → Q) ∧ (Q → P)) ∨ (Q → P) | 10 | 0.0 | 0.0 | 0.0 | neither |
| line_budget | P ∧ (Q ∨ R), P → ¬R ⊢ Q | 10 | 0.0 | 0.781 | 0.5 | rl only |
| no_dn | P ∨ Q ⊢ ¬(¬P ∧ ¬Q) | 10 | 0.0 | 0.406 | 0.875 | rl only |
| no_dn | ¬(P ∧ Q), ¬(P ∧ ¬Q) ⊢ ¬P | 10 | 0.0 | 0.0 | 0.0 | neither |
| no_dn | P ∨ ¬Q, ¬P ∨ ¬Q ⊢ ¬Q | 10 | 0.0 | 0.5 | 0.0 | rl only |
| curry | P ∨ Q ⊢ (P → Q) → ((Q → P) → (P ∧ Q)) | 11 | 0.0 | 0.0 | 0.656 | neither |
| curry | ⊢ ((P ∧ Q) → (R ∧ S)) → (((P ∧ Q) → R) ∧ ((P ∧ Q) → S)) | 11 | 0.0 | 0.562 | 0.438 | rl only |
| negated_conjunction | ⊢ ¬((P → ¬Q) ∧ ¬(Q → ¬P)) | 11 | 0.0 | 0.0 | 0.0 | neither |
| wrappers | ((P ∧ Q) ∨ (Q ∧ R)) ∨ ((P ∧ Q) ∨ (Q ∧ R)), Q → S ⊢ S | 11 | 0.0 | 0.031 | 0.0 | rl only |
| near_miss | P → (Q → R), (Q → R) → P ⊢ ¬R → ¬(P ∧ Q) | 11 | 0.0 | 0.0 | 0.0 | neither |
| permutation | ⊢ ¬((Q ∨ P) ∧ (¬Q ∧ ¬P)) | 11 | 0.0 | 0.375 | 0.0 | rl only |
| permutation | ⊢ (((Q → P) ∧ (P → Q)) → ((P → Q) ∧ (Q → P))) ∧ (((P → Q) ∧ (Q → P)) → ((Q → P) ∧ (P → Q))) | 11 | 0.0 | 0.0 | 0.0 | neither |
| permutation | P, Q ⊢ (¬Q → ¬P) ∧ (¬P → ¬Q) | 11 | 0.0 | 0.531 | 0.812 | rl only |
| distractor_scaling | P ∨ (¬P ∧ Q), P → (P ∧ ¬P), (P ∧ Q) ∨ R, (Q → R) ∧ (R → Q), ¬P → Q ⊢ P ∨ Q | 11 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | P, ¬Q → ¬P, P ∧ Q, (P ∧ Q) ∨ R, (P → Q) ∧ (Q → P), (Q → R) ∧ (R → Q) ⊢ Q | 11 | 0.0 | 0.0 | 0.0 | neither |
| line_budget | ⊢ P → ((¬Q → ¬R) → (R → Q)) | 11 | 0.0 | 0.344 | 0.375 | rl only |
| no_dn | ¬(P → Q), ¬(Q → R) ⊢ ¬S | 11 | 0.0 | 0.094 | 0.469 | rl only |
| force_ore | ⊢ ((P → Q) ∧ (R → Q)) → ((P ∨ R) → Q) | 11 | 0.0 | 0.0 | 0.625 | neither |
| force_ore | P ∧ (Q ∨ R), P → ¬R ⊢ Q ∨ S | 11 | 0.0 | 0.562 | 0.562 | rl only |
| negated_conjunction | ¬P → (Q ∨ P) ⊢ ¬(¬P ∧ ¬Q) | 12 | 0.0 | 0.188 | 0.625 | rl only |
| wrappers | (P ∧ ¬Q) → ¬R ⊢ ((P ∧ R) → Q) ∨ S | 12 | 0.0 | 0.0 | 0.031 | neither |
| near_miss | (P ∧ Q) → (R ∨ S), Q ∧ P, (R ∨ S) → (P ∧ Q) ⊢ S ∨ R | 12 | 0.0 | 0.0 | 0.0 | neither |
| substitution | ¬Q ∨ Q, ¬Q ∨ R, ¬R ⊢ ¬Q | 12 | 0.031 | 0.938 | 0.625 | both |
| demorgan | ¬(¬P ∧ ¬¬Q), Q ∨ R, ¬P ⊢ R | 12 | 0.0 | 0.062 | 0.312 | rl only |
| demorgan | ⊢ ¬(P ∧ Q) ∨ ((P → Q) → P) | 12 | 0.0 | 0.0 | 0.0 | neither |
| demorgan | ⊢ ¬((P → Q) ∧ (Q → P)) ∨ ((Q → P) ∧ (P → Q)) | 12 | 0.0 | 0.0 | 0.0 | neither |
| no_dn | ¬P → Q, R → ¬Q, ¬R → P ⊢ ¬¬P | 12 | 0.0 | 0.062 | 0.0 | rl only |
| force_ore | P ∨ ¬Q, Q ∨ R, ¬P ⊢ R | 12 | 0.0 | 0.219 | 0.0 | rl only |
| negated_conjunction | ¬P → Q ⊢ ¬((P → ¬Q) ∧ ¬((¬P → Q) ∧ (Q → ¬P))) | 13 | 0.0 | 0.0 | 0.0 | neither |
| near_miss | (P → Q) → R, S → (P → Q), ¬S → R, R → (P → Q) ⊢ ¬¬R | 13 | 0.0 | 0.0 | 0.0 | neither |
| substitution | ⊢ ¬((P ∧ Q) → Q) → ((P ∧ Q) ∧ ¬Q) | 13 | 0.0 | 0.969 | 0.688 | rl only |
| demorgan | ⊢ ¬(¬P → P) ∨ ¬¬P | 13 | 0.0 | 0.0 | 0.0 | neither |
| force_ore | (P ∨ Q) ∨ R, (R → Q) ∧ (Q → R) ⊢ R ∨ P | 13 | 0.0 | 0.438 | 0.0 | rl only |
| force_ore | P ∨ Q, (P → Q) ∧ (Q → P), ¬(P ∧ Q) ⊢ R | 13 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q, Q → P, (Q → ((Q → P) ∧ (P → Q))) ∧ (((Q → P) ∧ (P → Q)) → Q), (P ∧ Q) → R ⊢ P ∧ (P → R) | 14 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | ⊢ ¬(((P ∧ Q) → R) ∧ ¬((P ∧ ¬R) → ¬Q)) | 14 | 0.0 | 0.0 | 0.0 | neither |
| permutation | ¬(¬Q ∧ ¬P) ⊢ Q ∨ P | 14 | 0.0 | 0.0 | 0.0 | neither |
| demorgan | ¬P ∨ ¬Q, (¬(P ∧ Q) → R) ∧ (R → ¬(P ∧ Q)) ⊢ P ∨ R | 14 | 0.0 | 0.0 | 0.0 | neither |
| force_ore | P ∨ Q, Q ∨ R, ¬Q ⊢ P ∧ R | 14 | 0.0 | 0.094 | 0.031 | rl only |
| conjoined | P → Q, Q → R, R → S ⊢ (P → S) ∧ ((Q → R) → (P → R)) | 15 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q ⊢ ((P ∨ (R ∧ Q)) → Q) ∧ ((R → P) → (R → Q)) | 15 | 0.0 | 0.031 | 0.0 | rl only |
| permutation | (R → Q) ∧ (Q → R), (P → R) ∧ (R → P) ⊢ (P → Q) ∧ (Q → P) | 15 | 0.0 | 0.0 | 0.0 | neither |
| substitution | ¬Q ∨ Q ⊢ ¬Q ∨ (¬¬Q ∧ Q) | 15 | 0.0 | 0.031 | 0.0 | rl only |
| substitution | ⊢ ¬((P ∧ Q) ∧ Q) → (¬(P ∧ Q) ∨ ¬Q) | 15 | 0.0 | 0.0 | 0.094 | neither |
| line_budget | P → ¬P, Q → P, R → Q, ¬R → Q ⊢ S | 15 | 0.0 | 0.0 | 0.0 | neither |
| line_budget | ⊢ ((P ∨ Q) → (Q ∨ P)) ∧ ((Q ∨ P) → (P ∨ Q)) | 15 | 0.0 | 0.0 | 0.031 | neither |
| line_budget | (P → ¬Q) ∧ (¬Q → P) ⊢ ¬((P → Q) ∧ (Q → P)) | 15 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q, R ∧ ¬S, (¬S → (Q → P)) ∧ ((Q → P) → ¬S), (R → Q) → ¬P ⊢ ((P → Q) ∧ (Q → P)) ∧ ¬P | 16 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q, (Q ∨ R) → S, S → ¬P ⊢ ¬P ∧ ((P ∧ R) → (Q ∨ S)) | 16 | 0.0 | 0.094 | 0.0 | rl only |
| permutation | ((Q ∧ R) → (Q ∧ P)) ∧ ((Q ∧ P) → (Q ∧ R)) ⊢ Q → ((R → P) ∧ (P → R)) | 16 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → ¬((Q → ¬R) ∧ (¬R → Q))) ∧ (¬((Q → ¬R) ∧ (¬R → Q)) → P), ¬(P ∨ Q), P → (P ∧ ¬P) ⊢ R | 16 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | (P ∨ Q) ∨ R ⊢ ¬(((Q → R) ∧ (R → Q)) ∧ ¬(R ∨ P)) | 17 | 0.0 | 0.0 | 0.0 | neither |
| near_miss | (P ∨ Q) → (P ∨ R), (P ∨ R) → (P ∨ Q) ⊢ P ∨ (Q → R) | 17 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → ¬((Q → ¬R) ∧ (¬R → Q))) ∧ (¬((Q → ¬R) ∧ (¬R → Q)) → P), ¬(P ∨ Q), P → (P ∧ ¬P), (P ∧ Q) ∨ R ⊢ R | 17 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → Q) ∧ (Q → P), P → (P ∧ ¬P) ⊢ ((P ∧ R) → (Q ∧ R)) ∧ ((Q ∧ R) → (P ∧ R)) | 17 | 0.0 | 0.0 | 0.0 | neither |
| substitution | ⊢ (((P ∨ Q) → (Q → R)) → (((P ∨ Q) ∧ Q) → R)) ∧ ((((P ∨ Q) ∧ Q) → R) → ((P ∨ Q) → (Q → R))) | 17 | 0.0 | 0.0 | 0.0 | neither |
| force_ore | (P ∧ Q) ∨ ¬(R → ¬S), ¬Q ⊢ R ∧ S | 17 | 0.0 | 0.0 | 0.031 | neither |
| conjoined | P → Q, Q → ¬P ⊢ (¬P ∨ Q) ∧ ¬P | 18 | 0.0 | 0.625 | 0.344 | rl only |
| conjoined | P → Q, Q → ¬R, S → P, R → S ⊢ ¬R ∧ ¬(P ∧ ¬Q) | 18 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | ⊢ ¬((P → ¬Q) ∧ (¬(R ∧ P) ∧ ¬((Q ∨ R) → ¬P))) | 18 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → Q) ∧ (Q → P), P → (P ∧ ¬P), (P ∧ Q) ∨ R ⊢ ((P ∧ R) → (Q ∧ R)) ∧ ((Q ∧ R) → (P ∧ R)) | 18 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | ⊢ ¬(((¬P → Q) ∧ (Q → ¬P)) ∧ ¬¬((P → Q) ∧ (Q → P))) | 19 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | ⊢ ¬(((P → (¬Q ∨ R)) ∧ ((¬Q ∨ R) → P)) ∧ ((Q → R) ∧ ¬P)) | 19 | 0.0 | 0.0 | 0.0 | neither |
| near_miss | (P ∨ Q) ∨ (R ∨ S), Q → R, ¬S, R → Q ⊢ P ∨ R | 19 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → ¬((Q → ¬R) ∧ (¬R → Q))) ∧ (¬((Q → ¬R) ∧ (¬R → Q)) → P), ¬(P ∨ Q), P → (P ∧ ¬P), (P ∧ Q) ∨ R, (P → Q) ∧ (Q → P), P → R ⊢ R | 19 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q, Q → R, R → S, P → (Q → R) ⊢ (P → ¬(P → ¬S)) ∧ (P → (Q ∧ R)) | 20 | 0.0 | 0.0 | 0.0 | neither |
| distractor_scaling | (P → Q) ∧ (Q → P), P → (P ∧ ¬P), (P ∧ Q) ∨ R, P → R, Q → S ⊢ ((P ∧ R) → (Q ∧ R)) ∧ ((Q ∧ R) → (P ∧ R)) | 20 | 0.0 | 0.0 | 0.0 | neither |
| force_ore | ¬(P ∨ ¬Q) ∨ (¬Q → ¬R), Q → P, P → R ⊢ (Q → R) ∧ (R → Q) | 20 | 0.0 | 0.0 | 0.0 | neither |
| negated_conjunction | ⊢ ¬(((¬P ∨ Q) ∨ R) ∧ ((R → ¬S) ∧ ¬((P ∧ S) → Q))) | 22 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → Q, Q → R, R → S, P, R → (R → S) ⊢ ¬(P → ¬S) ∧ ((R ∨ P) → (Q ∨ S)) | 23 | 0.0 | 0.0 | 0.0 | neither |
| wrappers | ⊢ (((¬P → Q) → (P ∨ Q)) ∧ ((P ∨ Q) → (¬P → Q))) ∨ R | 23 | 0.0 | 0.0 | 0.0 | neither |
| no_dn | ⊢ (¬(P ∨ Q) → (¬P ∧ ¬Q)) ∧ ((¬P ∧ ¬Q) → ¬(P ∨ Q)) | 23 | 0.0 | 0.0 | 0.0 | neither |
| wrappers | ((P → Q) ∧ (Q → P)) ∧ ((P → Q) ∧ (Q → P)), (Q → ¬R) ∧ (¬R → Q) ⊢ ¬((P → R) ∧ (R → P)) | 24 | 0.0 | 0.0 | 0.0 | neither |
| conjoined | P → (Q → R), ¬S → (P ∨ R), P → Q, Q → (R ∧ P) ⊢ (S ∨ R) ∧ ((P → Q) ∧ (Q → P)) | 26 | 0.0 | 0.0 | 0.0 | neither |