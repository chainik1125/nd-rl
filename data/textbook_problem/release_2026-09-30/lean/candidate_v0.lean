-- candidate_v0: 104 problems, rendered by the lean_seq tokenizer pinned in code/textbook_problem/lean_tools/.
-- One theorem per problem; the ND reference proof is rendered as tactic text where one exists, otherwise `sorry`.
-- Checked with Lean 4.34.1 on 2026-09-30: no errors; the unsolved file elaborates with `sorry`.

-- textbook_30018c9c991412364923 | forallx-calgary 2 | \item $Q\eif(Q\eand\enot Q) \therefore \enot Q$
theorem textbook_30018c9c991412364923 ( P Q R S : Prop ) ( h1 : ( P → ( P ∧ ( ¬ P ) ) ) ) : ( ¬ P ) := by have n1 : ( P → ( P ∧ ( ¬ P ) ) ) := h1 ; have n2 : ( ¬ P ) := ( fun ( n3 : P ) => by have n4 : ( P ∧ ( ¬ P ) ) := n1 n3 ; have n5 : ( ¬ P ) := n4.2 ; have n6 : False := n5 n3 ; exact ( n6 : False ) ) ; exact n2

-- textbook_7c28a8f7423fde99f79c | forallx-calgary 4 | \item $K\eand L \therefore K\eiff L$
theorem textbook_7c28a8f7423fde99f79c ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : Q := n1.2 ; exact n4 ) ; have n5 : ( Q → P ) := ( fun ( n6 : Q ) => by have n7 : P := n1.1 ; exact n7 ) ; have n8 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n2 , n5 ⟩ ; exact n8

-- textbook_95041444f8cb17626d64 | forallx-calgary 5 | \item $(C\eand D)\eor E \therefore E\eor D$
theorem textbook_95041444f8cb17626d64 ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ R ) ) : ( R ∨ Q ) := by have n1 : ( ( P ∧ Q ) ∨ R ) := h1 ; have n2 : ( R ∨ Q ) := Or.elim n1 ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : Q := n3.2 ; have n5 : ( R ∨ Q ) := Or.inr n4 ; exact n5 ) ( fun ( n6 : R ) => by have n7 : ( R ∨ Q ) := Or.inl n6 ; exact n7 ) ; exact n2

-- textbook_96a56ad8e1132658e60b | forallx-calgary 6 | \item $A\eiff B, B\eiff C \therefore A\eiff C$
theorem textbook_96a56ad8e1132658e60b ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) ( h2 : ( ( Q → R ) ∧ ( R → Q ) ) ) : ( ( P → R ) ∧ ( R → P ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( ( Q → R ) ∧ ( R → Q ) ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : ( Q → R ) := n2.1 ; have n6 : ( P → Q ) := n1.1 ; have n7 : Q := n6 n4 ; have n8 : R := n5 n7 ; exact n8 ) ; have n9 : ( R → P ) := ( fun ( n10 : R ) => by have n11 : ( Q → P ) := n1.2 ; have n12 : ( R → Q ) := n2.2 ; have n13 : Q := n12 n10 ; have n14 : P := n11 n13 ; exact n14 ) ; have n15 : ( ( P → R ) ∧ ( R → P ) ) := ⟨ n3 , n9 ⟩ ; exact n15

-- textbook_17cf6bcfdfe010a1d9b2 | forallx-calgary 7 | \item $\enot F\eif G, F\eif H \therefore G\eor H$
theorem textbook_17cf6bcfdfe010a1d9b2 ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → Q ) ) ( h2 : ( P → R ) ) : ( Q ∨ R ) := by have n1 : ( ( ¬ P ) → Q ) := h1 ; have n2 : ( P → R ) := h2 ; have n3 : ( ¬ ( ¬ ( Q ∨ R ) ) ) := ( fun ( n4 : ( ¬ ( Q ∨ R ) ) ) => by have n5 : ( ¬ P ) := ( fun ( n6 : P ) => by have n7 : R := n2 n6 ; have n8 : ( Q ∨ R ) := Or.inr n7 ; have n9 : False := n4 n8 ; exact ( n9 : False ) ) ; have n10 : Q := n1 n5 ; have n11 : ( Q ∨ R ) := Or.inl n10 ; have n12 : False := n4 n11 ; exact ( n12 : False ) ) ; have n13 : ( Q ∨ R ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n13

-- textbook_80573e8091085933fe07 | forallx-calgary 8 | \item $(Z\eand K) \eor (K\eand M), K \eif D \therefore D$
theorem textbook_80573e8091085933fe07 ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( Q ∧ R ) ) ) ( h2 : ( Q → S ) ) : S := by have n1 : ( ( P ∧ Q ) ∨ ( Q ∧ R ) ) := h1 ; have n2 : ( Q → S ) := h2 ; have n3 : Q := Or.elim n1 ( fun ( n4 : ( P ∧ Q ) ) => by have n5 : Q := n4.2 ; exact n5 ) ( fun ( n6 : ( Q ∧ R ) ) => by have n7 : Q := n6.1 ; exact n7 ) ; have n8 : S := n2 n3 ; exact n8

-- textbook_12d1e9ba77154cecdaa4 | forallx-calgary 9 | \item $P \eand (Q\eor R), P\eif \enot R \therefore Q\eor E$
theorem textbook_12d1e9ba77154cecdaa4 ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∨ R ) ) ) ( h2 : ( P → ( ¬ R ) ) ) : ( Q ∨ S ) := by have n1 : ( P ∧ ( Q ∨ R ) ) := h1 ; have n2 : ( P → ( ¬ R ) ) := h2 ; have n3 : ( Q ∨ R ) := n1.2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : P := n1.1 ; have n8 : ( ¬ R ) := n2 n7 ; have n9 : False := n8 n6 ; have n10 : Q := n9.elim ; exact n10 ) ; have n11 : ( Q ∨ S ) := Or.inl n4 ; exact n11

-- textbook_093d53538c2072386890 | forallx-calgary 3 | \item $[(A \eif C) \eand (B \eif C)] \eif [(A \lor B) \eif C]$
theorem textbook_093d53538c2072386890 ( P Q R S : Prop ) : ( ( ( P → Q ) ∧ ( R → Q ) ) → ( ( P ∨ R ) → Q ) ) := by have n1 : ( ( ( P → Q ) ∧ ( R → Q ) ) → ( ( P ∨ R ) → Q ) ) := ( fun ( n2 : ( ( P → Q ) ∧ ( R → Q ) ) ) => by have n3 : ( ( P ∨ R ) → Q ) := ( fun ( n4 : ( P ∨ R ) ) => by have n5 : Q := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( P → Q ) := n2.1 ; have n8 : Q := n7 n6 ; exact n8 ) ( fun ( n9 : R ) => by have n10 : ( R → Q ) := n2.2 ; have n11 : Q := n10 n9 ; exact n11 ) ; exact n5 ) ; exact n3 ) ; exact n1

-- textbook_bdcc2efb2f66ee4a2721 | forallx-calgary 4 | \item $\enot(A \eif B) \eif (A \eand \enot B)$
theorem textbook_bdcc2efb2f66ee4a2721 ( P Q R S : Prop ) : ( ( ¬ ( P → Q ) ) → ( P ∧ ( ¬ Q ) ) ) := by have n1 : ( ( ¬ ( P → Q ) ) → ( P ∧ ( ¬ Q ) ) ) := ( fun ( n2 : ( ¬ ( P → Q ) ) ) => by have n3 : ( ¬ ( ¬ ( P ∧ ( ¬ Q ) ) ) ) := ( fun ( n4 : ( ¬ ( P ∧ ( ¬ Q ) ) ) ) => by have n5 : ( P → Q ) := ( fun ( n6 : P ) => by have n7 : ( ¬ ( ¬ Q ) ) := ( fun ( n8 : ( ¬ Q ) ) => by have n9 : ( P ∧ ( ¬ Q ) ) := ⟨ n6 , n8 ⟩ ; have n10 : False := n4 n9 ; exact ( n10 : False ) ) ; have n11 : Q := Classical.byContradiction ( fun hh => n7 hh ) ; exact n11 ) ; have n12 : False := n2 n5 ; exact ( n12 : False ) ) ; have n13 : ( P ∧ ( ¬ Q ) ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n13 ) ; exact n1

-- textbook_0557545ab024a85bd5ed | forallx-calgary 3 | \item $A \eif B \therefore \enot A \eor B$
theorem textbook_0557545ab024a85bd5ed ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( ¬ P ) ∨ Q ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ¬ ( ¬ ( ( ¬ P ) ∨ Q ) ) ) := ( fun ( n3 : ( ¬ ( ( ¬ P ) ∨ Q ) ) ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : Q := n1 n5 ; have n7 : ( ( ¬ P ) ∨ Q ) := Or.inr n6 ; have n8 : False := n3 n7 ; exact ( n8 : False ) ) ; have n9 : ( ( ¬ P ) ∨ Q ) := Or.inl n4 ; have n10 : False := n3 n9 ; exact ( n10 : False ) ) ; have n11 : ( ( ¬ P ) ∨ Q ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n11

-- textbook_c675c4cea7507f6c903c | forallx-calgary 4 | \item $\enot(A \eand B) \eif (\enot A \eor \enot B)$
theorem textbook_c675c4cea7507f6c903c ( P Q R S : Prop ) : ( ( ¬ ( P ∧ Q ) ) → ( ( ¬ P ) ∨ ( ¬ Q ) ) ) := by have n1 : ( ( ¬ ( P ∧ Q ) ) → ( ( ¬ P ) ∨ ( ¬ Q ) ) ) := ( fun ( n2 : ( ¬ ( P ∧ Q ) ) ) => by have n3 : ( ¬ ( ¬ ( ( ¬ P ) ∨ ( ¬ Q ) ) ) ) := ( fun ( n4 : ( ¬ ( ( ¬ P ) ∨ ( ¬ Q ) ) ) ) => by have n5 : ( ¬ P ) := ( fun ( n6 : P ) => by have n7 : ( ¬ Q ) := ( fun ( n8 : Q ) => by have n9 : ( P ∧ Q ) := ⟨ n6 , n8 ⟩ ; have n10 : False := n2 n9 ; exact ( n10 : False ) ) ; have n11 : ( ( ¬ P ) ∨ ( ¬ Q ) ) := Or.inr n7 ; have n12 : False := n4 n11 ; exact ( n12 : False ) ) ; have n13 : ( ( ¬ P ) ∨ ( ¬ Q ) ) := Or.inl n5 ; have n14 : False := n4 n13 ; exact ( n14 : False ) ) ; have n15 : ( ( ¬ P ) ∨ ( ¬ Q ) ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n15 ) ; exact n1

-- textbook_26c5501ec4eb97272dc1 | forallx-calgary 1 | \item $E\eor F$, $F\eor G$, $\enot F \therefore E \eand G$
theorem textbook_26c5501ec4eb97272dc1 ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) ( h2 : ( Q ∨ R ) ) ( h3 : ( ¬ Q ) ) : ( P ∧ R ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( Q ∨ R ) := h2 ; have n3 : ( ¬ Q ) := h3 ; have n4 : P := Or.elim n1 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : Q ) => by have n7 : False := n3 n6 ; have n8 : P := n7.elim ; exact n8 ) ; have n9 : R := Or.elim n2 ( fun ( n10 : Q ) => by have n11 : False := n3 n10 ; have n12 : R := n11.elim ; exact n12 ) ( fun ( n13 : R ) => by exact n13 ) ; have n14 : ( P ∧ R ) := ⟨ n4 , n9 ⟩ ; exact n14

-- textbook_545881ef2f5176d94567 | forallx-calgary 2 | \item $M\eor(N\eif M) \therefore \enot M \eif \enot N$
theorem textbook_545881ef2f5176d94567 ( P Q R S : Prop ) ( h1 : ( P ∨ ( Q → P ) ) ) : ( ( ¬ P ) → ( ¬ Q ) ) := by have n1 : ( P ∨ ( Q → P ) ) := h1 ; have n2 : ( ( ¬ P ) → ( ¬ Q ) ) := ( fun ( n3 : ( ¬ P ) ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : P := Or.elim n1 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( Q → P ) ) => by have n9 : P := n8 n5 ; exact n9 ) ; have n10 : False := n3 n6 ; exact ( n10 : False ) ) ; exact n4 ) ; exact n2

-- textbook_5b38b6ab0a04f4c4bb28 | forallx-calgary 3 | \item $(M \eor N) \eand (O \eor P)$, $N \eif P$, $\enot P \therefore M\eand O$
theorem textbook_5b38b6ab0a04f4c4bb28 ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∧ ( R ∨ S ) ) ) ( h2 : ( Q → S ) ) ( h3 : ( ¬ S ) ) : ( P ∧ R ) := by have n1 : ( ( P ∨ Q ) ∧ ( R ∨ S ) ) := h1 ; have n2 : ( Q → S ) := h2 ; have n3 : ( ¬ S ) := h3 ; have n4 : ( P ∨ Q ) := n1.1 ; have n5 : P := Or.elim n4 ( fun ( n6 : P ) => by exact n6 ) ( fun ( n7 : Q ) => by have n8 : S := n2 n7 ; have n9 : False := n3 n8 ; have n10 : P := n9.elim ; exact n10 ) ; have n11 : ( R ∨ S ) := n1.2 ; have n12 : R := Or.elim n11 ( fun ( n13 : R ) => by exact n13 ) ( fun ( n14 : S ) => by have n15 : False := n3 n14 ; have n16 : R := n15.elim ; exact n16 ) ; have n17 : ( P ∧ R ) := ⟨ n5 , n12 ⟩ ; exact n17

-- textbook_9f8530f273dc8191d06c | forallx-calgary 3 | \item $J \eiff [J\eor (L\eand\enot L)]$
theorem textbook_9f8530f273dc8191d06c ( P Q R S : Prop ) : ( ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) ∧ ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) ) := by have n1 : ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) := ( fun ( n2 : P ) => by have n3 : ( P ∨ ( Q ∧ ( ¬ Q ) ) ) := Or.inl n2 ; exact n3 ) ; have n4 : ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) := ( fun ( n5 : ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) => by have n6 : P := Or.elim n5 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( Q ∧ ( ¬ Q ) ) ) => by have n9 : Q := n8.1 ; have n10 : ( ¬ Q ) := n8.2 ; have n11 : False := n10 n9 ; have n12 : P := n11.elim ; exact n12 ) ; exact n6 ) ; have n13 : ( ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) ∧ ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) ) := ⟨ n1 , n4 ⟩ ; exact n13

-- textbook_d0b1395237936017f5f3 | forallx-calgary 2 | \item $M \eand (\enot N \eif \enot M) \proves (N \eand M) \eor \enot M$
theorem textbook_d0b1395237936017f5f3 ( P Q R S : Prop ) ( h1 : ( P ∧ ( ( ¬ Q ) → ( ¬ P ) ) ) ) : ( ( Q ∧ P ) ∨ ( ¬ P ) ) := by have n1 : ( P ∧ ( ( ¬ Q ) → ( ¬ P ) ) ) := h1 ; have n2 : P := n1.1 ; have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : ( ( ¬ Q ) → ( ¬ P ) ) := n1.2 ; have n6 : ( ¬ P ) := n5 n4 ; have n7 : False := n6 n2 ; exact ( n7 : False ) ) ; have n8 : Q := Classical.byContradiction ( fun hh => n3 hh ) ; have n9 : ( Q ∧ P ) := ⟨ n8 , n2 ⟩ ; have n10 : ( ( Q ∧ P ) ∨ ( ¬ P ) ) := Or.inl n9 ; exact n10

-- textbook_a276d846d75000fe58ae | forallx-calgary 4 | \item $(W \eor X) \eor (Y \eor Z), X\eif Y, \enot Z \proves W\eor Y$
theorem textbook_a276d846d75000fe58ae ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∨ ( R ∨ S ) ) ) ( h2 : ( Q → R ) ) ( h3 : ( ¬ S ) ) : ( P ∨ R ) := by have n1 : ( ( P ∨ Q ) ∨ ( R ∨ S ) ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( ¬ S ) := h3 ; have n4 : ( P ∨ R ) := Or.elim n1 ( fun ( n5 : ( P ∨ Q ) ) => by have n6 : ( P ∨ R ) := Or.elim n5 ( fun ( n7 : P ) => by have n8 : ( P ∨ R ) := Or.inl n7 ; exact n8 ) ( fun ( n9 : Q ) => by have n10 : R := n2 n9 ; have n11 : ( P ∨ R ) := Or.inr n10 ; exact n11 ) ; exact n6 ) ( fun ( n12 : ( R ∨ S ) ) => by have n13 : R := Or.elim n12 ( fun ( n14 : R ) => by exact n14 ) ( fun ( n15 : S ) => by have n16 : False := n3 n15 ; have n17 : R := n16.elim ; exact n17 ) ; have n18 : ( P ∨ R ) := Or.inr n13 ; exact n18 ) ; exact n4

-- textbook_0a3aa5c2d4abc9d9ba3d | forallx-calgary 1 | \item $R \eiff E$, $E \eiff R$
theorem textbook_0a3aa5c2d4abc9d9ba3d ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) : ( ( Q → P ) ∧ ( P → Q ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( Q → P ) := n1.2 ; have n3 : ( P → Q ) := n1.1 ; have n4 : ( ( Q → P ) ∧ ( P → Q ) ) := ⟨ n2 , n3 ⟩ ; exact n4

-- textbook_43216153b8cb762ae45f | forallx-calgary 2 | \item $G$, $\enot\enot\enot\enot G$
theorem textbook_43216153b8cb762ae45f ( P Q R S : Prop ) ( h1 : ( ¬ ( ¬ ( ¬ ( ¬ P ) ) ) ) ) : P := by have n1 : ( ¬ ( ¬ ( ¬ ( ¬ P ) ) ) ) := h1 ; have n2 : ( ¬ ( ¬ P ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; have n3 : P := Classical.byContradiction ( fun hh => n2 hh ) ; exact n3

-- textbook_46701b298f4b81ffd08a | forallx-calgary 4 | \item $U \eif I$, $\enot(U \eand \enot I)$
theorem textbook_46701b298f4b81ffd08a ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ¬ ( P ∧ ( ¬ Q ) ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ¬ ( P ∧ ( ¬ Q ) ) ) := ( fun ( n3 : ( P ∧ ( ¬ Q ) ) ) => by have n4 : P := n3.1 ; have n5 : Q := n1 n4 ; have n6 : ( ¬ Q ) := n3.2 ; have n7 : False := n6 n5 ; exact ( n7 : False ) ) ; exact n2

-- textbook_7e11198dd38f376da50d | forallx-calgary 6 | \item $\enot G \eiff H$, $\enot(G \eiff H)$
theorem textbook_7e11198dd38f376da50d ( P Q R S : Prop ) ( h1 : ( ( ( ¬ P ) → Q ) ∧ ( Q → ( ¬ P ) ) ) ) : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := by have n1 : ( ( ( ¬ P ) → Q ) ∧ ( Q → ( ¬ P ) ) ) := h1 ; have n2 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := ( fun ( n3 : ( ( P → Q ) ∧ ( Q → P ) ) ) => by have n4 : ( ( ¬ P ) → Q ) := n1.1 ; have n5 : ( Q → ( ¬ P ) ) := n1.2 ; have n6 : ( Q → P ) := n3.2 ; have n7 : ( P → Q ) := n3.1 ; have n8 : ( ¬ P ) := ( fun ( n9 : P ) => by have n10 : Q := n7 n9 ; have n11 : ( ¬ P ) := n5 n10 ; have n12 : False := n11 n9 ; exact ( n12 : False ) ) ; have n13 : Q := n4 n8 ; have n14 : P := n6 n13 ; have n15 : False := n8 n14 ; exact ( n15 : False ) ) ; exact n2

-- textbook_daf2d92294493b63e627 | forallx-calgary 4 | \item Show that the sentence $A \eiff \enot A$ is a contradiction.
theorem textbook_daf2d92294493b63e627 ( P Q R S : Prop ) ( h1 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) : False := by have n1 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) := h1 ; have n2 : ( P → ( ¬ P ) ) := n1.1 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( ¬ P ) := n2 n4 ; have n6 : False := n5 n4 ; exact ( n6 : False ) ) ; have n7 : ( ( ¬ P ) → P ) := n1.2 ; have n8 : P := n7 n3 ; have n9 : False := n3 n8 ; exact n9

-- textbook_b225c246ef315fe80088 | forallx-calgary 7 | \item Show that the sentence $B \eif \enot S$ is equivalent to the sentence $\enot \enot B \eif \enot S$.
theorem textbook_b225c246ef315fe80088 ( P Q R S : Prop ) ( h1 : ( ( ¬ ( ¬ P ) ) → ( ¬ Q ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( ( ¬ ( ¬ P ) ) → ( ¬ Q ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : False := n5 n3 ; exact ( n6 : False ) ) ; have n7 : ( ¬ Q ) := n1 n4 ; exact n7 ) ; exact n2

-- textbook_241044592d16d6e4efed | forallx-calgary 9 | \item Show that the sentences $\enot(A \eor B)$, $C$, $C \eif A$  are jointly inconsistent.
theorem textbook_241044592d16d6e4efed ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) ( h2 : R ) ( h3 : ( R → P ) ) : False := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : R := h2 ; have n3 : ( R → P ) := h3 ; have n4 : P := n3 n2 ; have n5 : ( P ∨ Q ) := Or.inl n4 ; have n6 : False := n1 n5 ; exact n6

-- textbook_2e44197318839943ac9c | forallx-calgary 5 | \item Show that $ (A \eiff A) \eand (B \eand \enot B)$ is a contradiction.
theorem textbook_2e44197318839943ac9c ( P Q R S : Prop ) ( h1 : ( ( ( P → P ) ∧ ( P → P ) ) ∧ ( Q ∧ ( ¬ Q ) ) ) ) : False := by have n1 : ( ( ( P → P ) ∧ ( P → P ) ) ∧ ( Q ∧ ( ¬ Q ) ) ) := h1 ; have n2 : ( Q ∧ ( ¬ Q ) ) := n1.2 ; have n3 : Q := n2.1 ; have n4 : ( ¬ Q ) := n2.2 ; have n5 : False := n4 n3 ; exact n5

-- textbook_a96986bb1c65943f0b03 | forallx-calgary 11 | \item Show that $\enot \enot (C \eiff \enot C), ((G \eor C) \eor G) \therefore ((G \eif C) \eand G) $ is valid.
theorem textbook_a96986bb1c65943f0b03 ( P Q R S : Prop ) ( h1 : ( ¬ ( ¬ ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) ) ) ( h2 : ( ( Q ∨ P ) ∨ Q ) ) : ( ( Q → P ) ∧ Q ) := by have n1 : ( ¬ ( ¬ ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) ) := h1 ; have n2 : ( ( Q ∨ P ) ∨ Q ) := h2 ; have n3 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; have n4 : ( P → ( ¬ P ) ) := n3.1 ; have n5 : ( ¬ P ) := ( fun ( n6 : P ) => by have n7 : ( ¬ P ) := n4 n6 ; have n8 : False := n7 n6 ; exact ( n8 : False ) ) ; have n9 : ( ( ¬ P ) → P ) := n3.2 ; have n10 : P := n9 n5 ; have n11 : False := n5 n10 ; have n12 : ( ( Q → P ) ∧ Q ) := n11.elim ; exact n12

-- textbook_34ece9058da2761b0326 | forallx-syracuse 1 | \item $A \eand (B \eand C) \proves (A \land B) \eand C$
theorem textbook_34ece9058da2761b0326 ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∧ R ) ) ) : ( ( P ∧ Q ) ∧ R ) := by have n1 : ( P ∧ ( Q ∧ R ) ) := h1 ; have n2 : P := n1.1 ; have n3 : ( Q ∧ R ) := n1.2 ; have n4 : Q := n3.1 ; have n5 : ( P ∧ Q ) := ⟨ n2 , n4 ⟩ ; have n6 : R := n3.2 ; have n7 : ( ( P ∧ Q ) ∧ R ) := ⟨ n5 , n6 ⟩ ; exact n7

-- textbook_3cce87a8e9784b381b83 | forallx-syracuse 2 | \item $P \eand  Q, \ R, \ (Q \eand  R) \eif S \vdash S$
theorem textbook_3cce87a8e9784b381b83 ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : R ) ( h3 : ( ( Q ∧ R ) → S ) ) : S := by have n1 : ( P ∧ Q ) := h1 ; have n2 : R := h2 ; have n3 : ( ( Q ∧ R ) → S ) := h3 ; have n4 : Q := n1.2 ; have n5 : ( Q ∧ R ) := ⟨ n4 , n2 ⟩ ; have n6 : S := n3 n5 ; exact n6

-- textbook_fcf1f314b60800e5aff4 | forallx-syracuse 3 | \item $P \eand Q, (Q \eand P) \eif R \proves R$
theorem textbook_fcf1f314b60800e5aff4 ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q ∧ P ) → R ) ) : R := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q ∧ P ) → R ) := h2 ; have n3 : Q := n1.2 ; have n4 : P := n1.1 ; have n5 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; have n6 : R := n2 n5 ; exact n6

-- textbook_4cb66e0be9e0934acab6 | forallx-syracuse 4 | \item $A \eand B, B \eif (A \eif C) \proves C \eand B$
theorem textbook_4cb66e0be9e0934acab6 ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( Q → ( P → R ) ) ) : ( R ∧ Q ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( Q → ( P → R ) ) := h2 ; have n3 : Q := n1.2 ; have n4 : ( P → R ) := n2 n3 ; have n5 : P := n1.1 ; have n6 : R := n4 n5 ; have n7 : ( R ∧ Q ) := ⟨ n6 , n3 ⟩ ; exact n7

-- textbook_d082ad7b7083a55db8a1 | forallx-syracuse 5 | \item $(A\eif ((C\eand  A)\eif D)), (A\eand  C) \vdash D$
theorem textbook_d082ad7b7083a55db8a1 ( P Q R S : Prop ) ( h1 : ( P → ( ( Q ∧ P ) → R ) ) ) ( h2 : ( P ∧ Q ) ) : R := by have n1 : ( P → ( ( Q ∧ P ) → R ) ) := h1 ; have n2 : ( P ∧ Q ) := h2 ; have n3 : Q := n2.2 ; have n4 : P := n2.1 ; have n5 : ( ( Q ∧ P ) → R ) := n1 n4 ; have n6 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; have n7 : R := n5 n6 ; exact n7

-- textbook_10a0b9b200c3d7fdc17b | forallx-syracuse 1 | \item $(A\eif (B\eand  D)), (A\eif C) \vdash (A\eif (C\eand  D))$
theorem textbook_10a0b9b200c3d7fdc17b ( P Q R S : Prop ) ( h1 : ( P → ( Q ∧ R ) ) ) ( h2 : ( P → S ) ) : ( P → ( S ∧ R ) ) := by have n1 : ( P → ( Q ∧ R ) ) := h1 ; have n2 : ( P → S ) := h2 ; have n3 : ( P → ( S ∧ R ) ) := ( fun ( n4 : P ) => by have n5 : S := n2 n4 ; have n6 : ( Q ∧ R ) := n1 n4 ; have n7 : R := n6.2 ; have n8 : ( S ∧ R ) := ⟨ n5 , n7 ⟩ ; exact n8 ) ; exact n3

-- textbook_6675f2c58a518d1d279c | forallx-syracuse 3 | \item $A \eif B, B \eif C \proves A \eif (B \eand C)$
theorem textbook_6675f2c58a518d1d279c ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → R ) ) : ( P → ( Q ∧ R ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( P → ( Q ∧ R ) ) := ( fun ( n4 : P ) => by have n5 : Q := n1 n4 ; have n6 : R := n2 n5 ; have n7 : ( Q ∧ R ) := ⟨ n5 , n6 ⟩ ; exact n7 ) ; exact n3

-- textbook_f2569bc3cd708f5eb866 | forallx-syracuse 4 | \item $(A\eand  B)\eif C, A\eif B \vdash A\eif C$
theorem textbook_f2569bc3cd708f5eb866 ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) ( h2 : ( P → Q ) ) : ( P → R ) := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : Q := n2 n4 ; have n6 : ( P ∧ Q ) := ⟨ n4 , n5 ⟩ ; have n7 : R := n1 n6 ; exact n7 ) ; exact n3

-- textbook_5422cf04abf318dd32a9 | forallx-syracuse 5 | \item $A \eif B \proves (A \eand C) \eif (B \eand C)$
theorem textbook_5422cf04abf318dd32a9 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( P ∧ R ) → ( Q ∧ R ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( P ∧ R ) → ( Q ∧ R ) ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : P := n3.1 ; have n5 : Q := n1 n4 ; have n6 : R := n3.2 ; have n7 : ( Q ∧ R ) := ⟨ n5 , n6 ⟩ ; exact n7 ) ; exact n2

-- textbook_7c718c1db52fdc350567 | forallx-syracuse 2 | \item $\vdash (A\eif B)\eif ((B\eif C)\eif (A\eif C))$
theorem textbook_7c718c1db52fdc350567 ( P Q R S : Prop ) : ( ( P → Q ) → ( ( Q → R ) → ( P → R ) ) ) := by have n1 : ( ( P → Q ) → ( ( Q → R ) → ( P → R ) ) ) := ( fun ( n2 : ( P → Q ) ) => by have n3 : ( ( Q → R ) → ( P → R ) ) := ( fun ( n4 : ( Q → R ) ) => by have n5 : ( P → R ) := ( fun ( n6 : P ) => by have n7 : Q := n2 n6 ; have n8 : R := n4 n7 ; exact n8 ) ; exact n5 ) ; exact n3 ) ; exact n1

-- textbook_4d60a4214d6ac5091846 | forallx-syracuse 3 | \item $\vdash (A \eif (B \eif C)) \eif (B \eif (A \eif C))$
theorem textbook_4d60a4214d6ac5091846 ( P Q R S : Prop ) : ( ( P → ( Q → R ) ) → ( Q → ( P → R ) ) ) := by have n1 : ( ( P → ( Q → R ) ) → ( Q → ( P → R ) ) ) := ( fun ( n2 : ( P → ( Q → R ) ) ) => by have n3 : ( Q → ( P → R ) ) := ( fun ( n4 : Q ) => by have n5 : ( P → R ) := ( fun ( n6 : P ) => by have n7 : ( Q → R ) := n2 n6 ; have n8 : R := n7 n4 ; exact n8 ) ; exact n5 ) ; exact n3 ) ; exact n1

-- textbook_cfa22f27d98eb6b067c6 | forallx-syracuse 3 | \item $A \eiff B \vdash (A \eand C) \eiff (B \eand C)$
theorem textbook_cfa22f27d98eb6b067c6 ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) : ( ( ( P ∧ R ) → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → ( P ∧ R ) ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( ( P ∧ R ) → ( Q ∧ R ) ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : ( P → Q ) := n1.1 ; have n5 : P := n3.1 ; have n6 : Q := n4 n5 ; have n7 : R := n3.2 ; have n8 : ( Q ∧ R ) := ⟨ n6 , n7 ⟩ ; exact n8 ) ; have n9 : ( ( Q ∧ R ) → ( P ∧ R ) ) := ( fun ( n10 : ( Q ∧ R ) ) => by have n11 : ( Q → P ) := n1.2 ; have n12 : Q := n10.1 ; have n13 : P := n11 n12 ; have n14 : R := n10.2 ; have n15 : ( P ∧ R ) := ⟨ n13 , n14 ⟩ ; exact n15 ) ; have n16 : ( ( ( P ∧ R ) → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → ( P ∧ R ) ) ) := ⟨ n2 , n9 ⟩ ; exact n16

-- textbook_9443266b19960950ac62 | forallx-syracuse 4 | \item $(A \eand B) \eiff (A \eand C) \vdash A \eif (B \eiff C)$
theorem textbook_9443266b19960950ac62 ( P Q R S : Prop ) ( h1 : ( ( ( P ∧ Q ) → ( P ∧ R ) ) ∧ ( ( P ∧ R ) → ( P ∧ Q ) ) ) ) : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := by have n1 : ( ( ( P ∧ Q ) → ( P ∧ R ) ) ∧ ( ( P ∧ R ) → ( P ∧ Q ) ) ) := h1 ; have n2 : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := ( fun ( n3 : P ) => by have n4 : ( Q → R ) := ( fun ( n5 : Q ) => by have n6 : ( ( P ∧ Q ) → ( P ∧ R ) ) := n1.1 ; have n7 : ( P ∧ Q ) := ⟨ n3 , n5 ⟩ ; have n8 : ( P ∧ R ) := n6 n7 ; have n9 : R := n8.2 ; exact n9 ) ; have n10 : ( R → Q ) := ( fun ( n11 : R ) => by have n12 : ( ( P ∧ R ) → ( P ∧ Q ) ) := n1.2 ; have n13 : ( P ∧ R ) := ⟨ n3 , n11 ⟩ ; have n14 : ( P ∧ Q ) := n12 n13 ; have n15 : Q := n14.2 ; exact n15 ) ; have n16 : ( ( Q → R ) ∧ ( R → Q ) ) := ⟨ n4 , n10 ⟩ ; exact n16 ) ; exact n2

-- textbook_87a23fa584426b3d9328 | forallx-syracuse 6 | \item $P \rightarrow R,  \ R\rightarrow (S \land P) \vdash P \leftrightarrow R$
theorem textbook_87a23fa584426b3d9328 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( R ∧ P ) ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( R ∧ P ) ) := h2 ; have n3 : ( Q → P ) := ( fun ( n4 : Q ) => by have n5 : ( R ∧ P ) := n2 n4 ; have n6 : P := n5.2 ; exact n6 ) ; have n7 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n1 , n3 ⟩ ; exact n7

-- textbook_36d8dc85a05b9a9d5115 | forallx-syracuse 7 | \item $P \leftrightarrow (Q \land R), \ R \rightarrow P  \ \vdash \ P \leftrightarrow R$
theorem textbook_36d8dc85a05b9a9d5115 ( P Q R S : Prop ) ( h1 : ( ( P → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → P ) ) ) ( h2 : ( R → P ) ) : ( ( P → R ) ∧ ( R → P ) ) := by have n1 : ( ( P → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → P ) ) := h1 ; have n2 : ( R → P ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : ( P → ( Q ∧ R ) ) := n1.1 ; have n6 : ( Q ∧ R ) := n5 n4 ; have n7 : R := n6.2 ; exact n7 ) ; have n8 : ( ( P → R ) ∧ ( R → P ) ) := ⟨ n3 , n2 ⟩ ; exact n8

-- textbook_7522199c8983b99735c5 | forallx-syracuse 2 | \item $P \rightarrow \lnot Q \vdash Q \rightarrow \lnot P$
theorem textbook_7522199c8983b99735c5 ( P Q R S : Prop ) ( h1 : ( P → ( ¬ Q ) ) ) : ( Q → ( ¬ P ) ) := by have n1 : ( P → ( ¬ Q ) ) := h1 ; have n2 : ( Q → ( ¬ P ) ) := ( fun ( n3 : Q ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( ¬ Q ) := n1 n5 ; have n7 : False := n6 n3 ; exact ( n7 : False ) ) ; exact n4 ) ; exact n2

-- textbook_2a874b10554baacdab52 | forallx-syracuse 7 | \item $A \eif B, B \eif \enot A \vdash \enot A$
theorem textbook_2a874b10554baacdab52 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( ¬ P ) ) ) : ( ¬ P ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( ¬ P ) ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : Q := n1 n4 ; have n6 : ( ¬ P ) := n2 n5 ; have n7 : False := n6 n4 ; exact ( n7 : False ) ) ; exact n3

-- textbook_03e2f5be685c0d2ac60a | forallx-syracuse 8 | \item $(A \eand B) \eif \enot A \vdash A \eif \enot B$
theorem textbook_03e2f5be685c0d2ac60a ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → ( ¬ P ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( ( P ∧ Q ) → ( ¬ P ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : ( P ∧ Q ) := ⟨ n3 , n5 ⟩ ; have n7 : ( ¬ P ) := n1 n6 ; have n8 : False := n7 n3 ; exact ( n8 : False ) ) ; exact n4 ) ; exact n2

-- textbook_4814343dfe60095b930f | forallx-syracuse 10 | \item $(A \eand \enot B) \eif B \vdash A \eif B$
theorem textbook_4814343dfe60095b930f ( P Q R S : Prop ) ( h1 : ( ( P ∧ ( ¬ Q ) ) → Q ) ) : ( P → Q ) := by have n1 : ( ( P ∧ ( ¬ Q ) ) → Q ) := h1 ; have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : ( P ∧ ( ¬ Q ) ) := ⟨ n3 , n5 ⟩ ; have n7 : Q := n1 n6 ; have n8 : False := n5 n7 ; exact ( n8 : False ) ) ; have n9 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n9 ) ; exact n2

-- textbook_ed9c41eb74dfb88bfa3f | forallx-syracuse 11 | \item $(A\eiff \enot B) \vdash \enot (A\eiff B)$
theorem textbook_ed9c41eb74dfb88bfa3f ( P Q R S : Prop ) ( h1 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) ) : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := by have n1 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := h1 ; have n2 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := ( fun ( n3 : ( ( P → Q ) ∧ ( Q → P ) ) ) => by have n4 : ( P → ( ¬ Q ) ) := n1.1 ; have n5 : ( Q → P ) := n3.2 ; have n6 : ( ¬ Q ) := ( fun ( n7 : Q ) => by have n8 : P := n5 n7 ; have n9 : ( ¬ Q ) := n4 n8 ; have n10 : False := n9 n7 ; exact ( n10 : False ) ) ; have n11 : ( P → Q ) := n3.1 ; have n12 : ( ( ¬ Q ) → P ) := n1.2 ; have n13 : P := n12 n6 ; have n14 : Q := n11 n13 ; have n15 : False := n6 n14 ; exact ( n15 : False ) ) ; exact n2

-- textbook_7413a625e07c41fff3a3 | forallx-syracuse 16 | \item $\vdash \enot P \eif( P \eif Q)$
theorem textbook_7413a625e07c41fff3a3 ( P Q R S : Prop ) : ( ( ¬ P ) → ( P → Q ) ) := by have n1 : ( ( ¬ P ) → ( P → Q ) ) := ( fun ( n2 : ( ¬ P ) ) => by have n3 : ( P → Q ) := ( fun ( n4 : P ) => by have n5 : False := n2 n4 ; have n6 : Q := n5.elim ; exact n6 ) ; exact n3 ) ; exact n1

-- textbook_588452d1ce99814559b0 | forallx-syracuse 18 | \item $\vdash (\lnot A \rightarrow A) \rightarrow A$
theorem textbook_588452d1ce99814559b0 ( P Q R S : Prop ) : ( ( ( ¬ P ) → P ) → P ) := by have n1 : ( ( ( ¬ P ) → P ) → P ) := ( fun ( n2 : ( ( ¬ P ) → P ) ) => by have n3 : ( ¬ ( ¬ P ) ) := ( fun ( n4 : ( ¬ P ) ) => by have n5 : P := n2 n4 ; have n6 : False := n4 n5 ; exact ( n6 : False ) ) ; have n7 : P := Classical.byContradiction ( fun hh => n3 hh ) ; exact n7 ) ; exact n1

-- textbook_bb93cb70858a3de279af | forallx-syracuse 2 | \item $(B \eor A) \eif C \vdash A \eif C$
theorem textbook_bb93cb70858a3de279af ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) → R ) ) : ( Q → R ) := by have n1 : ( ( P ∨ Q ) → R ) := h1 ; have n2 : ( Q → R ) := ( fun ( n3 : Q ) => by have n4 : ( P ∨ Q ) := Or.inr n3 ; have n5 : R := n1 n4 ; exact n5 ) ; exact n2

-- textbook_5de38c2037f06951c19e | forallx-syracuse 3 | \item $(A \eand B) \eor (A \eand C) \vdash A$
theorem textbook_5de38c2037f06951c19e ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( P ∧ R ) ) ) : P := by have n1 : ( ( P ∧ Q ) ∨ ( P ∧ R ) ) := h1 ; have n2 : P := Or.elim n1 ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : P := n3.1 ; exact n4 ) ( fun ( n5 : ( P ∧ R ) ) => by have n6 : P := n5.1 ; exact n6 ) ; exact n2

-- textbook_681e5816d2388b53e3d9 | forallx-syracuse 8 | \item $A \eor (B \eor C) \vdash (A \eor B) \eor C$
theorem textbook_681e5816d2388b53e3d9 ( P Q R S : Prop ) ( h1 : ( P ∨ ( Q ∨ R ) ) ) : ( ( P ∨ Q ) ∨ R ) := by have n1 : ( P ∨ ( Q ∨ R ) ) := h1 ; have n2 : ( ( P ∨ Q ) ∨ R ) := Or.elim n1 ( fun ( n3 : P ) => by have n4 : ( P ∨ Q ) := Or.inl n3 ; have n5 : ( ( P ∨ Q ) ∨ R ) := Or.inl n4 ; exact n5 ) ( fun ( n6 : ( Q ∨ R ) ) => by have n7 : ( ( P ∨ Q ) ∨ R ) := Or.elim n6 ( fun ( n8 : Q ) => by have n9 : ( P ∨ Q ) := Or.inr n8 ; have n10 : ( ( P ∨ Q ) ∨ R ) := Or.inl n9 ; exact n10 ) ( fun ( n11 : R ) => by have n12 : ( ( P ∨ Q ) ∨ R ) := Or.inr n11 ; exact n12 ) ; exact n7 ) ; exact n2

-- textbook_06ccd6ec0200aeb51862 | forallx-syracuse 10 | \item $(A \eor B) \eor C, \ C \eiff B \proves C \eor A$
theorem textbook_06ccd6ec0200aeb51862 ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∨ R ) ) ( h2 : ( ( R → Q ) ∧ ( Q → R ) ) ) : ( R ∨ P ) := by have n1 : ( ( P ∨ Q ) ∨ R ) := h1 ; have n2 : ( ( R → Q ) ∧ ( Q → R ) ) := h2 ; have n3 : ( R ∨ P ) := Or.elim n1 ( fun ( n4 : ( P ∨ Q ) ) => by have n5 : ( R ∨ P ) := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( R ∨ P ) := Or.inr n6 ; exact n7 ) ( fun ( n8 : Q ) => by have n9 : ( Q → R ) := n2.2 ; have n10 : R := n9 n8 ; have n11 : ( R ∨ P ) := Or.inl n10 ; exact n11 ) ; exact n5 ) ( fun ( n12 : R ) => by have n13 : ( R ∨ P ) := Or.inl n12 ; exact n13 ) ; exact n3

-- textbook_b1f54ffd10be0d8d9b52 | forallx-syracuse 12 | \item $A \eif  C \vdash (A \eor (B \eand C)) \eif C$
theorem textbook_b1f54ffd10be0d8d9b52 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( P ∨ ( R ∧ Q ) ) → Q ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( P ∨ ( R ∧ Q ) ) → Q ) := ( fun ( n3 : ( P ∨ ( R ∧ Q ) ) ) => by have n4 : Q := Or.elim n3 ( fun ( n5 : P ) => by have n6 : Q := n1 n5 ; exact n6 ) ( fun ( n7 : ( R ∧ Q ) ) => by have n8 : Q := n7.2 ; exact n8 ) ; exact n4 ) ; exact n2

-- textbook_911ca699a25f56477d3e | forallx-syracuse 13 | \item $J \eif I, \ L \rightarrow (L \rightarrow K)  \proves (L \eor J) \eif (I \eor K)$
theorem textbook_911ca699a25f56477d3e ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R → ( R → S ) ) ) : ( ( R ∨ P ) → ( Q ∨ S ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R → ( R → S ) ) := h2 ; have n3 : ( ( R ∨ P ) → ( Q ∨ S ) ) := ( fun ( n4 : ( R ∨ P ) ) => by have n5 : ( Q ∨ S ) := Or.elim n4 ( fun ( n6 : R ) => by have n7 : ( R → S ) := n2 n6 ; have n8 : S := n7 n6 ; have n9 : ( Q ∨ S ) := Or.inr n8 ; exact n9 ) ( fun ( n10 : P ) => by have n11 : Q := n1 n10 ; have n12 : ( Q ∨ S ) := Or.inl n11 ; exact n12 ) ; exact n5 ) ; exact n3

-- textbook_5fe19ee857dfdf3b89d2 | forallx-syracuse 14 | \item $D \eif A, \ C \eif (B \eand A) \proves (D \eor C) \eif (B \eor A)$
theorem textbook_5fe19ee857dfdf3b89d2 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R → ( S ∧ Q ) ) ) : ( ( P ∨ R ) → ( S ∨ Q ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R → ( S ∧ Q ) ) := h2 ; have n3 : ( ( P ∨ R ) → ( S ∨ Q ) ) := ( fun ( n4 : ( P ∨ R ) ) => by have n5 : Q := Or.elim n4 ( fun ( n6 : P ) => by have n7 : Q := n1 n6 ; exact n7 ) ( fun ( n8 : R ) => by have n9 : ( S ∧ Q ) := n2 n8 ; have n10 : Q := n9.2 ; exact n10 ) ; have n11 : ( S ∨ Q ) := Or.inr n5 ; exact n11 ) ; exact n3

-- textbook_c5140455df9355840a6c | forallx-syracuse 15 | \item $Z \lor M, \ M \eif K \vdash  Z \lor K $
theorem textbook_c5140455df9355840a6c ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) ( h2 : ( Q → R ) ) : ( P ∨ R ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( P ∨ R ) := Or.elim n1 ( fun ( n4 : P ) => by have n5 : ( P ∨ R ) := Or.inl n4 ; exact n5 ) ( fun ( n6 : Q ) => by have n7 : R := n2 n6 ; have n8 : ( P ∨ R ) := Or.inr n7 ; exact n8 ) ; exact n3

-- textbook_4a8cfd28c11708238126 | forallx-syracuse 4 | \item $\enot R \eor (P \eif Q) \vdash (R \eand P) \eif Q$
theorem textbook_4a8cfd28c11708238126 ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) ∨ ( Q → R ) ) ) : ( ( P ∧ Q ) → R ) := by have n1 : ( ( ¬ P ) ∨ ( Q → R ) ) := h1 ; have n2 : ( ( P ∧ Q ) → R ) := ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : ( Q → R ) := Or.elim n1 ( fun ( n5 : ( ¬ P ) ) => by have n6 : P := n3.1 ; have n7 : False := n5 n6 ; have n8 : ( Q → R ) := n7.elim ; exact n8 ) ( fun ( n9 : ( Q → R ) ) => by exact n9 ) ; have n10 : Q := n3.2 ; have n11 : R := n4 n10 ; exact n11 ) ; exact n2

-- textbook_0ad92464052745011778 | forallx-syracuse 3 | \item $P \eor Q \pequiv \enot (\enot P \eand \enot Q)$
theorem textbook_0ad92464052745011778 ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := ( fun ( n3 : ( ( ¬ P ) ∧ ( ¬ Q ) ) ) => by have n4 : False := Or.elim n1 ( fun ( n5 : P ) => by have n6 : ( ¬ P ) := n3.1 ; have n7 : False := n6 n5 ; exact n7 ) ( fun ( n8 : Q ) => by have n9 : ( ¬ Q ) := n3.2 ; have n10 : False := n9 n8 ; exact n10 ) ; exact ( n4 : False ) ) ; exact n2

-- textbook_a44e77e0ea54ed00a768 | forallx-syracuse 3 | \item $P \eor Q \pequiv \enot (\enot P \eand \enot Q)$
theorem textbook_a44e77e0ea54ed00a768 ( P Q R S : Prop ) ( h1 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) : ( P ∨ Q ) := by have n1 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := h1 ; have n2 : ( ¬ ( ¬ ( P ∨ Q ) ) ) := ( fun ( n3 : ( ¬ ( P ∨ Q ) ) ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( P ∨ Q ) := Or.inl n5 ; have n7 : False := n3 n6 ; exact ( n7 : False ) ) ; have n8 : ( ¬ Q ) := ( fun ( n9 : Q ) => by have n10 : ( P ∨ Q ) := Or.inr n9 ; have n11 : False := n3 n10 ; exact ( n11 : False ) ) ; have n12 : ( ( ¬ P ) ∧ ( ¬ Q ) ) := ⟨ n4 , n8 ⟩ ; have n13 : False := n1 n12 ; exact ( n13 : False ) ) ; have n14 : ( P ∨ Q ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n14

-- textbook_b7132b7858d742201ec0 | forallx-syracuse 7 | \item $P \eiff \enot Q \pequiv \enot(P \eiff Q)$
theorem textbook_b7132b7858d742201ec0 ( P Q R S : Prop ) ( h1 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) ) : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := by have n1 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : ( P → Q ) := ( fun ( n7 : P ) => by have n8 : Q := n5 ; exact n8 ) ; have n9 : ( Q → P ) := ( fun ( n10 : Q ) => by have n11 : P := n3 ; exact n11 ) ; have n12 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n6 , n9 ⟩ ; have n13 : False := n1 n12 ; exact ( n13 : False ) ) ; exact n4 ) ; have n14 : ( ( ¬ Q ) → P ) := ( fun ( n15 : ( ¬ Q ) ) => by have n16 : ( ¬ ( ¬ P ) ) := ( fun ( n17 : ( ¬ P ) ) => by have n18 : ( P → Q ) := ( fun ( n19 : P ) => by have n20 : False := n17 n19 ; have n21 : Q := n20.elim ; exact n21 ) ; have n22 : ( Q → P ) := ( fun ( n23 : Q ) => by have n24 : False := n15 n23 ; have n25 : P := n24.elim ; exact n25 ) ; have n26 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n18 , n22 ⟩ ; have n27 : False := n1 n26 ; exact ( n27 : False ) ) ; have n28 : P := Classical.byContradiction ( fun hh => n16 hh ) ; exact n28 ) ; have n29 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := ⟨ n2 , n14 ⟩ ; exact n29

-- textbook_de9fce762bb7ce5e677f | forallx-syracuse 8 | \item $P \eor Q \pequiv P \eor (\enot P \eand Q)$
theorem textbook_de9fce762bb7ce5e677f ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ¬ ( ¬ ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) ) := ( fun ( n3 : ( ¬ ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) ) => by have n4 : P := Or.elim n1 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : Q ) => by have n7 : ( ¬ ( ¬ P ) ) := ( fun ( n8 : ( ¬ P ) ) => by have n9 : ( ( ¬ P ) ∧ Q ) := ⟨ n8 , n6 ⟩ ; have n10 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Or.inr n9 ; have n11 : False := n3 n10 ; exact ( n11 : False ) ) ; have n12 : P := Classical.byContradiction ( fun hh => n7 hh ) ; exact n12 ) ; have n13 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Or.inl n4 ; have n14 : False := n3 n13 ; exact ( n14 : False ) ) ; have n15 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n15

-- textbook_9b042a4a7fa74ab578cb | forallx-syracuse 8 | \item $P \eor Q \pequiv P \eor (\enot P \eand Q)$
theorem textbook_9b042a4a7fa74ab578cb ( P Q R S : Prop ) ( h1 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) : ( P ∨ Q ) := by have n1 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := h1 ; have n2 : ( P ∨ Q ) := Or.elim n1 ( fun ( n3 : P ) => by have n4 : ( P ∨ Q ) := Or.inl n3 ; exact n4 ) ( fun ( n5 : ( ( ¬ P ) ∧ Q ) ) => by have n6 : Q := n5.2 ; have n7 : ( P ∨ Q ) := Or.inr n6 ; exact n7 ) ; exact n2

-- textbook_4447e76b45cccb03d640 | forallx-syracuse 1 | \item $(A \eor B) \eif C , \enot C \proves \enot A$
theorem textbook_4447e76b45cccb03d640 ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) → R ) ) ( h2 : ( ¬ R ) ) : ( ¬ P ) := by have n1 : ( ( P ∨ Q ) → R ) := h1 ; have n2 : ( ¬ R ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( P ∨ Q ) := Or.inl n4 ; have n6 : R := n1 n5 ; have n7 : False := n2 n6 ; exact ( n7 : False ) ) ; exact n3

-- textbook_1b0c86b846be078dbfc8 | forallx-syracuse 3 | \item $\enot (A \eif (B \eor \enot C)) \proves (B \eor C) \eif A$
theorem textbook_1b0c86b846be078dbfc8 ( P Q R S : Prop ) ( h1 : ( ¬ ( P → ( Q ∨ ( ¬ R ) ) ) ) ) : ( ( Q ∨ R ) → P ) := by have n1 : ( ¬ ( P → ( Q ∨ ( ¬ R ) ) ) ) := h1 ; have n2 : ( ( Q ∨ R ) → P ) := ( fun ( n3 : ( Q ∨ R ) ) => by have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : ( P → ( Q ∨ ( ¬ R ) ) ) := ( fun ( n7 : P ) => by have n8 : False := n5 n7 ; have n9 : ( Q ∨ ( ¬ R ) ) := n8.elim ; exact n9 ) ; have n10 : False := n1 n6 ; exact ( n10 : False ) ) ; have n11 : P := Classical.byContradiction ( fun hh => n4 hh ) ; exact n11 ) ; exact n2

-- textbook_17db74b2c27db844c116 | forallx-syracuse 8 | \item $(B \eand C) \eor \enot (A \eif \enot D), \enot C \proves A \eand D$
theorem textbook_17db74b2c27db844c116 ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( ¬ ( R → ( ¬ S ) ) ) ) ) ( h2 : ( ¬ Q ) ) : ( R ∧ S ) := by have n1 : ( ( P ∧ Q ) ∨ ( ¬ ( R → ( ¬ S ) ) ) ) := h1 ; have n2 : ( ¬ Q ) := h2 ; have n3 : ( ¬ ( ¬ ( R ∧ S ) ) ) := ( fun ( n4 : ( ¬ ( R ∧ S ) ) ) => by have n5 : False := Or.elim n1 ( fun ( n6 : ( P ∧ Q ) ) => by have n7 : Q := n6.2 ; have n8 : False := n2 n7 ; exact n8 ) ( fun ( n9 : ( ¬ ( R → ( ¬ S ) ) ) ) => by have n10 : ( R → ( ¬ S ) ) := ( fun ( n11 : R ) => by have n12 : ( ¬ S ) := ( fun ( n13 : S ) => by have n14 : ( R ∧ S ) := ⟨ n11 , n13 ⟩ ; have n15 : False := n4 n14 ; exact ( n15 : False ) ) ; exact n12 ) ; have n16 : False := n9 n10 ; exact n16 ) ; exact ( n5 : False ) ) ; have n17 : ( R ∧ S ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n17

-- textbook_bf61a48e793cb331db6e | forallx-msu 1 | \item $\enot P \eif (Q \eor P), \enot P \proves Q$\\
theorem textbook_bf61a48e793cb331db6e ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → ( Q ∨ P ) ) ) ( h2 : ( ¬ P ) ) : Q := by have n1 : ( ( ¬ P ) → ( Q ∨ P ) ) := h1 ; have n2 : ( ¬ P ) := h2 ; have n3 : ( Q ∨ P ) := n1 n2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : P ) => by have n7 : False := n2 n6 ; have n8 : Q := n7.elim ; exact n8 ) ; exact n4

-- textbook_9010f2a1ef772ceb138c | forallx-msu 2 | \item $P \eif (Q \eor \enot P), P \proves Q$\\
theorem textbook_9010f2a1ef772ceb138c ( P Q R S : Prop ) ( h1 : ( P → ( Q ∨ ( ¬ P ) ) ) ) ( h2 : P ) : Q := by have n1 : ( P → ( Q ∨ ( ¬ P ) ) ) := h1 ; have n2 : P := h2 ; have n3 : ( Q ∨ ( ¬ P ) ) := n1 n2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : ( ¬ P ) ) => by have n7 : False := n6 n2 ; have n8 : Q := n7.elim ; exact n8 ) ; exact n4

-- textbook_2af1c15bbd58c631b0db | forallx-msu 3 | \item $D \eand H, H \eiff J  \proves J \eor N$\smallskip
theorem textbook_2af1c15bbd58c631b0db ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q → R ) ∧ ( R → Q ) ) ) : ( R ∨ S ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q → R ) ∧ ( R → Q ) ) := h2 ; have n3 : ( Q → R ) := n2.1 ; have n4 : Q := n1.2 ; have n5 : R := n3 n4 ; have n6 : ( R ∨ S ) := Or.inl n5 ; exact n6

-- textbook_9b3b736284930ddf1e19 | forallx-msu 4 | \item $R \eand S, (S \eor Q) \eif T \proves T$\\
theorem textbook_9b3b736284930ddf1e19 ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q ∨ R ) → S ) ) : S := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q ∨ R ) → S ) := h2 ; have n3 : Q := n1.2 ; have n4 : ( Q ∨ R ) := Or.inl n3 ; have n5 : S := n2 n4 ; exact n5

-- textbook_1b669a5753a97113d412 | forallx-msu 6 | \item $P \eand (Q \eor R), P \eif \enot R \proves Q$\smallskip
theorem textbook_1b669a5753a97113d412 ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∨ R ) ) ) ( h2 : ( P → ( ¬ R ) ) ) : Q := by have n1 : ( P ∧ ( Q ∨ R ) ) := h1 ; have n2 : ( P → ( ¬ R ) ) := h2 ; have n3 : ( Q ∨ R ) := n1.2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : P := n1.1 ; have n8 : ( ¬ R ) := n2 n7 ; have n9 : False := n8 n6 ; have n10 : Q := n9.elim ; exact n10 ) ; exact n4

-- textbook_6a87e09cff5017971e42 | forallx-msu 7 | \item $(P \eor \enot Q) \eiff R, R \eand Q \proves P \eand R$\smallskip
theorem textbook_6a87e09cff5017971e42 ( P Q R S : Prop ) ( h1 : ( ( ( P ∨ ( ¬ Q ) ) → R ) ∧ ( R → ( P ∨ ( ¬ Q ) ) ) ) ) ( h2 : ( R ∧ Q ) ) : ( P ∧ R ) := by have n1 : ( ( ( P ∨ ( ¬ Q ) ) → R ) ∧ ( R → ( P ∨ ( ¬ Q ) ) ) ) := h1 ; have n2 : ( R ∧ Q ) := h2 ; have n3 : R := n2.1 ; have n4 : ( R → ( P ∨ ( ¬ Q ) ) ) := n1.2 ; have n5 : ( P ∨ ( ¬ Q ) ) := n4 n3 ; have n6 : P := Or.elim n5 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( ¬ Q ) ) => by have n9 : Q := n2.2 ; have n10 : False := n8 n9 ; have n11 : P := n10.elim ; exact n11 ) ; have n12 : ( P ∧ R ) := ⟨ n6 , n3 ⟩ ; exact n12

-- textbook_6fafb78b1ceff04e0dee | forallx-msu 8 | \item $(R \eand T) \eif Q, R \eor \enot P, P \eand T \proves Q$\\
theorem textbook_6fafb78b1ceff04e0dee ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) ( h2 : ( P ∨ ( ¬ S ) ) ) ( h3 : ( S ∧ Q ) ) : R := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( P ∨ ( ¬ S ) ) := h2 ; have n3 : ( S ∧ Q ) := h3 ; have n4 : P := Or.elim n2 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : ( ¬ S ) ) => by have n7 : S := n3.1 ; have n8 : False := n6 n7 ; have n9 : P := n8.elim ; exact n9 ) ; have n10 : Q := n3.2 ; have n11 : ( P ∧ Q ) := ⟨ n4 , n10 ⟩ ; have n12 : R := n1 n11 ; exact n12

-- textbook_c1d05f6df5cf79a1edb7 | forallx-msu 9 | \item $S \eif T, Q \eand \enot R, \enot R \eiff (T \eif S) \proves S \eiff T $\smallskip
theorem textbook_c1d05f6df5cf79a1edb7 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R ∧ ( ¬ S ) ) ) ( h3 : ( ( ( ¬ S ) → ( Q → P ) ) ∧ ( ( Q → P ) → ( ¬ S ) ) ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R ∧ ( ¬ S ) ) := h2 ; have n3 : ( ( ( ¬ S ) → ( Q → P ) ) ∧ ( ( Q → P ) → ( ¬ S ) ) ) := h3 ; have n4 : ( ( ¬ S ) → ( Q → P ) ) := n3.1 ; have n5 : ( ¬ S ) := n2.2 ; have n6 : ( Q → P ) := n4 n5 ; have n7 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n1 , n6 ⟩ ; exact n7

-- textbook_f84555d8d61f7d647653 | forallx-msu 10 | \item $(L \eor M) \eif N, P \eiff N, L \proves L \eand P$\smallskip
theorem textbook_f84555d8d61f7d647653 ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) → R ) ) ( h2 : ( ( S → R ) ∧ ( R → S ) ) ) ( h3 : P ) : ( P ∧ S ) := by have n1 : ( ( P ∨ Q ) → R ) := h1 ; have n2 : ( ( S → R ) ∧ ( R → S ) ) := h2 ; have n3 : P := h3 ; have n4 : ( R → S ) := n2.2 ; have n5 : ( P ∨ Q ) := Or.inl n3 ; have n6 : R := n1 n5 ; have n7 : S := n4 n6 ; have n8 : ( P ∧ S ) := ⟨ n3 , n7 ⟩ ; exact n8

-- textbook_e2c593f0543fc6234b2d | forallx-msu 12 | \item $S \eif T, T \eif S, T \eiff (T \eiff S) \proves S$\smallskip
theorem textbook_e2c593f0543fc6234b2d ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → P ) ) ( h3 : ( ( Q → ( ( Q → P ) ∧ ( P → Q ) ) ) ∧ ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) ) ) : P := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → P ) := h2 ; have n3 : ( ( Q → ( ( Q → P ) ∧ ( P → Q ) ) ) ∧ ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) ) := h3 ; have n4 : ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) := n3.2 ; have n5 : ( ( Q → P ) ∧ ( P → Q ) ) := ⟨ n2 , n1 ⟩ ; have n6 : Q := n4 n5 ; have n7 : P := n2 n6 ; exact n7

-- textbook_e5cfe82b533b6643a29c | forallx-msu 2 | \item $Q \eif R \proves (Q \eand S) \eif (R \eor T)$\smallskip
theorem textbook_e5cfe82b533b6643a29c ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( P ∧ R ) → ( Q ∨ S ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( P ∧ R ) → ( Q ∨ S ) ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : P := n3.1 ; have n5 : Q := n1 n4 ; have n6 : ( Q ∨ S ) := Or.inl n5 ; exact n6 ) ; exact n2

-- textbook_cd721dbd56f203710bb0 | forallx-msu 4 | \item $\proves \enot((R \eor T) \eand (\enot R  \eand \enot T))$\smallskip
theorem textbook_cd721dbd56f203710bb0 ( P Q R S : Prop ) : ( ¬ ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) := by have n1 : ( ¬ ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) := ( fun ( n2 : ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) => by have n3 : ( ( ¬ P ) ∧ ( ¬ Q ) ) := n2.2 ; have n4 : ( P ∨ Q ) := n2.1 ; have n5 : False := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( ¬ P ) := n3.1 ; have n8 : False := n7 n6 ; exact n8 ) ( fun ( n9 : Q ) => by have n10 : ( ¬ Q ) := n3.2 ; have n11 : False := n10 n9 ; exact n11 ) ; exact ( n5 : False ) ) ; exact n1

-- textbook_0b779b81d28876e7d49d | forallx-msu 5 | \item $\proves (R \eiff M) \eif (M \eif R)$\smallskip
theorem textbook_0b779b81d28876e7d49d ( P Q R S : Prop ) : ( ( ( P → Q ) ∧ ( Q → P ) ) → ( Q → P ) ) := by have n1 : ( ( ( P → Q ) ∧ ( Q → P ) ) → ( Q → P ) ) := ( fun ( n2 : ( ( P → Q ) ∧ ( Q → P ) ) ) => by have n3 : ( Q → P ) := n2.2 ; exact n3 ) ; exact n1

-- textbook_9d21fcccbcd6c3ebf207 | forallx-msu 2 | \item $\enot(M \eor P) \proves Q \eif \enot M$
theorem textbook_9d21fcccbcd6c3ebf207 ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) : ( R → ( ¬ P ) ) := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : ( R → ( ¬ P ) ) := ( fun ( n3 : R ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( P ∨ Q ) := Or.inl n5 ; have n7 : False := n1 n6 ; exact ( n7 : False ) ) ; exact n4 ) ; exact n2

-- textbook_bfb605b094950125182a | forallx-msu 3 | \item $\enot(M \eor P) \proves P \eif M$
theorem textbook_bfb605b094950125182a ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) : ( Q → P ) := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : ( Q → P ) := ( fun ( n3 : Q ) => by have n4 : ( P ∨ Q ) := Or.inr n3 ; have n5 : False := n1 n4 ; have n6 : P := n5.elim ; exact n6 ) ; exact n2

-- textbook_5c8618f27cf57fd6c99b | forallx-msu 7 | \item $F \eand (K \eand R) \proves F \eiff (K \eiff R)$
theorem textbook_5c8618f27cf57fd6c99b ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∧ R ) ) ) : ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) := by have n1 : ( P ∧ ( Q ∧ R ) ) := h1 ; have n2 : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := ( fun ( n3 : P ) => by have n4 : ( Q ∧ R ) := n1.2 ; have n5 : ( Q → R ) := ( fun ( n6 : Q ) => by have n7 : R := n4.2 ; exact n7 ) ; have n8 : ( R → Q ) := ( fun ( n9 : R ) => by have n10 : Q := n4.1 ; exact n10 ) ; have n11 : ( ( Q → R ) ∧ ( R → Q ) ) := ⟨ n5 , n8 ⟩ ; exact n11 ) ; have n12 : ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) := ( fun ( n13 : ( ( Q → R ) ∧ ( R → Q ) ) ) => by have n14 : P := n1.1 ; exact n14 ) ; have n15 : ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) := ⟨ n2 , n12 ⟩ ; exact n15

-- textbook_61c2a7da79562d1ee741 | carnap-book 4.1 | P:|-: P
theorem textbook_61c2a7da79562d1ee741 ( P Q R S : Prop ) ( h1 : P ) : P := by have n1 : P := h1 ; exact n1

-- textbook_5f07170d9486281717b7 | carnap-book 4.4 | ~P, ~Q->P :|-: Q
theorem textbook_5f07170d9486281717b7 ( P Q R S : Prop ) ( h1 : ( ¬ P ) ) ( h2 : ( ( ¬ Q ) → P ) ) : Q := by have n1 : ( ¬ P ) := h1 ; have n2 : ( ( ¬ Q ) → P ) := h2 ; have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : P := n2 n4 ; have n6 : False := n1 n5 ; exact ( n6 : False ) ) ; have n7 : Q := Classical.byContradiction ( fun hh => n3 hh ) ; exact n7

-- textbook_d37c5cb0774c7c951ead | carnap-book 4.5 | P, ~Q->~P :|-: Q
theorem textbook_d37c5cb0774c7c951ead ( P Q R S : Prop ) ( h1 : P ) ( h2 : ( ( ¬ Q ) → ( ¬ P ) ) ) : Q := by have n1 : P := h1 ; have n2 : ( ( ¬ Q ) → ( ¬ P ) ) := h2 ; have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : ( ¬ P ) := n2 n4 ; have n6 : False := n5 n1 ; exact ( n6 : False ) ) ; have n7 : Q := Classical.byContradiction ( fun hh => n3 hh ) ; exact n7

-- textbook_64e1ce87d7887b0f482e | carnap-book 5.10 | P->-Q, R->Q :|-: P-> -R
theorem textbook_64e1ce87d7887b0f482e ( P Q R S : Prop ) ( h1 : ( P → ( ¬ Q ) ) ) ( h2 : ( R → Q ) ) : ( P → ( ¬ R ) ) := by have n1 : ( P → ( ¬ Q ) ) := h1 ; have n2 : ( R → Q ) := h2 ; have n3 : ( P → ( ¬ R ) ) := ( fun ( n4 : P ) => by have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : Q := n2 n6 ; have n8 : ( ¬ Q ) := n1 n4 ; have n9 : False := n8 n7 ; exact ( n9 : False ) ) ; exact n5 ) ; exact n3

-- textbook_a6623fde05d2761f32e9 | carnap-book 6.2 | P->Q :|-: (Q->R)->(P->R)
theorem textbook_a6623fde05d2761f32e9 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( Q → R ) → ( P → R ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( Q → R ) → ( P → R ) ) := ( fun ( n3 : ( Q → R ) ) => by have n4 : ( P → R ) := ( fun ( n5 : P ) => by have n6 : Q := n1 n5 ; have n7 : R := n3 n6 ; exact n7 ) ; exact n4 ) ; exact n2

-- textbook_70f616e852eadc25705c | carnap-book 6.5 | (S->R)->Q, R :|-: Q
theorem textbook_70f616e852eadc25705c ( P Q R S : Prop ) ( h1 : ( ( P → Q ) → R ) ) ( h2 : Q ) : R := by have n1 : ( ( P → Q ) → R ) := h1 ; have n2 : Q := h2 ; have n3 : ( P → Q ) := ( fun ( n4 : P ) => by have n5 : Q := n2 ; exact n5 ) ; have n6 : R := n1 n3 ; exact n6

-- textbook_3512647041a54fd55274 | carnap-book 7.3 | -Q->R, S-> -R, -S->Q :|-: --Q
theorem textbook_3512647041a54fd55274 ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → Q ) ) ( h2 : ( R → ( ¬ Q ) ) ) ( h3 : ( ( ¬ R ) → P ) ) : ( ¬ ( ¬ P ) ) := by have n1 : ( ( ¬ P ) → Q ) := h1 ; have n2 : ( R → ( ¬ Q ) ) := h2 ; have n3 : ( ( ¬ R ) → P ) := h3 ; have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : ( ¬ R ) := ( fun ( n7 : R ) => by have n8 : Q := n1 n5 ; have n9 : ( ¬ Q ) := n2 n7 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; have n11 : P := n3 n6 ; have n12 : False := n5 n11 ; exact ( n12 : False ) ) ; exact n4

-- textbook_7726f61a9bc2c7c9f1b4 | carnap-book 7.4 | P->Q, Q->R, R->S, P :|-: -(P->-S)
theorem textbook_7726f61a9bc2c7c9f1b4 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → R ) ) ( h3 : ( R → S ) ) ( h4 : P ) : ( ¬ ( P → ( ¬ S ) ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( R → S ) := h3 ; have n4 : P := h4 ; have n5 : ( ¬ ( P → ( ¬ S ) ) ) := ( fun ( n6 : ( P → ( ¬ S ) ) ) => by have n7 : Q := n1 n4 ; have n8 : R := n2 n7 ; have n9 : S := n3 n8 ; have n10 : ( ¬ S ) := n6 n4 ; have n11 : False := n10 n9 ; exact ( n11 : False ) ) ; exact n5

-- textbook_cd69a60630955cea9a6d | carnap-book 7.5 | (P->Q)->R, S->(P->Q), -S->R :|-: --R
theorem textbook_cd69a60630955cea9a6d ( P Q R S : Prop ) ( h1 : ( ( P → Q ) → R ) ) ( h2 : ( S → ( P → Q ) ) ) ( h3 : ( ( ¬ S ) → R ) ) : ( ¬ ( ¬ R ) ) := by have n1 : ( ( P → Q ) → R ) := h1 ; have n2 : ( S → ( P → Q ) ) := h2 ; have n3 : ( ( ¬ S ) → R ) := h3 ; have n4 : ( ¬ ( ¬ R ) ) := ( fun ( n5 : ( ¬ R ) ) => by have n6 : ( ¬ S ) := ( fun ( n7 : S ) => by have n8 : ( P → Q ) := n2 n7 ; have n9 : R := n1 n8 ; have n10 : False := n5 n9 ; exact ( n10 : False ) ) ; have n11 : R := n3 n6 ; have n12 : False := n5 n11 ; exact ( n12 : False ) ) ; exact n4

-- textbook_95d404da37f07dfd2b34 | carnap-book 7.7 | :|-: (-P->P) -> --P
theorem textbook_95d404da37f07dfd2b34 ( P Q R S : Prop ) : ( ( ( ¬ P ) → P ) → ( ¬ ( ¬ P ) ) ) := by have n1 : ( ( ( ¬ P ) → P ) → ( ¬ ( ¬ P ) ) ) := ( fun ( n2 : ( ( ¬ P ) → P ) ) => by have n3 : ( ¬ ( ¬ P ) ) := ( fun ( n4 : ( ¬ P ) ) => by have n5 : P := n2 n4 ; have n6 : False := n4 n5 ; exact ( n6 : False ) ) ; exact n3 ) ; exact n1

-- textbook_d9dc2ee1269d002c72d2 | carnap-book 7.8 | R->(Q->P), Q->-P :|-: R->-Q
theorem textbook_d9dc2ee1269d002c72d2 ( P Q R S : Prop ) ( h1 : ( P → ( Q → R ) ) ) ( h2 : ( Q → ( ¬ R ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( P → ( Q → R ) ) := h1 ; have n2 : ( Q → ( ¬ R ) ) := h2 ; have n3 : ( P → ( ¬ Q ) ) := ( fun ( n4 : P ) => by have n5 : ( ¬ Q ) := ( fun ( n6 : Q ) => by have n7 : ( Q → R ) := n1 n4 ; have n8 : R := n7 n6 ; have n9 : ( ¬ R ) := n2 n6 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; exact n5 ) ; exact n3

-- textbook_c367f2ad3a68caa103ce | carnap-book 7.9 | P->Q, Q->R, R->S :|-: P->-(P->-S)
theorem textbook_c367f2ad3a68caa103ce ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → R ) ) ( h3 : ( R → S ) ) : ( P → ( ¬ ( P → ( ¬ S ) ) ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( R → S ) := h3 ; have n4 : ( P → ( ¬ ( P → ( ¬ S ) ) ) ) := ( fun ( n5 : P ) => by have n6 : ( ¬ ( P → ( ¬ S ) ) ) := ( fun ( n7 : ( P → ( ¬ S ) ) ) => by have n8 : Q := n1 n5 ; have n9 : R := n2 n8 ; have n10 : S := n3 n9 ; have n11 : ( ¬ S ) := n7 n5 ; have n12 : False := n11 n10 ; exact ( n12 : False ) ) ; exact n6 ) ; exact n4

-- textbook_d66e5d2ec8cab38d5b68 | carnap-book 7.10 | P->Q, (R->Q)->-P :|-: -P
theorem textbook_d66e5d2ec8cab38d5b68 ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( ( R → Q ) → ( ¬ P ) ) ) : ( ¬ P ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( R → Q ) → ( ¬ P ) ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( R → Q ) := ( fun ( n6 : R ) => by have n7 : Q := n1 n4 ; exact n7 ) ; have n8 : ( ¬ P ) := n2 n5 ; have n9 : False := n8 n4 ; exact ( n9 : False ) ) ; exact n3

-- textbook_0ab235a31e23b0bbbc17 | carnap-book 8.2 | P->(P->R), (P->R)->R :|-: R
theorem textbook_0ab235a31e23b0bbbc17 ( P Q R S : Prop ) ( h1 : ( P → ( P → Q ) ) ) ( h2 : ( ( P → Q ) → Q ) ) : Q := by have n1 : ( P → ( P → Q ) ) := h1 ; have n2 : ( ( P → Q ) → Q ) := h2 ; have n3 : ( P → Q ) := ( fun ( n4 : P ) => by have n5 : ( P → Q ) := n1 n4 ; have n6 : Q := n2 n5 ; exact n6 ) ; have n7 : Q := n2 n3 ; exact n7

-- textbook_00ac94727e7280640ead | carnap-book 8.5 | :|-: S -> ((-P->-Q)->(Q->P))
theorem textbook_00ac94727e7280640ead ( P Q R S : Prop ) : ( P → ( ( ( ¬ Q ) → ( ¬ R ) ) → ( R → Q ) ) ) := by have n1 : ( P → ( ( ( ¬ Q ) → ( ¬ R ) ) → ( R → Q ) ) ) := ( fun ( n2 : P ) => by have n3 : ( ( ( ¬ Q ) → ( ¬ R ) ) → ( R → Q ) ) := ( fun ( n4 : ( ( ¬ Q ) → ( ¬ R ) ) ) => by have n5 : ( R → Q ) := ( fun ( n6 : R ) => by have n7 : ( ¬ ( ¬ Q ) ) := ( fun ( n8 : ( ¬ Q ) ) => by have n9 : ( ¬ R ) := n4 n8 ; have n10 : False := n9 n6 ; exact ( n10 : False ) ) ; have n11 : Q := Classical.byContradiction ( fun hh => n7 hh ) ; exact n11 ) ; exact n5 ) ; exact n3 ) ; exact n1

-- textbook_43e30ad215c242c5605c | carnap-book 8.7 | P -> R, R -> -Q, S-> P, Q->S :|-: -Q
theorem textbook_43e30ad215c242c5605c ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( ¬ R ) ) ) ( h3 : ( S → P ) ) ( h4 : ( R → S ) ) : ( ¬ R ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( ¬ R ) ) := h2 ; have n3 : ( S → P ) := h3 ; have n4 : ( R → S ) := h4 ; have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : S := n4 n6 ; have n8 : P := n3 n7 ; have n9 : Q := n1 n8 ; have n10 : ( ¬ R ) := n2 n9 ; have n11 : False := n10 n6 ; exact ( n11 : False ) ) ; exact n5

-- textbook_49b7d961bb8d378750b8 | carnap-book 8.8 | -Q->--P, --R->-Q, -R-> P :|-: P
theorem textbook_49b7d961bb8d378750b8 ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → ( ¬ ( ¬ Q ) ) ) ) ( h2 : ( ( ¬ ( ¬ R ) ) → ( ¬ P ) ) ) ( h3 : ( ( ¬ R ) → Q ) ) : Q := by have n1 : ( ( ¬ P ) → ( ¬ ( ¬ Q ) ) ) := h1 ; have n2 : ( ( ¬ ( ¬ R ) ) → ( ¬ P ) ) := h2 ; have n3 : ( ( ¬ R ) → Q ) := h3 ; have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : ( ¬ ( ¬ R ) ) := ( fun ( n7 : ( ¬ R ) ) => by have n8 : Q := n3 n7 ; have n9 : False := n5 n8 ; exact ( n9 : False ) ) ; have n10 : ( ¬ P ) := n2 n6 ; have n11 : ( ¬ ( ¬ Q ) ) := n1 n10 ; have n12 : False := n11 n5 ; exact ( n12 : False ) ) ; have n13 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n13

-- textbook_f0b811e0abb0c54501ad | carnap-book 8.9 | P->-P, R->P, Q->R, -Q->R :|-: S
theorem textbook_f0b811e0abb0c54501ad ( P Q R S : Prop ) ( h1 : ( P → ( ¬ P ) ) ) ( h2 : ( Q → P ) ) ( h3 : ( R → Q ) ) ( h4 : ( ( ¬ R ) → Q ) ) : S := by have n1 : ( P → ( ¬ P ) ) := h1 ; have n2 : ( Q → P ) := h2 ; have n3 : ( R → Q ) := h3 ; have n4 : ( ( ¬ R ) → Q ) := h4 ; have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : Q := n3 n6 ; have n8 : P := n2 n7 ; have n9 : ( ¬ P ) := n1 n8 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; have n11 : Q := n4 n5 ; have n12 : P := n2 n11 ; have n13 : ( ¬ P ) := n1 n12 ; have n14 : False := n13 n12 ; have n15 : S := n14.elim ; exact n15

-- textbook_9f32715a4fdbe0ab0197 | carnap-book 9.3 | ~Q :|-: ~(P/\Q)
theorem textbook_9f32715a4fdbe0ab0197 ( P Q R S : Prop ) ( h1 : ( ¬ P ) ) : ( ¬ ( Q ∧ P ) ) := by have n1 : ( ¬ P ) := h1 ; have n2 : ( ¬ ( Q ∧ P ) ) := ( fun ( n3 : ( Q ∧ P ) ) => by have n4 : P := n3.2 ; have n5 : False := n1 n4 ; exact ( n5 : False ) ) ; exact n2

-- textbook_0961c3d8c294d4ccb124 | carnap-book 9.6 | :|-: P/\Q <-> Q/\P
theorem textbook_0961c3d8c294d4ccb124 ( P Q R S : Prop ) : ( ( ( P ∧ Q ) → ( Q ∧ P ) ) ∧ ( ( Q ∧ P ) → ( P ∧ Q ) ) ) := by have n1 : ( ( P ∧ Q ) → ( Q ∧ P ) ) := ( fun ( n2 : ( P ∧ Q ) ) => by have n3 : Q := n2.2 ; have n4 : P := n2.1 ; have n5 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; exact n5 ) ; have n6 : ( ( Q ∧ P ) → ( P ∧ Q ) ) := ( fun ( n7 : ( Q ∧ P ) ) => by have n8 : P := n7.2 ; have n9 : Q := n7.1 ; have n10 : ( P ∧ Q ) := ⟨ n8 , n9 ⟩ ; exact n10 ) ; have n11 : ( ( ( P ∧ Q ) → ( Q ∧ P ) ) ∧ ( ( Q ∧ P ) → ( P ∧ Q ) ) ) := ⟨ n1 , n6 ⟩ ; exact n11

-- textbook_dd8c336103f01138f76d | carnap-book 9.7 | P\/~Q, Q\/R, ~P :|-: R
theorem textbook_dd8c336103f01138f76d ( P Q R S : Prop ) ( h1 : ( P ∨ ( ¬ Q ) ) ) ( h2 : ( Q ∨ R ) ) ( h3 : ( ¬ P ) ) : R := by have n1 : ( P ∨ ( ¬ Q ) ) := h1 ; have n2 : ( Q ∨ R ) := h2 ; have n3 : ( ¬ P ) := h3 ; have n4 : R := Or.elim n2 ( fun ( n5 : Q ) => by have n6 : False := Or.elim n1 ( fun ( n7 : P ) => by have n8 : False := n3 n7 ; exact n8 ) ( fun ( n9 : ( ¬ Q ) ) => by have n10 : False := n9 n5 ; exact n10 ) ; have n11 : R := n6.elim ; exact n11 ) ( fun ( n12 : R ) => by exact n12 ) ; exact n4

-- textbook_87d996531769036ca6a8 | carnap-book 9.8 | P/\Q->R :|-: P/\~R->~Q
theorem textbook_87d996531769036ca6a8 ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) : ( ( P ∧ ( ¬ R ) ) → ( ¬ Q ) ) := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( ( P ∧ ( ¬ R ) ) → ( ¬ Q ) ) := ( fun ( n3 : ( P ∧ ( ¬ R ) ) ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : P := n3.1 ; have n7 : ( P ∧ Q ) := ⟨ n6 , n5 ⟩ ; have n8 : R := n1 n7 ; have n9 : ( ¬ R ) := n3.2 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; exact n4 ) ; exact n2

-- textbook_921620f6be6e3b10986f | carnap-book 9.9 | P/\~R -> ~Q :|-: P/\Q->R
theorem textbook_921620f6be6e3b10986f ( P Q R S : Prop ) ( h1 : ( ( P ∧ ( ¬ Q ) ) → ( ¬ R ) ) ) : ( ( P ∧ R ) → Q ) := by have n1 : ( ( P ∧ ( ¬ Q ) ) → ( ¬ R ) ) := h1 ; have n2 : ( ( P ∧ R ) → Q ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : R := n3.2 ; have n7 : P := n3.1 ; have n8 : ( P ∧ ( ¬ Q ) ) := ⟨ n7 , n5 ⟩ ; have n9 : ( ¬ R ) := n1 n8 ; have n10 : False := n9 n6 ; exact ( n10 : False ) ) ; have n11 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n11 ) ; exact n2

