-- textbook_9dce54aa7519dfd38167
theorem t ( P Q R S : Prop ) : ( ( ¬ ( P → Q ) ) → ( Q → P ) ) := by have n1 : ( ( ¬ ( P → Q ) ) → ( Q → P ) ) := ( fun ( n2 : ( ¬ ( P → Q ) ) ) => by have n3 : ( Q → P ) := ( fun ( n4 : Q ) => by have n5 : ( P → Q ) := ( fun ( n6 : P ) => by have n7 : Q := n4 ; exact n7 ) ; have n8 : False := n2 n5 ; have n9 : P := n8.elim ; exact n9 ) ; exact n3 ) ; exact n1

-- textbook_fd057a138ffd9a9c188a
theorem t ( P Q R S : Prop ) : ( ( ( ( ¬ P ) → Q ) → ( ( ¬ Q ) → P ) ) ∧ ( ( ( ¬ Q ) → P ) → ( ( ¬ P ) → Q ) ) ) := by have n1 : ( ( ( ¬ P ) → Q ) → ( ( ¬ Q ) → P ) ) := ( fun ( n2 : ( ( ¬ P ) → Q ) ) => by have n3 : ( ( ¬ Q ) → P ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : ( ¬ ( ¬ P ) ) := ( fun ( n6 : ( ¬ P ) ) => by have n7 : Q := n2 n6 ; have n8 : False := n4 n7 ; exact ( n8 : False ) ) ; have n9 : P := Classical.byContradiction ( fun hh => n5 hh ) ; exact n9 ) ; exact n3 ) ; have n10 : ( ( ( ¬ Q ) → P ) → ( ( ¬ P ) → Q ) ) := ( fun ( n11 : ( ( ¬ Q ) → P ) ) => by have n12 : ( ( ¬ P ) → Q ) := ( fun ( n13 : ( ¬ P ) ) => by have n14 : ( ¬ ( ¬ Q ) ) := ( fun ( n15 : ( ¬ Q ) ) => by have n16 : P := n11 n15 ; have n17 : False := n13 n16 ; exact ( n17 : False ) ) ; have n18 : Q := Classical.byContradiction ( fun hh => n14 hh ) ; exact n18 ) ; exact n12 ) ; have n19 : ( ( ( ( ¬ P ) → Q ) → ( ( ¬ Q ) → P ) ) ∧ ( ( ( ¬ Q ) → P ) → ( ( ¬ P ) → Q ) ) ) := ⟨ n1 , n10 ⟩ ; exact n19

-- textbook_05d366482a3b2495a3ec
theorem t ( P Q R S : Prop ) : ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) := by have n1 : ( ¬ ( ¬ ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) ) ) := ( fun ( n2 : ( ¬ ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) ) ) => by have n3 : ( ¬ ( ¬ ( ¬ P ) ) ) := ( fun ( n4 : ( ¬ ( ¬ P ) ) ) => by have n5 : P := Classical.byContradiction ( fun hh => n4 hh ) ; have n6 : ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) := Or.inl n5 ; have n7 : False := n2 n6 ; exact ( n7 : False ) ) ; have n8 : ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) := Or.inr n3 ; have n9 : False := n2 n8 ; exact ( n9 : False ) ) ; have n10 : ( P ∨ ( ¬ ( ¬ ( ¬ P ) ) ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; exact n10

-- textbook_8d12d86c4a1d9843debc
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( R ∧ P ) ) ) ( h3 : ( R → ( P ∨ Q ) ) ) : ( ( R → P ) ∧ ( P → R ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( R ∧ P ) ) := h2 ; have n3 : ( R → ( P ∨ Q ) ) := h3 ; have n4 : ( R → P ) := ( fun ( n5 : R ) => by have n6 : ( P ∨ Q ) := n3 n5 ; have n7 : P := Or.elim n6 ( fun ( n8 : P ) => by exact n8 ) ( fun ( n9 : Q ) => by have n10 : ( R ∧ P ) := n2 n9 ; have n11 : P := n10.2 ; exact n11 ) ; exact n7 ) ; have n12 : ( P → R ) := ( fun ( n13 : P ) => by have n14 : Q := n1 n13 ; have n15 : ( R ∧ P ) := n2 n14 ; have n16 : R := n15.1 ; exact n16 ) ; have n17 : ( ( R → P ) ∧ ( P → R ) ) := ⟨ n4 , n12 ⟩ ; exact n17

-- textbook_99f1eb6743d83db9b248
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( ( P → ( ¬ Q ) ) ∨ ( ¬ R ) ) ) ) : Q := by have n1 : ( ¬ ( ( P → ( ¬ Q ) ) ∨ ( ¬ R ) ) ) := h1 ; have n2 : ( ¬ ( ¬ Q ) ) := ( fun ( n3 : ( ¬ Q ) ) => by have n4 : ( P → ( ¬ Q ) ) := ( fun ( n5 : P ) => by have n6 : ( ¬ Q ) := n3 ; exact n6 ) ; have n7 : ( ( P → ( ¬ Q ) ) ∨ ( ¬ R ) ) := Or.inl n4 ; have n8 : False := n1 n7 ; exact ( n8 : False ) ) ; have n9 : Q := Classical.byContradiction ( fun hh => n2 hh ) ; exact n9

-- textbook_0e262e8e0d687eaeed5f
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ ( ¬ Q ) ) → R ) ) : ( ( ¬ R ) → ( ( ¬ P ) ∧ Q ) ) := by have n1 : ( ( P ∨ ( ¬ Q ) ) → R ) := h1 ; have n2 : ( ( ¬ R ) → ( ( ¬ P ) ∧ Q ) ) := ( fun ( n3 : ( ¬ R ) ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( P ∨ ( ¬ Q ) ) := Or.inl n5 ; have n7 : R := n1 n6 ; have n8 : False := n3 n7 ; exact ( n8 : False ) ) ; have n9 : ( ¬ ( ¬ Q ) ) := ( fun ( n10 : ( ¬ Q ) ) => by have n11 : ( P ∨ ( ¬ Q ) ) := Or.inr n10 ; have n12 : R := n1 n11 ; have n13 : False := n3 n12 ; exact ( n13 : False ) ) ; have n14 : Q := Classical.byContradiction ( fun hh => n9 hh ) ; have n15 : ( ( ¬ P ) ∧ Q ) := ⟨ n4 , n14 ⟩ ; exact n15 ) ; exact n2

-- textbook_9fc6c4ccff942858d7b3
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( P → R ) ) ( h3 : ( Q → S ) ) : ( R ∧ S ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( P → R ) := h2 ; have n3 : ( Q → S ) := h3 ; have n4 : P := n1.1 ; have n5 : R := n2 n4 ; have n6 : Q := n1.2 ; have n7 : S := n3 n6 ; have n8 : ( R ∧ S ) := ⟨ n5 , n7 ⟩ ; exact n8

-- textbook_27bbad163b491d468407
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∨ R ) ) ( h2 : ( ¬ R ) ) ( h3 : ( ¬ Q ) ) : ( P ∨ S ) := by have n1 : ( ( P ∨ Q ) ∨ R ) := h1 ; have n2 : ( ¬ R ) := h2 ; have n3 : ( ¬ Q ) := h3 ; have n4 : ( P ∨ Q ) := Or.elim n1 ( fun ( n5 : ( P ∨ Q ) ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : False := n2 n6 ; have n8 : ( P ∨ Q ) := n7.elim ; exact n8 ) ; have n9 : P := Or.elim n4 ( fun ( n10 : P ) => by exact n10 ) ( fun ( n11 : Q ) => by have n12 : False := n3 n11 ; have n13 : P := n12.elim ; exact n13 ) ; have n14 : ( P ∨ S ) := Or.inl n9 ; exact n14

-- textbook_2db1a03407fe63d072d5
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → Q ) ) ( h2 : ( ( ¬ P ) ∨ R ) ) ( h3 : ( ¬ R ) ) : Q := by have n1 : ( ( ¬ P ) → Q ) := h1 ; have n2 : ( ( ¬ P ) ∨ R ) := h2 ; have n3 : ( ¬ R ) := h3 ; have n4 : ( ¬ P ) := Or.elim n2 ( fun ( n5 : ( ¬ P ) ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : False := n3 n6 ; have n8 : ( ¬ P ) := n7.elim ; exact n8 ) ; have n9 : Q := n1 n4 ; exact n9

-- textbook_6f2140ac492f40fc582a
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( ( ¬ P ) ∨ Q ) ) ) ( h2 : ( Q → R ) ) : R := by have n1 : ( P ∧ ( ( ¬ P ) ∨ Q ) ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( ( ¬ P ) ∨ Q ) := n1.2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : ( ¬ P ) ) => by have n6 : P := n1.1 ; have n7 : False := n5 n6 ; have n8 : Q := n7.elim ; exact n8 ) ( fun ( n9 : Q ) => by exact n9 ) ; have n10 : R := n2 n4 ; exact n10

-- textbook_3f60b3f9f2c6ee461226
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( ( ¬ Q ) ∨ R ) ) ( h3 : ( ¬ R ) ) : ( ¬ P ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( ¬ Q ) ∨ R ) := h2 ; have n3 : ( ¬ R ) := h3 ; have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : False := Or.elim n2 ( fun ( n7 : ( ¬ Q ) ) => by have n8 : Q := n1 n5 ; have n9 : False := n7 n8 ; exact n9 ) ( fun ( n10 : R ) => by have n11 : False := n3 n10 ; exact n11 ) ; exact ( n6 : False ) ) ; exact n4

-- textbook_a67af6b24dccabdf9e78
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ R ) ) ( h2 : ( ¬ R ) ) ( h3 : ( ( ¬ ( ( ¬ P ) ∨ ( ¬ Q ) ) ) → S ) ) : S := by have n1 : ( ( P ∧ Q ) ∨ R ) := h1 ; have n2 : ( ¬ R ) := h2 ; have n3 : ( ( ¬ ( ( ¬ P ) ∨ ( ¬ Q ) ) ) → S ) := h3 ; have n4 : ( ¬ ( ( ¬ P ) ∨ ( ¬ Q ) ) ) := ( fun ( n5 : ( ( ¬ P ) ∨ ( ¬ Q ) ) ) => by have n6 : False := Or.elim n1 ( fun ( n7 : ( P ∧ Q ) ) => by have n8 : False := Or.elim n5 ( fun ( n9 : ( ¬ P ) ) => by have n10 : P := n7.1 ; have n11 : False := n9 n10 ; exact n11 ) ( fun ( n12 : ( ¬ Q ) ) => by have n13 : Q := n7.2 ; have n14 : False := n12 n13 ; exact n14 ) ; exact n8 ) ( fun ( n15 : R ) => by have n16 : False := n2 n15 ; exact n16 ) ; exact ( n6 : False ) ) ; have n17 : S := n3 n4 ; exact n17

-- textbook_a09d53a197d8697f9da5
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) ∨ Q ) ) ( h2 : ( P → R ) ) ( h3 : P ) : ( Q ∧ R ) := by have n1 : ( ( ¬ P ) ∨ Q ) := h1 ; have n2 : ( P → R ) := h2 ; have n3 : P := h3 ; have n4 : Q := Or.elim n1 ( fun ( n5 : ( ¬ P ) ) => by have n6 : False := n5 n3 ; have n7 : Q := n6.elim ; exact n7 ) ( fun ( n8 : Q ) => by exact n8 ) ; have n9 : R := n2 n3 ; have n10 : ( Q ∧ R ) := ⟨ n4 , n9 ⟩ ; exact n10

-- textbook_5d8d72fcae79efa3d780
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( P ∨ R ) ) ( h3 : ( R → S ) ) ( h4 : ( ( Q ∨ S ) → ( ¬ R ) ) ) : P := by have n1 : ( P → Q ) := h1 ; have n2 : ( P ∨ R ) := h2 ; have n3 : ( R → S ) := h3 ; have n4 : ( ( Q ∨ S ) → ( ¬ R ) ) := h4 ; have n5 : P := Or.elim n2 ( fun ( n6 : P ) => by exact n6 ) ( fun ( n7 : R ) => by have n8 : S := n3 n7 ; have n9 : ( Q ∨ S ) := Or.inr n8 ; have n10 : ( ¬ R ) := n4 n9 ; have n11 : False := n10 n7 ; have n12 : P := n11.elim ; exact n12 ) ; exact n5

