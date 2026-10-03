-- textbook_30018c9c991412364923
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( P ∧ ( ¬ P ) ) ) ) : ( ¬ P ) := by have n1 : ( P → ( P ∧ ( ¬ P ) ) ) := h1 ; have n2 : ( ¬ P ) := ( fun ( n3 : P ) => by have n4 : ( P ∧ ( ¬ P ) ) := n1 n3 ; have n5 : ( ¬ P ) := n4.2 ; have n6 : False := n5 n3 ; exact ( n6 : False ) ) ; exact n2

-- textbook_7c28a8f7423fde99f79c
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : Q := n1.2 ; exact n4 ) ; have n5 : ( Q → P ) := ( fun ( n6 : Q ) => by have n7 : P := n1.1 ; exact n7 ) ; have n8 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n2 , n5 ⟩ ; exact n8

-- textbook_95041444f8cb17626d64
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ R ) ) : ( R ∨ Q ) := by have n1 : ( ( P ∧ Q ) ∨ R ) := h1 ; have n2 : ( R ∨ Q ) := Or.elim n1 ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : Q := n3.2 ; have n5 : ( R ∨ Q ) := Or.inr n4 ; exact n5 ) ( fun ( n6 : R ) => by have n7 : ( R ∨ Q ) := Or.inl n6 ; exact n7 ) ; exact n2

-- textbook_96a56ad8e1132658e60b
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) ( h2 : ( ( Q → R ) ∧ ( R → Q ) ) ) : ( ( P → R ) ∧ ( R → P ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( ( Q → R ) ∧ ( R → Q ) ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : ( Q → R ) := n2.1 ; have n6 : ( P → Q ) := n1.1 ; have n7 : Q := n6 n4 ; have n8 : R := n5 n7 ; exact n8 ) ; have n9 : ( R → P ) := ( fun ( n10 : R ) => by have n11 : ( Q → P ) := n1.2 ; have n12 : ( R → Q ) := n2.2 ; have n13 : Q := n12 n10 ; have n14 : P := n11 n13 ; exact n14 ) ; have n15 : ( ( P → R ) ∧ ( R → P ) ) := ⟨ n3 , n9 ⟩ ; exact n15

-- textbook_17cf6bcfdfe010a1d9b2
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → Q ) ) ( h2 : ( P → R ) ) : ( Q ∨ R ) := by have n1 : ( ( ¬ P ) → Q ) := h1 ; have n2 : ( P → R ) := h2 ; have n3 : ( ¬ ( ¬ ( Q ∨ R ) ) ) := ( fun ( n4 : ( ¬ ( Q ∨ R ) ) ) => by have n5 : ( ¬ P ) := ( fun ( n6 : P ) => by have n7 : R := n2 n6 ; have n8 : ( Q ∨ R ) := Or.inr n7 ; have n9 : False := n4 n8 ; exact ( n9 : False ) ) ; have n10 : Q := n1 n5 ; have n11 : ( Q ∨ R ) := Or.inl n10 ; have n12 : False := n4 n11 ; exact ( n12 : False ) ) ; have n13 : ( Q ∨ R ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n13

-- textbook_80573e8091085933fe07
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( Q ∧ R ) ) ) ( h2 : ( Q → S ) ) : S := by have n1 : ( ( P ∧ Q ) ∨ ( Q ∧ R ) ) := h1 ; have n2 : ( Q → S ) := h2 ; have n3 : Q := Or.elim n1 ( fun ( n4 : ( P ∧ Q ) ) => by have n5 : Q := n4.2 ; exact n5 ) ( fun ( n6 : ( Q ∧ R ) ) => by have n7 : Q := n6.1 ; exact n7 ) ; have n8 : S := n2 n3 ; exact n8

-- textbook_12d1e9ba77154cecdaa4
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∨ R ) ) ) ( h2 : ( P → ( ¬ R ) ) ) : ( Q ∨ S ) := by have n1 : ( P ∧ ( Q ∨ R ) ) := h1 ; have n2 : ( P → ( ¬ R ) ) := h2 ; have n3 : ( Q ∨ R ) := n1.2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : P := n1.1 ; have n8 : ( ¬ R ) := n2 n7 ; have n9 : False := n8 n6 ; have n10 : Q := n9.elim ; exact n10 ) ; have n11 : ( Q ∨ S ) := Or.inl n4 ; exact n11

-- textbook_093d53538c2072386890
theorem t ( P Q R S : Prop ) : ( ( ( P → Q ) ∧ ( R → Q ) ) → ( ( P ∨ R ) → Q ) ) := by have n1 : ( ( ( P → Q ) ∧ ( R → Q ) ) → ( ( P ∨ R ) → Q ) ) := ( fun ( n2 : ( ( P → Q ) ∧ ( R → Q ) ) ) => by have n3 : ( ( P ∨ R ) → Q ) := ( fun ( n4 : ( P ∨ R ) ) => by have n5 : Q := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( P → Q ) := n2.1 ; have n8 : Q := n7 n6 ; exact n8 ) ( fun ( n9 : R ) => by have n10 : ( R → Q ) := n2.2 ; have n11 : Q := n10 n9 ; exact n11 ) ; exact n5 ) ; exact n3 ) ; exact n1

-- textbook_0557545ab024a85bd5ed
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( ¬ P ) ∨ Q ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ¬ ( ¬ ( ( ¬ P ) ∨ Q ) ) ) := ( fun ( n3 : ( ¬ ( ( ¬ P ) ∨ Q ) ) ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : Q := n1 n5 ; have n7 : ( ( ¬ P ) ∨ Q ) := Or.inr n6 ; have n8 : False := n3 n7 ; exact ( n8 : False ) ) ; have n9 : ( ( ¬ P ) ∨ Q ) := Or.inl n4 ; have n10 : False := n3 n9 ; exact ( n10 : False ) ) ; have n11 : ( ( ¬ P ) ∨ Q ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n11

-- textbook_26c5501ec4eb97272dc1
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) ( h2 : ( Q ∨ R ) ) ( h3 : ( ¬ Q ) ) : ( P ∧ R ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( Q ∨ R ) := h2 ; have n3 : ( ¬ Q ) := h3 ; have n4 : P := Or.elim n1 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : Q ) => by have n7 : False := n3 n6 ; have n8 : P := n7.elim ; exact n8 ) ; have n9 : R := Or.elim n2 ( fun ( n10 : Q ) => by have n11 : False := n3 n10 ; have n12 : R := n11.elim ; exact n12 ) ( fun ( n13 : R ) => by exact n13 ) ; have n14 : ( P ∧ R ) := ⟨ n4 , n9 ⟩ ; exact n14

-- textbook_545881ef2f5176d94567
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ ( Q → P ) ) ) : ( ( ¬ P ) → ( ¬ Q ) ) := by have n1 : ( P ∨ ( Q → P ) ) := h1 ; have n2 : ( ( ¬ P ) → ( ¬ Q ) ) := ( fun ( n3 : ( ¬ P ) ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : P := Or.elim n1 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( Q → P ) ) => by have n9 : P := n8 n5 ; exact n9 ) ; have n10 : False := n3 n6 ; exact ( n10 : False ) ) ; exact n4 ) ; exact n2

-- textbook_5b38b6ab0a04f4c4bb28
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∧ ( R ∨ S ) ) ) ( h2 : ( Q → S ) ) ( h3 : ( ¬ S ) ) : ( P ∧ R ) := by have n1 : ( ( P ∨ Q ) ∧ ( R ∨ S ) ) := h1 ; have n2 : ( Q → S ) := h2 ; have n3 : ( ¬ S ) := h3 ; have n4 : ( P ∨ Q ) := n1.1 ; have n5 : P := Or.elim n4 ( fun ( n6 : P ) => by exact n6 ) ( fun ( n7 : Q ) => by have n8 : S := n2 n7 ; have n9 : False := n3 n8 ; have n10 : P := n9.elim ; exact n10 ) ; have n11 : ( R ∨ S ) := n1.2 ; have n12 : R := Or.elim n11 ( fun ( n13 : R ) => by exact n13 ) ( fun ( n14 : S ) => by have n15 : False := n3 n14 ; have n16 : R := n15.elim ; exact n16 ) ; have n17 : ( P ∧ R ) := ⟨ n5 , n12 ⟩ ; exact n17

-- textbook_9f8530f273dc8191d06c
theorem t ( P Q R S : Prop ) : ( ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) ∧ ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) ) := by have n1 : ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) := ( fun ( n2 : P ) => by have n3 : ( P ∨ ( Q ∧ ( ¬ Q ) ) ) := Or.inl n2 ; exact n3 ) ; have n4 : ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) := ( fun ( n5 : ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) => by have n6 : P := Or.elim n5 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( Q ∧ ( ¬ Q ) ) ) => by have n9 : Q := n8.1 ; have n10 : ( ¬ Q ) := n8.2 ; have n11 : False := n10 n9 ; have n12 : P := n11.elim ; exact n12 ) ; exact n6 ) ; have n13 : ( ( P → ( P ∨ ( Q ∧ ( ¬ Q ) ) ) ) ∧ ( ( P ∨ ( Q ∧ ( ¬ Q ) ) ) → P ) ) := ⟨ n1 , n4 ⟩ ; exact n13

-- textbook_d0b1395237936017f5f3
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( ( ¬ Q ) → ( ¬ P ) ) ) ) : ( ( Q ∧ P ) ∨ ( ¬ P ) ) := by have n1 : ( P ∧ ( ( ¬ Q ) → ( ¬ P ) ) ) := h1 ; have n2 : P := n1.1 ; have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : ( ( ¬ Q ) → ( ¬ P ) ) := n1.2 ; have n6 : ( ¬ P ) := n5 n4 ; have n7 : False := n6 n2 ; exact ( n7 : False ) ) ; have n8 : Q := Classical.byContradiction ( fun hh => n3 hh ) ; have n9 : ( Q ∧ P ) := ⟨ n8 , n2 ⟩ ; have n10 : ( ( Q ∧ P ) ∨ ( ¬ P ) ) := Or.inl n9 ; exact n10

-- textbook_a276d846d75000fe58ae
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∨ ( R ∨ S ) ) ) ( h2 : ( Q → R ) ) ( h3 : ( ¬ S ) ) : ( P ∨ R ) := by have n1 : ( ( P ∨ Q ) ∨ ( R ∨ S ) ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( ¬ S ) := h3 ; have n4 : ( P ∨ R ) := Or.elim n1 ( fun ( n5 : ( P ∨ Q ) ) => by have n6 : ( P ∨ R ) := Or.elim n5 ( fun ( n7 : P ) => by have n8 : ( P ∨ R ) := Or.inl n7 ; exact n8 ) ( fun ( n9 : Q ) => by have n10 : R := n2 n9 ; have n11 : ( P ∨ R ) := Or.inr n10 ; exact n11 ) ; exact n6 ) ( fun ( n12 : ( R ∨ S ) ) => by have n13 : R := Or.elim n12 ( fun ( n14 : R ) => by exact n14 ) ( fun ( n15 : S ) => by have n16 : False := n3 n15 ; have n17 : R := n16.elim ; exact n17 ) ; have n18 : ( P ∨ R ) := Or.inr n13 ; exact n18 ) ; exact n4

-- textbook_0a3aa5c2d4abc9d9ba3d
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) : ( ( Q → P ) ∧ ( P → Q ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( Q → P ) := n1.2 ; have n3 : ( P → Q ) := n1.1 ; have n4 : ( ( Q → P ) ∧ ( P → Q ) ) := ⟨ n2 , n3 ⟩ ; exact n4

-- textbook_43216153b8cb762ae45f
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( ¬ ( ¬ ( ¬ P ) ) ) ) ) : P := by have n1 : ( ¬ ( ¬ ( ¬ ( ¬ P ) ) ) ) := h1 ; have n2 : ( ¬ ( ¬ P ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; have n3 : P := Classical.byContradiction ( fun hh => n2 hh ) ; exact n3

-- textbook_46701b298f4b81ffd08a
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ¬ ( P ∧ ( ¬ Q ) ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ¬ ( P ∧ ( ¬ Q ) ) ) := ( fun ( n3 : ( P ∧ ( ¬ Q ) ) ) => by have n4 : P := n3.1 ; have n5 : Q := n1 n4 ; have n6 : ( ¬ Q ) := n3.2 ; have n7 : False := n6 n5 ; exact ( n7 : False ) ) ; exact n2

-- textbook_7e11198dd38f376da50d
theorem t ( P Q R S : Prop ) ( h1 : ( ( ( ¬ P ) → Q ) ∧ ( Q → ( ¬ P ) ) ) ) : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := by have n1 : ( ( ( ¬ P ) → Q ) ∧ ( Q → ( ¬ P ) ) ) := h1 ; have n2 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := ( fun ( n3 : ( ( P → Q ) ∧ ( Q → P ) ) ) => by have n4 : ( ( ¬ P ) → Q ) := n1.1 ; have n5 : ( Q → ( ¬ P ) ) := n1.2 ; have n6 : ( Q → P ) := n3.2 ; have n7 : ( P → Q ) := n3.1 ; have n8 : ( ¬ P ) := ( fun ( n9 : P ) => by have n10 : Q := n7 n9 ; have n11 : ( ¬ P ) := n5 n10 ; have n12 : False := n11 n9 ; exact ( n12 : False ) ) ; have n13 : Q := n4 n8 ; have n14 : P := n6 n13 ; have n15 : False := n8 n14 ; exact ( n15 : False ) ) ; exact n2

-- textbook_daf2d92294493b63e627
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) : False := by have n1 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) := h1 ; have n2 : ( P → ( ¬ P ) ) := n1.1 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( ¬ P ) := n2 n4 ; have n6 : False := n5 n4 ; exact ( n6 : False ) ) ; have n7 : ( ( ¬ P ) → P ) := n1.2 ; have n8 : P := n7 n3 ; have n9 : False := n3 n8 ; exact n9

-- textbook_b225c246ef315fe80088
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ ( ¬ P ) ) → ( ¬ Q ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( ( ¬ ( ¬ P ) ) → ( ¬ Q ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : False := n5 n3 ; exact ( n6 : False ) ) ; have n7 : ( ¬ Q ) := n1 n4 ; exact n7 ) ; exact n2

-- textbook_241044592d16d6e4efed
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) ( h2 : R ) ( h3 : ( R → P ) ) : False := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : R := h2 ; have n3 : ( R → P ) := h3 ; have n4 : P := n3 n2 ; have n5 : ( P ∨ Q ) := Or.inl n4 ; have n6 : False := n1 n5 ; exact n6

-- textbook_2e44197318839943ac9c
theorem t ( P Q R S : Prop ) ( h1 : ( ( ( P → P ) ∧ ( P → P ) ) ∧ ( Q ∧ ( ¬ Q ) ) ) ) : False := by have n1 : ( ( ( P → P ) ∧ ( P → P ) ) ∧ ( Q ∧ ( ¬ Q ) ) ) := h1 ; have n2 : ( Q ∧ ( ¬ Q ) ) := n1.2 ; have n3 : Q := n2.1 ; have n4 : ( ¬ Q ) := n2.2 ; have n5 : False := n4 n3 ; exact n5

-- textbook_a96986bb1c65943f0b03
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( ¬ ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) ) ) ( h2 : ( ( Q ∨ P ) ∨ Q ) ) : ( ( Q → P ) ∧ Q ) := by have n1 : ( ¬ ( ¬ ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) ) ) := h1 ; have n2 : ( ( Q ∨ P ) ∨ Q ) := h2 ; have n3 : ( ( P → ( ¬ P ) ) ∧ ( ( ¬ P ) → P ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; have n4 : ( P → ( ¬ P ) ) := n3.1 ; have n5 : ( ¬ P ) := ( fun ( n6 : P ) => by have n7 : ( ¬ P ) := n4 n6 ; have n8 : False := n7 n6 ; exact ( n8 : False ) ) ; have n9 : ( ( ¬ P ) → P ) := n3.2 ; have n10 : P := n9 n5 ; have n11 : False := n5 n10 ; have n12 : ( ( Q → P ) ∧ Q ) := n11.elim ; exact n12

-- textbook_34ece9058da2761b0326
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∧ R ) ) ) : ( ( P ∧ Q ) ∧ R ) := by have n1 : ( P ∧ ( Q ∧ R ) ) := h1 ; have n2 : P := n1.1 ; have n3 : ( Q ∧ R ) := n1.2 ; have n4 : Q := n3.1 ; have n5 : ( P ∧ Q ) := ⟨ n2 , n4 ⟩ ; have n6 : R := n3.2 ; have n7 : ( ( P ∧ Q ) ∧ R ) := ⟨ n5 , n6 ⟩ ; exact n7

-- textbook_3cce87a8e9784b381b83
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : R ) ( h3 : ( ( Q ∧ R ) → S ) ) : S := by have n1 : ( P ∧ Q ) := h1 ; have n2 : R := h2 ; have n3 : ( ( Q ∧ R ) → S ) := h3 ; have n4 : Q := n1.2 ; have n5 : ( Q ∧ R ) := ⟨ n4 , n2 ⟩ ; have n6 : S := n3 n5 ; exact n6

-- textbook_fcf1f314b60800e5aff4
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q ∧ P ) → R ) ) : R := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q ∧ P ) → R ) := h2 ; have n3 : Q := n1.2 ; have n4 : P := n1.1 ; have n5 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; have n6 : R := n2 n5 ; exact n6

-- textbook_4cb66e0be9e0934acab6
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( Q → ( P → R ) ) ) : ( R ∧ Q ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( Q → ( P → R ) ) := h2 ; have n3 : Q := n1.2 ; have n4 : ( P → R ) := n2 n3 ; have n5 : P := n1.1 ; have n6 : R := n4 n5 ; have n7 : ( R ∧ Q ) := ⟨ n6 , n3 ⟩ ; exact n7

-- textbook_d082ad7b7083a55db8a1
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( ( Q ∧ P ) → R ) ) ) ( h2 : ( P ∧ Q ) ) : R := by have n1 : ( P → ( ( Q ∧ P ) → R ) ) := h1 ; have n2 : ( P ∧ Q ) := h2 ; have n3 : Q := n2.2 ; have n4 : P := n2.1 ; have n5 : ( ( Q ∧ P ) → R ) := n1 n4 ; have n6 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; have n7 : R := n5 n6 ; exact n7

-- textbook_10a0b9b200c3d7fdc17b
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( Q ∧ R ) ) ) ( h2 : ( P → S ) ) : ( P → ( S ∧ R ) ) := by have n1 : ( P → ( Q ∧ R ) ) := h1 ; have n2 : ( P → S ) := h2 ; have n3 : ( P → ( S ∧ R ) ) := ( fun ( n4 : P ) => by have n5 : S := n2 n4 ; have n6 : ( Q ∧ R ) := n1 n4 ; have n7 : R := n6.2 ; have n8 : ( S ∧ R ) := ⟨ n5 , n7 ⟩ ; exact n8 ) ; exact n3

-- textbook_f2569bc3cd708f5eb866
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) ( h2 : ( P → Q ) ) : ( P → R ) := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : Q := n2 n4 ; have n6 : ( P ∧ Q ) := ⟨ n4 , n5 ⟩ ; have n7 : R := n1 n6 ; exact n7 ) ; exact n3

-- textbook_cfa22f27d98eb6b067c6
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) : ( ( ( P ∧ R ) → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → ( P ∧ R ) ) ) := by have n1 : ( ( P → Q ) ∧ ( Q → P ) ) := h1 ; have n2 : ( ( P ∧ R ) → ( Q ∧ R ) ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : ( P → Q ) := n1.1 ; have n5 : P := n3.1 ; have n6 : Q := n4 n5 ; have n7 : R := n3.2 ; have n8 : ( Q ∧ R ) := ⟨ n6 , n7 ⟩ ; exact n8 ) ; have n9 : ( ( Q ∧ R ) → ( P ∧ R ) ) := ( fun ( n10 : ( Q ∧ R ) ) => by have n11 : ( Q → P ) := n1.2 ; have n12 : Q := n10.1 ; have n13 : P := n11 n12 ; have n14 : R := n10.2 ; have n15 : ( P ∧ R ) := ⟨ n13 , n14 ⟩ ; exact n15 ) ; have n16 : ( ( ( P ∧ R ) → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → ( P ∧ R ) ) ) := ⟨ n2 , n9 ⟩ ; exact n16

-- textbook_9443266b19960950ac62
theorem t ( P Q R S : Prop ) ( h1 : ( ( ( P ∧ Q ) → ( P ∧ R ) ) ∧ ( ( P ∧ R ) → ( P ∧ Q ) ) ) ) : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := by have n1 : ( ( ( P ∧ Q ) → ( P ∧ R ) ) ∧ ( ( P ∧ R ) → ( P ∧ Q ) ) ) := h1 ; have n2 : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := ( fun ( n3 : P ) => by have n4 : ( Q → R ) := ( fun ( n5 : Q ) => by have n6 : ( ( P ∧ Q ) → ( P ∧ R ) ) := n1.1 ; have n7 : ( P ∧ Q ) := ⟨ n3 , n5 ⟩ ; have n8 : ( P ∧ R ) := n6 n7 ; have n9 : R := n8.2 ; exact n9 ) ; have n10 : ( R → Q ) := ( fun ( n11 : R ) => by have n12 : ( ( P ∧ R ) → ( P ∧ Q ) ) := n1.2 ; have n13 : ( P ∧ R ) := ⟨ n3 , n11 ⟩ ; have n14 : ( P ∧ Q ) := n12 n13 ; have n15 : Q := n14.2 ; exact n15 ) ; have n16 : ( ( Q → R ) ∧ ( R → Q ) ) := ⟨ n4 , n10 ⟩ ; exact n16 ) ; exact n2

-- textbook_87a23fa584426b3d9328
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( R ∧ P ) ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( R ∧ P ) ) := h2 ; have n3 : ( Q → P ) := ( fun ( n4 : Q ) => by have n5 : ( R ∧ P ) := n2 n4 ; have n6 : P := n5.2 ; exact n6 ) ; have n7 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n1 , n3 ⟩ ; exact n7

-- textbook_36d8dc85a05b9a9d5115
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → P ) ) ) ( h2 : ( R → P ) ) : ( ( P → R ) ∧ ( R → P ) ) := by have n1 : ( ( P → ( Q ∧ R ) ) ∧ ( ( Q ∧ R ) → P ) ) := h1 ; have n2 : ( R → P ) := h2 ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : ( P → ( Q ∧ R ) ) := n1.1 ; have n6 : ( Q ∧ R ) := n5 n4 ; have n7 : R := n6.2 ; exact n7 ) ; have n8 : ( ( P → R ) ∧ ( R → P ) ) := ⟨ n3 , n2 ⟩ ; exact n8

-- textbook_2a874b10554baacdab52
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( ¬ P ) ) ) : ( ¬ P ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( ¬ P ) ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : Q := n1 n4 ; have n6 : ( ¬ P ) := n2 n5 ; have n7 : False := n6 n4 ; exact ( n7 : False ) ) ; exact n3

-- textbook_03e2f5be685c0d2ac60a
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → ( ¬ P ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( ( P ∧ Q ) → ( ¬ P ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : ( P ∧ Q ) := ⟨ n3 , n5 ⟩ ; have n7 : ( ¬ P ) := n1 n6 ; have n8 : False := n7 n3 ; exact ( n8 : False ) ) ; exact n4 ) ; exact n2

-- textbook_4814343dfe60095b930f
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ ( ¬ Q ) ) → Q ) ) : ( P → Q ) := by have n1 : ( ( P ∧ ( ¬ Q ) ) → Q ) := h1 ; have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : ( P ∧ ( ¬ Q ) ) := ⟨ n3 , n5 ⟩ ; have n7 : Q := n1 n6 ; have n8 : False := n5 n7 ; exact ( n8 : False ) ) ; have n9 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n9 ) ; exact n2

-- textbook_ed9c41eb74dfb88bfa3f
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) ) : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := by have n1 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := h1 ; have n2 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := ( fun ( n3 : ( ( P → Q ) ∧ ( Q → P ) ) ) => by have n4 : ( P → ( ¬ Q ) ) := n1.1 ; have n5 : ( Q → P ) := n3.2 ; have n6 : ( ¬ Q ) := ( fun ( n7 : Q ) => by have n8 : P := n5 n7 ; have n9 : ( ¬ Q ) := n4 n8 ; have n10 : False := n9 n7 ; exact ( n10 : False ) ) ; have n11 : ( P → Q ) := n3.1 ; have n12 : ( ( ¬ Q ) → P ) := n1.2 ; have n13 : P := n12 n6 ; have n14 : Q := n11 n13 ; have n15 : False := n6 n14 ; exact ( n15 : False ) ) ; exact n2

-- textbook_5de38c2037f06951c19e
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( P ∧ R ) ) ) : P := by have n1 : ( ( P ∧ Q ) ∨ ( P ∧ R ) ) := h1 ; have n2 : P := Or.elim n1 ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : P := n3.1 ; exact n4 ) ( fun ( n5 : ( P ∧ R ) ) => by have n6 : P := n5.1 ; exact n6 ) ; exact n2

-- textbook_681e5816d2388b53e3d9
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ ( Q ∨ R ) ) ) : ( ( P ∨ Q ) ∨ R ) := by have n1 : ( P ∨ ( Q ∨ R ) ) := h1 ; have n2 : ( ( P ∨ Q ) ∨ R ) := Or.elim n1 ( fun ( n3 : P ) => by have n4 : ( P ∨ Q ) := Or.inl n3 ; have n5 : ( ( P ∨ Q ) ∨ R ) := Or.inl n4 ; exact n5 ) ( fun ( n6 : ( Q ∨ R ) ) => by have n7 : ( ( P ∨ Q ) ∨ R ) := Or.elim n6 ( fun ( n8 : Q ) => by have n9 : ( P ∨ Q ) := Or.inr n8 ; have n10 : ( ( P ∨ Q ) ∨ R ) := Or.inl n9 ; exact n10 ) ( fun ( n11 : R ) => by have n12 : ( ( P ∨ Q ) ∨ R ) := Or.inr n11 ; exact n12 ) ; exact n7 ) ; exact n2

-- textbook_06ccd6ec0200aeb51862
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) ∨ R ) ) ( h2 : ( ( R → Q ) ∧ ( Q → R ) ) ) : ( R ∨ P ) := by have n1 : ( ( P ∨ Q ) ∨ R ) := h1 ; have n2 : ( ( R → Q ) ∧ ( Q → R ) ) := h2 ; have n3 : ( R ∨ P ) := Or.elim n1 ( fun ( n4 : ( P ∨ Q ) ) => by have n5 : ( R ∨ P ) := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( R ∨ P ) := Or.inr n6 ; exact n7 ) ( fun ( n8 : Q ) => by have n9 : ( Q → R ) := n2.2 ; have n10 : R := n9 n8 ; have n11 : ( R ∨ P ) := Or.inl n10 ; exact n11 ) ; exact n5 ) ( fun ( n12 : R ) => by have n13 : ( R ∨ P ) := Or.inl n12 ; exact n13 ) ; exact n3

-- textbook_b1f54ffd10be0d8d9b52
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( P ∨ ( R ∧ Q ) ) → Q ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( P ∨ ( R ∧ Q ) ) → Q ) := ( fun ( n3 : ( P ∨ ( R ∧ Q ) ) ) => by have n4 : Q := Or.elim n3 ( fun ( n5 : P ) => by have n6 : Q := n1 n5 ; exact n6 ) ( fun ( n7 : ( R ∧ Q ) ) => by have n8 : Q := n7.2 ; exact n8 ) ; exact n4 ) ; exact n2

-- textbook_911ca699a25f56477d3e
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R → ( R → S ) ) ) : ( ( R ∨ P ) → ( Q ∨ S ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R → ( R → S ) ) := h2 ; have n3 : ( ( R ∨ P ) → ( Q ∨ S ) ) := ( fun ( n4 : ( R ∨ P ) ) => by have n5 : ( Q ∨ S ) := Or.elim n4 ( fun ( n6 : R ) => by have n7 : ( R → S ) := n2 n6 ; have n8 : S := n7 n6 ; have n9 : ( Q ∨ S ) := Or.inr n8 ; exact n9 ) ( fun ( n10 : P ) => by have n11 : Q := n1 n10 ; have n12 : ( Q ∨ S ) := Or.inl n11 ; exact n12 ) ; exact n5 ) ; exact n3

-- textbook_5fe19ee857dfdf3b89d2
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R → ( S ∧ Q ) ) ) : ( ( P ∨ R ) → ( S ∨ Q ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R → ( S ∧ Q ) ) := h2 ; have n3 : ( ( P ∨ R ) → ( S ∨ Q ) ) := ( fun ( n4 : ( P ∨ R ) ) => by have n5 : Q := Or.elim n4 ( fun ( n6 : P ) => by have n7 : Q := n1 n6 ; exact n7 ) ( fun ( n8 : R ) => by have n9 : ( S ∧ Q ) := n2 n8 ; have n10 : Q := n9.2 ; exact n10 ) ; have n11 : ( S ∨ Q ) := Or.inr n5 ; exact n11 ) ; exact n3

-- textbook_c5140455df9355840a6c
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) ( h2 : ( Q → R ) ) : ( P ∨ R ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( P ∨ R ) := Or.elim n1 ( fun ( n4 : P ) => by have n5 : ( P ∨ R ) := Or.inl n4 ; exact n5 ) ( fun ( n6 : Q ) => by have n7 : R := n2 n6 ; have n8 : ( P ∨ R ) := Or.inr n7 ; exact n8 ) ; exact n3

-- textbook_4a8cfd28c11708238126
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) ∨ ( Q → R ) ) ) : ( ( P ∧ Q ) → R ) := by have n1 : ( ( ¬ P ) ∨ ( Q → R ) ) := h1 ; have n2 : ( ( P ∧ Q ) → R ) := ( fun ( n3 : ( P ∧ Q ) ) => by have n4 : ( Q → R ) := Or.elim n1 ( fun ( n5 : ( ¬ P ) ) => by have n6 : P := n3.1 ; have n7 : False := n5 n6 ; have n8 : ( Q → R ) := n7.elim ; exact n8 ) ( fun ( n9 : ( Q → R ) ) => by exact n9 ) ; have n10 : Q := n3.2 ; have n11 : R := n4 n10 ; exact n11 ) ; exact n2

-- textbook_0ad92464052745011778
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := ( fun ( n3 : ( ( ¬ P ) ∧ ( ¬ Q ) ) ) => by have n4 : False := Or.elim n1 ( fun ( n5 : P ) => by have n6 : ( ¬ P ) := n3.1 ; have n7 : False := n6 n5 ; exact n7 ) ( fun ( n8 : Q ) => by have n9 : ( ¬ Q ) := n3.2 ; have n10 : False := n9 n8 ; exact n10 ) ; exact ( n4 : False ) ) ; exact n2

-- textbook_a44e77e0ea54ed00a768
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) : ( P ∨ Q ) := by have n1 : ( ¬ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) := h1 ; have n2 : ( ¬ ( ¬ ( P ∨ Q ) ) ) := ( fun ( n3 : ( ¬ ( P ∨ Q ) ) ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( P ∨ Q ) := Or.inl n5 ; have n7 : False := n3 n6 ; exact ( n7 : False ) ) ; have n8 : ( ¬ Q ) := ( fun ( n9 : Q ) => by have n10 : ( P ∨ Q ) := Or.inr n9 ; have n11 : False := n3 n10 ; exact ( n11 : False ) ) ; have n12 : ( ( ¬ P ) ∧ ( ¬ Q ) ) := ⟨ n4 , n8 ⟩ ; have n13 : False := n1 n12 ; exact ( n13 : False ) ) ; have n14 : ( P ∨ Q ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n14

-- textbook_b7132b7858d742201ec0
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) ) : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := by have n1 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) := h1 ; have n2 : ( P → ( ¬ Q ) ) := ( fun ( n3 : P ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : ( P → Q ) := ( fun ( n7 : P ) => by have n8 : Q := n5 ; exact n8 ) ; have n9 : ( Q → P ) := ( fun ( n10 : Q ) => by have n11 : P := n3 ; exact n11 ) ; have n12 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n6 , n9 ⟩ ; have n13 : False := n1 n12 ; exact ( n13 : False ) ) ; exact n4 ) ; have n14 : ( ( ¬ Q ) → P ) := ( fun ( n15 : ( ¬ Q ) ) => by have n16 : ( ¬ ( ¬ P ) ) := ( fun ( n17 : ( ¬ P ) ) => by have n18 : ( P → Q ) := ( fun ( n19 : P ) => by have n20 : False := n17 n19 ; have n21 : Q := n20.elim ; exact n21 ) ; have n22 : ( Q → P ) := ( fun ( n23 : Q ) => by have n24 : False := n15 n23 ; have n25 : P := n24.elim ; exact n25 ) ; have n26 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n18 , n22 ⟩ ; have n27 : False := n1 n26 ; exact ( n27 : False ) ) ; have n28 : P := Classical.byContradiction ( fun hh => n16 hh ) ; exact n28 ) ; have n29 : ( ( P → ( ¬ Q ) ) ∧ ( ( ¬ Q ) → P ) ) := ⟨ n2 , n14 ⟩ ; exact n29

-- textbook_de9fce762bb7ce5e677f
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ Q ) ) : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := by have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ¬ ( ¬ ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) ) := ( fun ( n3 : ( ¬ ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) ) => by have n4 : P := Or.elim n1 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : Q ) => by have n7 : ( ¬ ( ¬ P ) ) := ( fun ( n8 : ( ¬ P ) ) => by have n9 : ( ( ¬ P ) ∧ Q ) := ⟨ n8 , n6 ⟩ ; have n10 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Or.inr n9 ; have n11 : False := n3 n10 ; exact ( n11 : False ) ) ; have n12 : P := Classical.byContradiction ( fun hh => n7 hh ) ; exact n12 ) ; have n13 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Or.inl n4 ; have n14 : False := n3 n13 ; exact ( n14 : False ) ) ; have n15 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := Classical.byContradiction ( fun hh => n2 hh ) ; exact n15

-- textbook_9b042a4a7fa74ab578cb
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) ) : ( P ∨ Q ) := by have n1 : ( P ∨ ( ( ¬ P ) ∧ Q ) ) := h1 ; have n2 : ( P ∨ Q ) := Or.elim n1 ( fun ( n3 : P ) => by have n4 : ( P ∨ Q ) := Or.inl n3 ; exact n4 ) ( fun ( n5 : ( ( ¬ P ) ∧ Q ) ) => by have n6 : Q := n5.2 ; have n7 : ( P ∨ Q ) := Or.inr n6 ; exact n7 ) ; exact n2

-- textbook_4447e76b45cccb03d640
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) → R ) ) ( h2 : ( ¬ R ) ) : ( ¬ P ) := by have n1 : ( ( P ∨ Q ) → R ) := h1 ; have n2 : ( ¬ R ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( P ∨ Q ) := Or.inl n4 ; have n6 : R := n1 n5 ; have n7 : False := n2 n6 ; exact ( n7 : False ) ) ; exact n3

-- textbook_1b0c86b846be078dbfc8
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( P → ( Q ∨ ( ¬ R ) ) ) ) ) : ( ( Q ∨ R ) → P ) := by have n1 : ( ¬ ( P → ( Q ∨ ( ¬ R ) ) ) ) := h1 ; have n2 : ( ( Q ∨ R ) → P ) := ( fun ( n3 : ( Q ∨ R ) ) => by have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : ( P → ( Q ∨ ( ¬ R ) ) ) := ( fun ( n7 : P ) => by have n8 : False := n5 n7 ; have n9 : ( Q ∨ ( ¬ R ) ) := n8.elim ; exact n9 ) ; have n10 : False := n1 n6 ; exact ( n10 : False ) ) ; have n11 : P := Classical.byContradiction ( fun hh => n4 hh ) ; exact n11 ) ; exact n2

-- textbook_17db74b2c27db844c116
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) ∨ ( ¬ ( R → ( ¬ S ) ) ) ) ) ( h2 : ( ¬ Q ) ) : ( R ∧ S ) := by have n1 : ( ( P ∧ Q ) ∨ ( ¬ ( R → ( ¬ S ) ) ) ) := h1 ; have n2 : ( ¬ Q ) := h2 ; have n3 : ( ¬ ( ¬ ( R ∧ S ) ) ) := ( fun ( n4 : ( ¬ ( R ∧ S ) ) ) => by have n5 : False := Or.elim n1 ( fun ( n6 : ( P ∧ Q ) ) => by have n7 : Q := n6.2 ; have n8 : False := n2 n7 ; exact n8 ) ( fun ( n9 : ( ¬ ( R → ( ¬ S ) ) ) ) => by have n10 : ( R → ( ¬ S ) ) := ( fun ( n11 : R ) => by have n12 : ( ¬ S ) := ( fun ( n13 : S ) => by have n14 : ( R ∧ S ) := ⟨ n11 , n13 ⟩ ; have n15 : False := n4 n14 ; exact ( n15 : False ) ) ; exact n12 ) ; have n16 : False := n9 n10 ; exact n16 ) ; exact ( n5 : False ) ) ; have n17 : ( R ∧ S ) := Classical.byContradiction ( fun hh => n3 hh ) ; exact n17

-- textbook_bf61a48e793cb331db6e
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → ( Q ∨ P ) ) ) ( h2 : ( ¬ P ) ) : Q := by have n1 : ( ( ¬ P ) → ( Q ∨ P ) ) := h1 ; have n2 : ( ¬ P ) := h2 ; have n3 : ( Q ∨ P ) := n1 n2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : P ) => by have n7 : False := n2 n6 ; have n8 : Q := n7.elim ; exact n8 ) ; exact n4

-- textbook_9010f2a1ef772ceb138c
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( Q ∨ ( ¬ P ) ) ) ) ( h2 : P ) : Q := by have n1 : ( P → ( Q ∨ ( ¬ P ) ) ) := h1 ; have n2 : P := h2 ; have n3 : ( Q ∨ ( ¬ P ) ) := n1 n2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : ( ¬ P ) ) => by have n7 : False := n6 n2 ; have n8 : Q := n7.elim ; exact n8 ) ; exact n4

-- textbook_2af1c15bbd58c631b0db
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q → R ) ∧ ( R → Q ) ) ) : ( R ∨ S ) := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q → R ) ∧ ( R → Q ) ) := h2 ; have n3 : ( Q → R ) := n2.1 ; have n4 : Q := n1.2 ; have n5 : R := n3 n4 ; have n6 : ( R ∨ S ) := Or.inl n5 ; exact n6

-- textbook_9b3b736284930ddf1e19
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( ( Q ∨ R ) → S ) ) : S := by have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( Q ∨ R ) → S ) := h2 ; have n3 : Q := n1.2 ; have n4 : ( Q ∨ R ) := Or.inl n3 ; have n5 : S := n2 n4 ; exact n5

-- textbook_1b669a5753a97113d412
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∨ R ) ) ) ( h2 : ( P → ( ¬ R ) ) ) : Q := by have n1 : ( P ∧ ( Q ∨ R ) ) := h1 ; have n2 : ( P → ( ¬ R ) ) := h2 ; have n3 : ( Q ∨ R ) := n1.2 ; have n4 : Q := Or.elim n3 ( fun ( n5 : Q ) => by exact n5 ) ( fun ( n6 : R ) => by have n7 : P := n1.1 ; have n8 : ( ¬ R ) := n2 n7 ; have n9 : False := n8 n6 ; have n10 : Q := n9.elim ; exact n10 ) ; exact n4

-- textbook_6a87e09cff5017971e42
theorem t ( P Q R S : Prop ) ( h1 : ( ( ( P ∨ ( ¬ Q ) ) → R ) ∧ ( R → ( P ∨ ( ¬ Q ) ) ) ) ) ( h2 : ( R ∧ Q ) ) : ( P ∧ R ) := by have n1 : ( ( ( P ∨ ( ¬ Q ) ) → R ) ∧ ( R → ( P ∨ ( ¬ Q ) ) ) ) := h1 ; have n2 : ( R ∧ Q ) := h2 ; have n3 : R := n2.1 ; have n4 : ( R → ( P ∨ ( ¬ Q ) ) ) := n1.2 ; have n5 : ( P ∨ ( ¬ Q ) ) := n4 n3 ; have n6 : P := Or.elim n5 ( fun ( n7 : P ) => by exact n7 ) ( fun ( n8 : ( ¬ Q ) ) => by have n9 : Q := n2.2 ; have n10 : False := n8 n9 ; have n11 : P := n10.elim ; exact n11 ) ; have n12 : ( P ∧ R ) := ⟨ n6 , n3 ⟩ ; exact n12

-- textbook_6fafb78b1ceff04e0dee
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) ( h2 : ( P ∨ ( ¬ S ) ) ) ( h3 : ( S ∧ Q ) ) : R := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( P ∨ ( ¬ S ) ) := h2 ; have n3 : ( S ∧ Q ) := h3 ; have n4 : P := Or.elim n2 ( fun ( n5 : P ) => by exact n5 ) ( fun ( n6 : ( ¬ S ) ) => by have n7 : S := n3.1 ; have n8 : False := n6 n7 ; have n9 : P := n8.elim ; exact n9 ) ; have n10 : Q := n3.2 ; have n11 : ( P ∧ Q ) := ⟨ n4 , n10 ⟩ ; have n12 : R := n1 n11 ; exact n12

-- textbook_c1d05f6df5cf79a1edb7
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( R ∧ ( ¬ S ) ) ) ( h3 : ( ( ( ¬ S ) → ( Q → P ) ) ∧ ( ( Q → P ) → ( ¬ S ) ) ) ) : ( ( P → Q ) ∧ ( Q → P ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( R ∧ ( ¬ S ) ) := h2 ; have n3 : ( ( ( ¬ S ) → ( Q → P ) ) ∧ ( ( Q → P ) → ( ¬ S ) ) ) := h3 ; have n4 : ( ( ¬ S ) → ( Q → P ) ) := n3.1 ; have n5 : ( ¬ S ) := n2.2 ; have n6 : ( Q → P ) := n4 n5 ; have n7 : ( ( P → Q ) ∧ ( Q → P ) ) := ⟨ n1 , n6 ⟩ ; exact n7

-- textbook_f84555d8d61f7d647653
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∨ Q ) → R ) ) ( h2 : ( ( S → R ) ∧ ( R → S ) ) ) ( h3 : P ) : ( P ∧ S ) := by have n1 : ( ( P ∨ Q ) → R ) := h1 ; have n2 : ( ( S → R ) ∧ ( R → S ) ) := h2 ; have n3 : P := h3 ; have n4 : ( R → S ) := n2.2 ; have n5 : ( P ∨ Q ) := Or.inl n3 ; have n6 : R := n1 n5 ; have n7 : S := n4 n6 ; have n8 : ( P ∧ S ) := ⟨ n3 , n7 ⟩ ; exact n8

-- textbook_e2c593f0543fc6234b2d
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → P ) ) ( h3 : ( ( Q → ( ( Q → P ) ∧ ( P → Q ) ) ) ∧ ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) ) ) : P := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → P ) := h2 ; have n3 : ( ( Q → ( ( Q → P ) ∧ ( P → Q ) ) ) ∧ ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) ) := h3 ; have n4 : ( ( ( Q → P ) ∧ ( P → Q ) ) → Q ) := n3.2 ; have n5 : ( ( Q → P ) ∧ ( P → Q ) ) := ⟨ n2 , n1 ⟩ ; have n6 : Q := n4 n5 ; have n7 : P := n2 n6 ; exact n7

-- textbook_e5cfe82b533b6643a29c
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) : ( ( P ∧ R ) → ( Q ∨ S ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( P ∧ R ) → ( Q ∨ S ) ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : P := n3.1 ; have n5 : Q := n1 n4 ; have n6 : ( Q ∨ S ) := Or.inl n5 ; exact n6 ) ; exact n2

-- textbook_cd721dbd56f203710bb0
theorem t ( P Q R S : Prop ) : ( ¬ ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) := by have n1 : ( ¬ ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) := ( fun ( n2 : ( ( P ∨ Q ) ∧ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) => by have n3 : ( ( ¬ P ) ∧ ( ¬ Q ) ) := n2.2 ; have n4 : ( P ∨ Q ) := n2.1 ; have n5 : False := Or.elim n4 ( fun ( n6 : P ) => by have n7 : ( ¬ P ) := n3.1 ; have n8 : False := n7 n6 ; exact n8 ) ( fun ( n9 : Q ) => by have n10 : ( ¬ Q ) := n3.2 ; have n11 : False := n10 n9 ; exact n11 ) ; exact ( n5 : False ) ) ; exact n1

-- textbook_9d21fcccbcd6c3ebf207
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) : ( R → ( ¬ P ) ) := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : ( R → ( ¬ P ) ) := ( fun ( n3 : R ) => by have n4 : ( ¬ P ) := ( fun ( n5 : P ) => by have n6 : ( P ∨ Q ) := Or.inl n5 ; have n7 : False := n1 n6 ; exact ( n7 : False ) ) ; exact n4 ) ; exact n2

-- textbook_bfb605b094950125182a
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ ( P ∨ Q ) ) ) : ( Q → P ) := by have n1 : ( ¬ ( P ∨ Q ) ) := h1 ; have n2 : ( Q → P ) := ( fun ( n3 : Q ) => by have n4 : ( P ∨ Q ) := Or.inr n3 ; have n5 : False := n1 n4 ; have n6 : P := n5.elim ; exact n6 ) ; exact n2

-- textbook_5c8618f27cf57fd6c99b
theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ ( Q ∧ R ) ) ) : ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) := by have n1 : ( P ∧ ( Q ∧ R ) ) := h1 ; have n2 : ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) := ( fun ( n3 : P ) => by have n4 : ( Q ∧ R ) := n1.2 ; have n5 : ( Q → R ) := ( fun ( n6 : Q ) => by have n7 : R := n4.2 ; exact n7 ) ; have n8 : ( R → Q ) := ( fun ( n9 : R ) => by have n10 : Q := n4.1 ; exact n10 ) ; have n11 : ( ( Q → R ) ∧ ( R → Q ) ) := ⟨ n5 , n8 ⟩ ; exact n11 ) ; have n12 : ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) := ( fun ( n13 : ( ( Q → R ) ∧ ( R → Q ) ) ) => by have n14 : P := n1.1 ; exact n14 ) ; have n15 : ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) := ⟨ n2 , n12 ⟩ ; exact n15

-- textbook_5f07170d9486281717b7
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ P ) ) ( h2 : ( ( ¬ Q ) → P ) ) : Q := by have n1 : ( ¬ P ) := h1 ; have n2 : ( ( ¬ Q ) → P ) := h2 ; have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by have n5 : P := n2 n4 ; have n6 : False := n1 n5 ; exact ( n6 : False ) ) ; have n7 : Q := Classical.byContradiction ( fun hh => n3 hh ) ; exact n7

-- textbook_64e1ce87d7887b0f482e
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( ¬ Q ) ) ) ( h2 : ( R → Q ) ) : ( P → ( ¬ R ) ) := by have n1 : ( P → ( ¬ Q ) ) := h1 ; have n2 : ( R → Q ) := h2 ; have n3 : ( P → ( ¬ R ) ) := ( fun ( n4 : P ) => by have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : Q := n2 n6 ; have n8 : ( ¬ Q ) := n1 n4 ; have n9 : False := n8 n7 ; exact ( n9 : False ) ) ; exact n5 ) ; exact n3

-- textbook_70f616e852eadc25705c
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → Q ) → R ) ) ( h2 : Q ) : R := by have n1 : ( ( P → Q ) → R ) := h1 ; have n2 : Q := h2 ; have n3 : ( P → Q ) := ( fun ( n4 : P ) => by have n5 : Q := n2 ; exact n5 ) ; have n6 : R := n1 n3 ; exact n6

-- textbook_3512647041a54fd55274
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → Q ) ) ( h2 : ( R → ( ¬ Q ) ) ) ( h3 : ( ( ¬ R ) → P ) ) : ( ¬ ( ¬ P ) ) := by have n1 : ( ( ¬ P ) → Q ) := h1 ; have n2 : ( R → ( ¬ Q ) ) := h2 ; have n3 : ( ( ¬ R ) → P ) := h3 ; have n4 : ( ¬ ( ¬ P ) ) := ( fun ( n5 : ( ¬ P ) ) => by have n6 : ( ¬ R ) := ( fun ( n7 : R ) => by have n8 : Q := n1 n5 ; have n9 : ( ¬ Q ) := n2 n7 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; have n11 : P := n3 n6 ; have n12 : False := n5 n11 ; exact ( n12 : False ) ) ; exact n4

-- textbook_7726f61a9bc2c7c9f1b4
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → R ) ) ( h3 : ( R → S ) ) ( h4 : P ) : ( ¬ ( P → ( ¬ S ) ) ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : ( R → S ) := h3 ; have n4 : P := h4 ; have n5 : ( ¬ ( P → ( ¬ S ) ) ) := ( fun ( n6 : ( P → ( ¬ S ) ) ) => by have n7 : Q := n1 n4 ; have n8 : R := n2 n7 ; have n9 : S := n3 n8 ; have n10 : ( ¬ S ) := n6 n4 ; have n11 : False := n10 n9 ; exact ( n11 : False ) ) ; exact n5

-- textbook_cd69a60630955cea9a6d
theorem t ( P Q R S : Prop ) ( h1 : ( ( P → Q ) → R ) ) ( h2 : ( S → ( P → Q ) ) ) ( h3 : ( ( ¬ S ) → R ) ) : ( ¬ ( ¬ R ) ) := by have n1 : ( ( P → Q ) → R ) := h1 ; have n2 : ( S → ( P → Q ) ) := h2 ; have n3 : ( ( ¬ S ) → R ) := h3 ; have n4 : ( ¬ ( ¬ R ) ) := ( fun ( n5 : ( ¬ R ) ) => by have n6 : ( ¬ S ) := ( fun ( n7 : S ) => by have n8 : ( P → Q ) := n2 n7 ; have n9 : R := n1 n8 ; have n10 : False := n5 n9 ; exact ( n10 : False ) ) ; have n11 : R := n3 n6 ; have n12 : False := n5 n11 ; exact ( n12 : False ) ) ; exact n4

-- textbook_d9dc2ee1269d002c72d2
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( Q → R ) ) ) ( h2 : ( Q → ( ¬ R ) ) ) : ( P → ( ¬ Q ) ) := by have n1 : ( P → ( Q → R ) ) := h1 ; have n2 : ( Q → ( ¬ R ) ) := h2 ; have n3 : ( P → ( ¬ Q ) ) := ( fun ( n4 : P ) => by have n5 : ( ¬ Q ) := ( fun ( n6 : Q ) => by have n7 : ( Q → R ) := n1 n4 ; have n8 : R := n7 n6 ; have n9 : ( ¬ R ) := n2 n6 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; exact n5 ) ; exact n3

-- textbook_d66e5d2ec8cab38d5b68
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( ( R → Q ) → ( ¬ P ) ) ) : ( ¬ P ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( ( R → Q ) → ( ¬ P ) ) := h2 ; have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( R → Q ) := ( fun ( n6 : R ) => by have n7 : Q := n1 n4 ; exact n7 ) ; have n8 : ( ¬ P ) := n2 n5 ; have n9 : False := n8 n4 ; exact ( n9 : False ) ) ; exact n3

-- textbook_0ab235a31e23b0bbbc17
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( P → Q ) ) ) ( h2 : ( ( P → Q ) → Q ) ) : Q := by have n1 : ( P → ( P → Q ) ) := h1 ; have n2 : ( ( P → Q ) → Q ) := h2 ; have n3 : ( P → Q ) := ( fun ( n4 : P ) => by have n5 : ( P → Q ) := n1 n4 ; have n6 : Q := n2 n5 ; exact n6 ) ; have n7 : Q := n2 n3 ; exact n7

-- textbook_43e30ad215c242c5605c
theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → ( ¬ R ) ) ) ( h3 : ( S → P ) ) ( h4 : ( R → S ) ) : ( ¬ R ) := by have n1 : ( P → Q ) := h1 ; have n2 : ( Q → ( ¬ R ) ) := h2 ; have n3 : ( S → P ) := h3 ; have n4 : ( R → S ) := h4 ; have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : S := n4 n6 ; have n8 : P := n3 n7 ; have n9 : Q := n1 n8 ; have n10 : ( ¬ R ) := n2 n9 ; have n11 : False := n10 n6 ; exact ( n11 : False ) ) ; exact n5

-- textbook_49b7d961bb8d378750b8
theorem t ( P Q R S : Prop ) ( h1 : ( ( ¬ P ) → ( ¬ ( ¬ Q ) ) ) ) ( h2 : ( ( ¬ ( ¬ R ) ) → ( ¬ P ) ) ) ( h3 : ( ( ¬ R ) → Q ) ) : Q := by have n1 : ( ( ¬ P ) → ( ¬ ( ¬ Q ) ) ) := h1 ; have n2 : ( ( ¬ ( ¬ R ) ) → ( ¬ P ) ) := h2 ; have n3 : ( ( ¬ R ) → Q ) := h3 ; have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : ( ¬ ( ¬ R ) ) := ( fun ( n7 : ( ¬ R ) ) => by have n8 : Q := n3 n7 ; have n9 : False := n5 n8 ; exact ( n9 : False ) ) ; have n10 : ( ¬ P ) := n2 n6 ; have n11 : ( ¬ ( ¬ Q ) ) := n1 n10 ; have n12 : False := n11 n5 ; exact ( n12 : False ) ) ; have n13 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n13

-- textbook_f0b811e0abb0c54501ad
theorem t ( P Q R S : Prop ) ( h1 : ( P → ( ¬ P ) ) ) ( h2 : ( Q → P ) ) ( h3 : ( R → Q ) ) ( h4 : ( ( ¬ R ) → Q ) ) : S := by have n1 : ( P → ( ¬ P ) ) := h1 ; have n2 : ( Q → P ) := h2 ; have n3 : ( R → Q ) := h3 ; have n4 : ( ( ¬ R ) → Q ) := h4 ; have n5 : ( ¬ R ) := ( fun ( n6 : R ) => by have n7 : Q := n3 n6 ; have n8 : P := n2 n7 ; have n9 : ( ¬ P ) := n1 n8 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; have n11 : Q := n4 n5 ; have n12 : P := n2 n11 ; have n13 : ( ¬ P ) := n1 n12 ; have n14 : False := n13 n12 ; have n15 : S := n14.elim ; exact n15

-- textbook_9f32715a4fdbe0ab0197
theorem t ( P Q R S : Prop ) ( h1 : ( ¬ P ) ) : ( ¬ ( Q ∧ P ) ) := by have n1 : ( ¬ P ) := h1 ; have n2 : ( ¬ ( Q ∧ P ) ) := ( fun ( n3 : ( Q ∧ P ) ) => by have n4 : P := n3.2 ; have n5 : False := n1 n4 ; exact ( n5 : False ) ) ; exact n2

-- textbook_0961c3d8c294d4ccb124
theorem t ( P Q R S : Prop ) : ( ( ( P ∧ Q ) → ( Q ∧ P ) ) ∧ ( ( Q ∧ P ) → ( P ∧ Q ) ) ) := by have n1 : ( ( P ∧ Q ) → ( Q ∧ P ) ) := ( fun ( n2 : ( P ∧ Q ) ) => by have n3 : Q := n2.2 ; have n4 : P := n2.1 ; have n5 : ( Q ∧ P ) := ⟨ n3 , n4 ⟩ ; exact n5 ) ; have n6 : ( ( Q ∧ P ) → ( P ∧ Q ) ) := ( fun ( n7 : ( Q ∧ P ) ) => by have n8 : P := n7.2 ; have n9 : Q := n7.1 ; have n10 : ( P ∧ Q ) := ⟨ n8 , n9 ⟩ ; exact n10 ) ; have n11 : ( ( ( P ∧ Q ) → ( Q ∧ P ) ) ∧ ( ( Q ∧ P ) → ( P ∧ Q ) ) ) := ⟨ n1 , n6 ⟩ ; exact n11

-- textbook_dd8c336103f01138f76d
theorem t ( P Q R S : Prop ) ( h1 : ( P ∨ ( ¬ Q ) ) ) ( h2 : ( Q ∨ R ) ) ( h3 : ( ¬ P ) ) : R := by have n1 : ( P ∨ ( ¬ Q ) ) := h1 ; have n2 : ( Q ∨ R ) := h2 ; have n3 : ( ¬ P ) := h3 ; have n4 : R := Or.elim n2 ( fun ( n5 : Q ) => by have n6 : False := Or.elim n1 ( fun ( n7 : P ) => by have n8 : False := n3 n7 ; exact n8 ) ( fun ( n9 : ( ¬ Q ) ) => by have n10 : False := n9 n5 ; exact n10 ) ; have n11 : R := n6.elim ; exact n11 ) ( fun ( n12 : R ) => by exact n12 ) ; exact n4

-- textbook_87d996531769036ca6a8
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ Q ) → R ) ) : ( ( P ∧ ( ¬ R ) ) → ( ¬ Q ) ) := by have n1 : ( ( P ∧ Q ) → R ) := h1 ; have n2 : ( ( P ∧ ( ¬ R ) ) → ( ¬ Q ) ) := ( fun ( n3 : ( P ∧ ( ¬ R ) ) ) => by have n4 : ( ¬ Q ) := ( fun ( n5 : Q ) => by have n6 : P := n3.1 ; have n7 : ( P ∧ Q ) := ⟨ n6 , n5 ⟩ ; have n8 : R := n1 n7 ; have n9 : ( ¬ R ) := n3.2 ; have n10 : False := n9 n8 ; exact ( n10 : False ) ) ; exact n4 ) ; exact n2

-- textbook_921620f6be6e3b10986f
theorem t ( P Q R S : Prop ) ( h1 : ( ( P ∧ ( ¬ Q ) ) → ( ¬ R ) ) ) : ( ( P ∧ R ) → Q ) := by have n1 : ( ( P ∧ ( ¬ Q ) ) → ( ¬ R ) ) := h1 ; have n2 : ( ( P ∧ R ) → Q ) := ( fun ( n3 : ( P ∧ R ) ) => by have n4 : ( ¬ ( ¬ Q ) ) := ( fun ( n5 : ( ¬ Q ) ) => by have n6 : R := n3.2 ; have n7 : P := n3.1 ; have n8 : ( P ∧ ( ¬ Q ) ) := ⟨ n7 , n5 ⟩ ; have n9 : ( ¬ R ) := n1 n8 ; have n10 : False := n9 n6 ; exact ( n10 : False ) ) ; have n11 : Q := Classical.byContradiction ( fun hh => n4 hh ) ; exact n11 ) ; exact n2

