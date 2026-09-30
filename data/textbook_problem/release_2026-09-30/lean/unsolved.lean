-- unsolved: 10 problems, rendered by the lean_seq tokenizer pinned in code/textbook_problem/lean_tools/.
-- One theorem per problem; the ND reference proof is rendered as tactic text where one exists, otherwise `sorry`.
-- Checked with Lean 4.34.1 on 2026-09-30: no errors; the unsolved file elaborates with `sorry`.

-- textbook_7a4714d44497ce53e5f5 | roy T6.29 | ⊢ (A ↔ B) ↔ [(A ∧ B) ∨ (¬A ∧ ¬B)]
theorem textbook_7a4714d44497ce53e5f5 ( P Q R S : Prop ) : ( ( ( ( P → Q ) ∧ ( Q → P ) ) → ( ( P ∧ Q ) ∨ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) ) ∧ ( ( ( P ∧ Q ) ∨ ( ( ¬ P ) ∧ ( ¬ Q ) ) ) → ( ( P → Q ) ∧ ( Q → P ) ) ) ) := by sorry

-- textbook_a38c8d47241d7ffa502f | roy T6.30 | ⊢ [A ↔ (B ↔ C)] ↔ [(A ↔ B) ↔ C]
theorem textbook_a38c8d47241d7ffa502f ( P Q R S : Prop ) : ( ( ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) → ( ( ( ( P → Q ) ∧ ( Q → P ) ) → R ) ∧ ( R → ( ( P → Q ) ∧ ( Q → P ) ) ) ) ) ∧ ( ( ( ( ( P → Q ) ∧ ( Q → P ) ) → R ) ∧ ( R → ( ( P → Q ) ∧ ( Q → P ) ) ) ) → ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) ) ) := by sorry

-- textbook_9111869d16ada6fce314 | roy E6.18i | A ↔ B ⊢ (C ↔ A) ↔ (C ↔ B)
theorem textbook_9111869d16ada6fce314 ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) : ( ( ( ( R → P ) ∧ ( P → R ) ) → ( ( R → Q ) ∧ ( Q → R ) ) ) ∧ ( ( ( R → Q ) ∧ ( Q → R ) ) → ( ( R → P ) ∧ ( P → R ) ) ) ) := by sorry

-- textbook_76f8dfbeb57747b32fae | roy E6.18o | A ↔ B, B ↔ ¬C ⊢ ¬(A ↔ C )
theorem textbook_76f8dfbeb57747b32fae ( P Q R S : Prop ) ( h1 : ( ( P → Q ) ∧ ( Q → P ) ) ) ( h2 : ( ( Q → ( ¬ R ) ) ∧ ( ( ¬ R ) → Q ) ) ) : ( ¬ ( ( P → R ) ∧ ( R → P ) ) ) := by sorry

-- textbook_c9433284b21bf158bf1d | roy E6.20c | (A ↔ B) ↔ (C ↔ D) ⊢ (A ↔ C ) → (B → D)
theorem textbook_c9433284b21bf158bf1d ( P Q R S : Prop ) ( h1 : ( ( ( ( P → Q ) ∧ ( Q → P ) ) → ( ( R → S ) ∧ ( S → R ) ) ) ∧ ( ( ( R → S ) ∧ ( S → R ) ) → ( ( P → Q ) ∧ ( Q → P ) ) ) ) ) : ( ( ( P → R ) ∧ ( R → P ) ) → ( Q → S ) ) := by sorry

-- textbook_6dd339f1fd99d36082f5 | roy E6.20d | ¬(A ↔ B), ¬(B ↔ C ), ¬(C ↔ A) ⊢ ¬K
theorem textbook_6dd339f1fd99d36082f5 ( P Q R S : Prop ) ( h1 : ( ¬ ( ( P → Q ) ∧ ( Q → P ) ) ) ) ( h2 : ( ¬ ( ( Q → R ) ∧ ( R → Q ) ) ) ) ( h3 : ( ¬ ( ( R → P ) ∧ ( P → R ) ) ) ) : ( ¬ S ) := by sorry

-- textbook_4344185c4f3bde6f4a37 | roy E6.20e | A ↔ (B ↔ C ) ⊢ (A ↔ B) ↔ C
theorem textbook_4344185c4f3bde6f4a37 ( P Q R S : Prop ) ( h1 : ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ∧ ( ( ( Q → R ) ∧ ( R → Q ) ) → P ) ) ) : ( ( ( ( P → Q ) ∧ ( Q → P ) ) → R ) ∧ ( R → ( ( P → Q ) ∧ ( Q → P ) ) ) ) := by sorry

-- textbook_931616bf029c05b823e1 | roy E6.21j | ⊢ [A → (B ↔ C )] ↔ (A → [(¬B ∨ C ) ∧ (¬C ∨ B)])
theorem textbook_931616bf029c05b823e1 ( P Q R S : Prop ) : ( ( ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) → ( P → ( ( ( ¬ Q ) ∨ R ) ∧ ( ( ¬ R ) ∨ Q ) ) ) ) ∧ ( ( P → ( ( ( ¬ Q ) ∨ R ) ∧ ( ( ¬ R ) ∨ Q ) ) ) → ( P → ( ( Q → R ) ∧ ( R → Q ) ) ) ) ) := by sorry

-- textbook_575abd80085c0295919c | roy E6.21l | ⊢ [¬A → (¬B → C )] → [(A ∨ B) ∨ (¬¬B ∨ C )]
theorem textbook_575abd80085c0295919c ( P Q R S : Prop ) : ( ( ( ¬ P ) → ( ( ¬ Q ) → R ) ) → ( ( P ∨ Q ) ∨ ( ( ¬ ( ¬ Q ) ) ∨ R ) ) ) := by sorry

-- textbook_95a721ce5c3b9b58ab99 | roy E6.21m | ⊢ (¬A ↔ ¬A) ↔ [¬(¬A → A) ↔ (A → ¬A)]
theorem textbook_95a721ce5c3b9b58ab99 ( P Q R S : Prop ) : ( ( ( ( ( ¬ P ) → ( ¬ P ) ) ∧ ( ( ¬ P ) → ( ¬ P ) ) ) → ( ( ( ¬ ( ( ¬ P ) → P ) ) → ( P → ( ¬ P ) ) ) ∧ ( ( P → ( ¬ P ) ) → ( ¬ ( ( ¬ P ) → P ) ) ) ) ) ∧ ( ( ( ( ¬ ( ( ¬ P ) → P ) ) → ( P → ( ¬ P ) ) ) ∧ ( ( P → ( ¬ P ) ) → ( ¬ ( ( ¬ P ) → P ) ) ) ) → ( ( ( ¬ P ) → ( ¬ P ) ) ∧ ( ( ¬ P ) → ( ¬ P ) ) ) ) ) := by sorry

