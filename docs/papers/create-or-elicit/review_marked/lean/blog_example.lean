theorem t (P R S : Prop) (h1 : ¬S → ¬P) (h2 : P) : R → S := by
  have n9 : R → S := (fun (n3 : R) => by
    have n7 : ¬¬S := (fun (n4 : ¬S) => by      -- assume ¬S ...
      have n5 : ¬P := h1 n4
      exact n5 h2)                              -- ... contradiction, so ¬¬S
    exact Classical.byContradiction (fun hh => n7 hh))   -- drop the double negation: σ
  exact n9

#print axioms t
