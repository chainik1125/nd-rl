set_option linter.unusedVariables false
-- ============ 1. Hilbert system / combinator basis: axioms are prelude constants, modus ponens is application ============
theorem axB {a b c : Prop} : (b → c) → (a → b) → (a → c) := fun g f x => g (f x)
theorem axC {a b c : Prop} : (a → b → c) → (b → a → c) := fun f y x => f x y
theorem axK {a b : Prop} : a → b → a := fun x _ => x
theorem axW {a b : Prop} : (a → a → b) → (a → b) := fun f x => f x x
theorem axI {a : Prop} : a → a := fun x => x
-- one line, same surface grammar as the group's format: `have n : F := <axiom | n n> ;`
theorem h1 ( P Q R S T U W X Y Z : Prop ) : ( ( P → ( P → Q ) ) → ( P → Q ) ) := by have n1 : ( ( P → ( P → Q ) ) → ( P → Q ) ) := axW ; exact n1
-- I from C and K:  C K K
theorem h2 ( P Q R S T U W X Y Z : Prop ) : ( P → P ) := by have n1 : ( ( P → ( ( P → ( Q → P ) ) → P ) ) → ( ( P → ( Q → P ) ) → ( P → P ) ) ) := axC ; have n2 : ( P → ( ( P → ( Q → P ) ) → P ) ) := axK ; have n3 : ( ( P → ( Q → P ) ) → ( P → P ) ) := n1 n2 ; have n4 : ( P → ( Q → P ) ) := axK ; have n5 : ( P → P ) := n3 n4 ; exact n5
-- with premises (a sequent), as in the current corpus
theorem h3 ( P Q R S T U W X Y Z : Prop ) ( h1 : ( P → Q ) ) ( h2 : ( Q → R ) ) : ( P → R ) := by have n1 : ( ( Q → R ) → ( ( P → Q ) → ( P → R ) ) ) := axB ; have n2 : ( ( P → Q ) → ( P → R ) ) := n1 h2 ; have n3 : ( P → R ) := n2 h1 ; exact n3
-- anonymous-axiom variant: the checker tries every schema, so no schema has a name token
macro "ax" : term => `(by first | exact axB | exact axC | exact axK | exact axW | exact axI)
theorem h4 ( P Q : Prop ) : ( ( P → ( P → Q ) ) → ( P → Q ) ) := by have n1 : ( ( P → ( P → Q ) ) → ( P → Q ) ) := ax ; exact n1
-- a wrong instance is rejected
-- theorem h5 ( P Q : Prop ) : ( ( P → ( P → Q ) ) → ( Q → P ) ) := by have n1 : ( ( P → ( P → Q ) ) → ( Q → P ) ) := ax ; exact n1

-- ============ 2. the same logic as lambda terms: weakening / contraction are patterns, not tokens ============
theorem l1 ( P Q R : Prop ) : ( P → ( ( P → ( P → Q ) ) → Q ) ) := by have n9 : ( P → ( ( P → ( P → Q ) ) → Q ) ) := ( fun ( n1 : P ) => by have n8 : ( ( P → ( P → Q ) ) → Q ) := ( fun ( n2 : ( P → ( P → Q ) ) ) => by have n3 : ( P → Q ) := n2 n1 ; have n4 : Q := n3 n1 ; exact n4 ) ; exact n8 ) ; exact n9
-- classical flag with an uninterpreted falsum F (no ex falso), DNE as a hypothesis schema
theorem l2 ( P Q F : Prop ) ( dn : ∀ { a : Prop } , ( ( a → F ) → F ) → a ) : ( ( ( P → F ) → F ) → P ) := by have n1 : ( ( ( P → F ) → F ) → P ) := dn ; exact n1

-- ============ 3. equational reasoning: axioms of a one-operation algebra as hypotheses, proof = chain of rewrites ============
theorem e1 ( α : Type ) ( op : α → α → α ) ( a b c : α )
    ( assoc : ∀ x y z , op ( op x y ) z = op x ( op y z ) ) ( comm : ∀ x y , op x y = op y x ) ( idem : ∀ x , op x x = x ) :
    op a ( op b a ) = op a b := by have n1 : op a ( op b a ) = op a ( op a b ) := ( by rw [ comm b a ] ) ; have n2 : op a ( op a b ) = op ( op a a ) b := ( by rw [ assoc a a b ] ) ; have n3 : op ( op a a ) b = op a b := ( by rw [ idem a ] ) ; exact n1.trans ( n2.trans n3 )
-- terse: only the rule instances, no intermediate terms
theorem e2 ( α : Type ) ( op : α → α → α ) ( a b c : α )
    ( assoc : ∀ x y z , op ( op x y ) z = op x ( op y z ) ) ( comm : ∀ x y , op x y = op y x ) ( idem : ∀ x , op x x = x ) :
    op a ( op b a ) = op a b := by rw [ comm b a , ← assoc a a b , idem a ]
-- anonymous steps: the model writes only the next term; the checker tries each axiom in each direction
macro "step" : tactic => `(tactic| first | rw [‹∀ x y z, _ = _›] | rw [← ‹∀ x y z, _ = _›] | (rw [‹∀ x y, _ = _› (_ : _) _]) | fail "no axiom justifies this step")
-- (anonymous matching of commutativity needs the instance, so a real build would write a small elaborator or enumerate instances)

-- ============ 4. Horn rules over relations: rule = hypothesis, proof = forward chain ============
theorem r1 ( α : Type ) ( R : α → α → Prop ) ( a b c d : α )
    ( symm : ∀ { x y } , R x y → R y x ) ( trans : ∀ { x y z } , R x y → R y z → R x z ) ( refl : ∀ { x } , R x x )
    ( h1 : R a b ) ( h2 : R c b ) : R a c := by have n1 : R b c := symm h2 ; have n2 : R a c := trans h1 n1 ; exact n2

-- ============ 5. permutation puzzle: a 'proof' is a word in the generators, checked by evaluation ============
def swap01 : List Nat → List Nat | a :: b :: t => b :: a :: t | l => l
def rot : List Nat → List Nat | a :: t => t ++ [a] | l => l
def rev3 : List Nat → List Nat | a :: b :: c :: t => c :: b :: a :: t | l => l
def run : List (List Nat → List Nat) → List Nat → List Nat | [], s => s | m :: ms, s => run ms (m s)
theorem p1 : run [ rot , swap01 , rot , rot , rot , rot ] [ 0 , 1 , 2 , 3 , 4 ] = [ 0 , 2 , 1 , 3 , 4 ] := by decide
