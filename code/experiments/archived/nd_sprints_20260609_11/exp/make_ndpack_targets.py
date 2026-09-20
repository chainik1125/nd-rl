"""Extract propositional sequents from The Natural Deduction Pack (Alastair Carr,
March 2018) into our machine format -> data/targets_ndpack.jsonl.

Each problem is transcribed in (near-)source notation using a tiny parser that
accepts the pack's connectives (-> & v ~ and the ascii aliases). Atom names in
the source (P, Q, R, P1, P2, ...) are remapped onto our vocab P,Q,R,S by order
of first appearance across (premises ++ conclusion). Problems with >4 distinct
atoms, with quantifiers/predicates/identity, or using the biconditional (not
representable in our connective set) are SKIPPED and counted.

Every emitted record is validated with a brute-force truth-table check
(classical semantics, F always false).  Uses only the stdlib + nd.core.
"""
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.core import (atom, f_not, f_and, f_or, f_imp, formula_str,
                     canon_premises, prompt_tokens, theorem_id)
from nd.tokenizer import STOI

OUR_ATOMS = ['P', 'Q', 'R', 'S']


# ---------------------------------------------------------------------------
# Tiny recursive-descent parser for the pack's propositional notation.
# Connectives (highest binding first): ~  &  v  >  (right assoc for >).
# We normalise unicode/ascii variants before parsing. Atom tokens are
# alnum identifiers (e.g. P, Q, R, P1, Q2). '#BICOND' marks <-> so we can
# reject biconditional problems. '#QUANT' marks any quantifier/identity.
# ---------------------------------------------------------------------------

def _normalise(s):
    # Biconditional first -> single sentinel token so the '->' inside '<->'
    # is not later rewritten into an implication arrow.
    s = s.replace('↔', ' <-> ').replace('<->', ' <-> ')
    s = re.sub(r'<\s*->', ' <-> ', s)          # any spacing variant
    s = s.replace('<->', ' \x00BICOND\x00 ')
    s = s.replace('∧', ' & ').replace('∨', ' v ')
    s = s.replace('→', ' > ').replace('->', ' > ').replace('⊃', ' > ')
    s = s.replace('¬', ' ~ ').replace('!', ' ~ ').replace('~', ' ~ ')
    s = s.replace('\x00BICOND\x00', ' <-> ')
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    return s


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, t):
        if self.peek() != t:
            raise ParseError(f"expected {t!r} got {self.peek()!r}")
        self.next()

    # grammar:  imp := or ('>' imp)?       (right assoc)
    #           or  := and ('v' and)*
    #           and := neg ('&' neg)*
    #           neg := '~' neg | atom | '(' imp ')'
    def parse(self):
        f = self.p_imp()
        if self.peek() is not None:
            raise ParseError(f"trailing {self.peek()!r}")
        return f

    def p_imp(self):
        left = self.p_or()
        if self.peek() == '>':
            self.next()
            right = self.p_imp()
            return f_imp(left, right)
        if self.peek() == '<->':
            raise ParseError('BICOND')
        return left

    def p_or(self):
        left = self.p_and()
        while self.peek() == 'v':
            self.next()
            left = f_or(left, self.p_and())
        return left

    def p_and(self):
        left = self.p_neg()
        while self.peek() == '&':
            self.next()
            left = f_and(left, self.p_neg())
        return left

    def p_neg(self):
        if self.peek() == '~':
            self.next()
            return f_not(self.p_neg())
        if self.peek() == '(':
            self.next()
            f = self.p_imp()
            self.expect(')')
            return f
        t = self.peek()
        if t is None or t in (')', '>', 'v', '&', '<->'):
            raise ParseError(f"expected atom got {t!r}")
        self.next()
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', t):
            raise ParseError(f"bad atom {t!r}")
        return ('rawatom', t)   # raw name; remapped later


def parse_formula(s):
    toks = _normalise(s).split()
    return Parser(toks).parse()


# ---- atom collection + remap ------------------------------------------------

def raw_atoms_in_order(f, acc):
    if f[0] == 'rawatom':
        if f[1] not in acc:
            acc.append(f[1])
        return
    for sub in f[1:]:
        if isinstance(sub, tuple):
            raw_atoms_in_order(sub, acc)


def remap(f, mapping):
    if f[0] == 'rawatom':
        return atom(mapping[f[1]])
    if f[0] in ('atom', 'bot'):
        return f
    if f[0] == 'not':
        return ('not', remap(f[1], mapping))
    return (f[0], remap(f[1], mapping), remap(f[2], mapping))


# ---------------------------------------------------------------------------
# Truth-table validation (classical; bot always false)
# ---------------------------------------------------------------------------

def eval_formula(f, val):
    t = f[0]
    if t == 'atom':
        return val[f[1]]
    if t == 'bot':
        return False
    if t == 'not':
        return not eval_formula(f[1], val)
    if t == 'and':
        return eval_formula(f[1], val) and eval_formula(f[2], val)
    if t == 'or':
        return eval_formula(f[1], val) or eval_formula(f[2], val)
    if t == 'imp':
        return (not eval_formula(f[1], val)) or eval_formula(f[2], val)
    raise ValueError(f)


def atoms_of(f):
    if f[0] == 'atom':
        return {f[1]}
    if f[0] == 'bot':
        return set()
    return set().union(*(atoms_of(s) for s in f[1:])) if len(f) > 1 else set()


def is_valid(premises, conclusion):
    names = sorted(set().union(atoms_of(conclusion), *(atoms_of(p) for p in premises)))
    for bits in itertools.product([False, True], repeat=len(names)):
        val = dict(zip(names, bits))
        if all(eval_formula(p, val) for p in premises) and not eval_formula(conclusion, val):
            return False
    return True


# ---------------------------------------------------------------------------
# The problems.  Each entry: (page, idx_label, [premise_strings], concl_string,
#                             has_solution, classical_only_guess, proof_text)
# idx_label is used for the record name.  proof_text is the prose strategy from
# the matching solution (or '' when no worked solution / pure problem statement).
# classical_only_guess: True iff (my judgment) the proof essentially needs
# double-negation-elim / excluded middle (i.e. not intuitionistically derivable).
# ---------------------------------------------------------------------------

P = lambda *a: a  # noqa  (cosmetic)

PROBLEMS = [
    # --- Worked example (section 3.1, page 5) -------------------------------
    (5, "wex1", [], "(P > Q) > ((P & R) > (Q & R))", True, False,
     "Bottom-up: conclusion is an implication, assume P>Q and prove (P&R)>(Q&R) "
     "by assuming P&R and proving Q&R via two branches (Q by ->Elim on P from "
     "AndElim with P>Q; R by AndElim), then AndIntro and two ->Intro steps."),

    # --- 4.1 Core (page 11) -------------------------------------------------
    # Core 1,2,3 are propositional/quantifier mix; 1 is propositional.
    (11, "core1", ["(P > P) <-> Q"], "Q v R", True, False, "BICOND-premise"),  # will be skipped (bicond)
    (11, "core4", ["~P1"], "~((P1 & P2) & P3)", True, False,
     "Conclusion ~((P1&P2)&P3): assume (P1&P2)&P3, derive P1 via AndElim twice, "
     "contradiction with ~P1, then ~Intro."),
    (11, "core5", ["~P v Q"], "P > Q", True, False,
     "Assume P; OrElim on ~P v Q: left case ~P with P gives Q by ~Elim, right "
     "case Q directly; then ->Intro."),
    (11, "core7", [], "(P & Q) <-> (Q & P)", True, False, "BICOND"),  # skipped

    # --- 4.3 Implication (page 14) -----------------------------------------
    (14, "imp1", [], "P > P", True, False,
     "Special case of ->Intro where antecedent and consequent coincide: assume "
     "P, discharge it immediately."),
    (14, "imp2", [], "P > (Q > P)", True, False,
     "Assume P, then ->Intro vacuously to get Q>P, then ->Intro discharging P."),
    (14, "imp3", ["P > Q", "Q > R"], "P > R", True, False,
     "Assume P; P with P>Q gives Q (->Elim); Q with Q>R gives R; ->Intro "
     "discharges P. Hypothetical syllogism."),
    (14, "imp4", [], "P > ((P > Q) > Q)", True, False,
     "Assume P and P>Q, ->Elim gives Q, then two ->Intro steps."),
    (14, "imp5", ["(P > Q) > (P > R)"], "Q > (P > R)", True, False,
     "Assume Q; ->Intro (vacuous) gives P>Q; premise gives P>R; ->Intro "
     "discharges Q."),
    (14, "imp6", ["(P > Q) > P"], "Q > P", True, False,
     "Assume Q; vacuous ->Intro gives P>Q; premise (P>Q)>P gives P; ->Intro "
     "discharges Q."),
    (14, "imp7", ["P > (Q > R)"], "Q > (P > R)", True, False,
     "Assume P and Q; premise+P gives Q>R; +Q gives R; two ->Intro steps."),
    (14, "imp8", ["P > (Q > R)", "P > Q"], "P > R", True, False,
     "Assume P; premises give Q>R and Q; ->Elim gives R; ->Intro discharges P."),
    (14, "imp9", ["(P > P) > Q"], "(Q > R) > R", True, False,
     "Assume Q>R; prove P>P (assume P, ->Intro); premise gives Q; Q with Q>R "
     "gives R; ->Intro. (2009 past paper)."),
    (14, "imp10", [], "(P > (Q > R)) > ((P > Q) > (P > R))", True, False,
     "Frege. Assume P>(Q>R), P>Q, P; derive Q and Q>R, then R; three ->Intro."),
    # Mixed with conjunction
    (14, "imp11", ["P & Q"], "P > Q", True, False,
     "P&Q gives Q by AndElim, then P>Q by vacuous ->Intro."),
    (14, "imp12", [], "(P & Q) > P", True, False,
     "Assume P&Q, AndElim gives P, ->Intro."),
    (14, "imp13", ["P > (Q & R)"], "P > Q", True, False,
     "Assume P; premise gives Q&R; AndElim gives Q; ->Intro."),
    (14, "imp14", ["((P & Q) > Q) > (Q > P)"], "Q > P", True, False,
     "Prove (P&Q)>Q (assume P&Q, AndElim Q, ->Intro); premise gives Q>P "
     "directly by ->Elim."),
    (14, "imp15", ["(P & Q) > R"], "P > (Q > R)", True, False,
     "Assume P and Q; AndIntro gives P&Q; premise gives R; two ->Intro. Export."),
    (14, "imp16", ["(P > Q) & (P > R)"], "P > (Q & R)", True, False,
     "Assume P; split premise into P>Q and P>R; derive Q and R; AndIntro; "
     "->Intro."),
    (14, "imp17", ["P > (Q & R)"], "(P > Q) & (P > R)", True, False,
     "Two sub-proofs: assume P, premise gives Q&R, AndElim to Q (resp R), "
     "->Intro each, then AndIntro."),
    # Bonus challenge (page 14)
    (14, "impbonus", ["(((P1 & P2) & P3) & P4) & P5"], "P1 & P1", True, False,
     "AndElim down to P1, prove P1>(P1&P1) via AndIntro of two assumptions, "
     "then ->Elim; 7 steps."),

    # --- 4.2 Conjunction (page 13) -----------------------------------------
    (13, "conj1", ["P", "Q"], "P & Q", True, False, "AndIntro."),
    (13, "conj2", ["(P1 & P2) & P3"], "P2", True, False,
     "AndElim1 then AndElim2."),
    (13, "conj3", ["P & Q"], "Q & P", True, False,
     "AndElim each side, AndIntro in swapped order."),
    (13, "conj4", ["Q & P", "R"], "P & (R & Q)", True, False,
     "AndElim2 of Q&P gives P; AndElim1 gives Q; AndIntro R&Q; AndIntro P&(R&Q)."),
    (13, "conj5", ["P1 & P2", "(Q1 & Q2) & R"], "(P1 & Q2) & R", True, False,
     "AndElim P1; from (Q1&Q2)&R get Q1&Q2 then Q2, and R; build (P1&Q2)&R."),
    (13, "conj6", ["P & (Q & R)"], "(R & P) & Q", True, False,
     "Project P, Q, R from nested conjunction, reassemble (R&P)&Q."),

    # --- 4.4 Disjunction (page 15) -----------------------------------------
    (15, "disj1", ["P v Q"], "Q v P", True, False,
     "OrElim on P v Q: each disjunct gives Q v P by OrIntro."),
    (15, "disj2", ["P v Q"], "P v (Q v R)", True, False,
     "OrElim: P gives P v (Q v R) by OrIntro1; Q gives Q v R then P v (Q v R)."),
    (15, "disj3", ["(P v Q) v R"], "P v (Q v R)", True, False,
     "Nested OrElim: outer split on (P v Q) v R, inner split on P v Q; rebuild."),
    (15, "disj4", ["(P v Q) v (R v P1)"], "(P v P1) v (R v Q)", True, False,
     "Systematic nested OrElim over four disjuncts P,Q,R,P1; OrIntro to target."),
    # Mixed with conjunction
    (15, "disj5", ["P & (Q v R)"], "(P & Q) v (P & R)", True, False,
     "Distribute: AndElim gives P and Q v R; OrElim on Q v R, AndIntro P with "
     "each, OrIntro. Distribution and over or."),
    (15, "disj6", ["(P v Q) & (P v R)"], "P v (Q & R)", True, False,
     "From the two disjunctive conjuncts, nested OrElim; when both Q and R hold "
     "form Q&R; otherwise P. Distribution."),
    (15, "disj7", ["(P & Q) v (P & R)"], "P & (Q v R)", True, False,
     "OrElim: each case yields P (AndElim) and Q v R (OrIntro on AndElim), then "
     "AndIntro."),
    (15, "disj8", ["P v (Q & R)"], "(P v Q) & (P v R)", True, False,
     "OrElim: P gives both P v Q and P v R; Q&R gives Q and R hence both; "
     "AndIntro."),
    # Mixed with implication
    (15, "disj9", ["(P > Q) v Q"], "P > Q", True, False,
     "OrElim: left case P>Q directly; right case Q gives P>Q by vacuous ->Intro."),
    (15, "disj10", ["P v Q"], "(P > Q) > Q", True, False,
     "Assume P>Q; OrElim on P v Q: P with P>Q gives Q, Q directly; ->Intro."),
    (15, "disj11", ["(P > Q) > (P > R)"], "(P v R) > (Q > R)", True, False,
     "Assume P v R and Q; OrElim: R gives R; P with (assume Q -> P>Q) premise "
     "gives P>R then R; ->Intro twice."),
    (15, "disj12", ["(P > Q) v (P > R)"], "P > (Q v R)", True, False,
     "Assume P; OrElim on premise: each conditional with P gives Q (resp R) "
     "then Q v R; ->Intro."),
    # Mixed with conjunction and implication
    (15, "disj13", ["(P > Q) & (Q > P)"], "(P v Q) > (P & Q)", True, False,
     "Assume P v Q; OrElim: P gives Q (via P>Q) hence P&Q; Q gives P hence "
     "P&Q; ->Intro."),
    (15, "disj14", ["(P v Q) > (P & Q)"], "(P > Q) & (Q > P)", True, False,
     "Two ->Intro proofs: from P get P v Q, premise gives P&Q, AndElim Q; "
     "symmetrically for Q>P; AndIntro."),
    (15, "disj15", ["(Q > R) & (Q v P)"], "(P > Q) > (R & Q)", True, False,
     "Assume P>Q; derive Q (OrElim on Q v P, P case uses P>Q), and R (Q>R), "
     "AndIntro; ->Intro."),

    # --- 4.6 Negation (page 17-18) -----------------------------------------
    # Negation introduction
    (17, "neg1", ["P"], "~~P", True, False,
     "Assume ~P; with P that's a contradiction; ~Intro discharges ~P giving ~~P."),
    (17, "neg2", ["~P"], "~(P & Q)", True, False,
     "Assume P&Q; AndElim gives P contradicting ~P; ~Intro."),
    (17, "neg3", ["P > ~P"], "~P", True, False,
     "Assume P; ->Elim with P>~P gives ~P; contradiction; ~Intro discharges P."),
    (17, "neg4", ["~(P > Q)"], "~Q", True, False,
     "Assume Q; vacuous ->Intro gives P>Q contradicting ~(P>Q); ~Intro."),
    (17, "neg5", ["~(P & Q)"], "P > ~Q", True, False,
     "Assume P; assume Q; AndIntro P&Q contradicts ~(P&Q); ~Intro gives ~Q; "
     "->Intro."),
    (17, "neg6", ["P > Q"], "~Q > ~P", True, False,
     "Assume ~Q; assume P; ->Elim gives Q contradicting ~Q; ~Intro gives ~P; "
     "->Intro. Contraposition."),
    (17, "neg7", [], "~((P & ~P) v (Q & ~Q))", True, False,
     "Assume (P&~P) v (Q&~Q); OrElim, each disjunct is its own contradiction; "
     "~Intro (re-assume to discharge across branches)."),
    (17, "neg8", ["~(P v Q)"], "~P & ~Q", True, False,
     "Assume P, OrIntro gives P v Q contradicting premise -> ~P; similarly ~Q; "
     "AndIntro. De Morgan."),
    (17, "neg9", ["~P v ~Q"], "~(P & Q)", True, False,
     "OrElim on ~P v ~Q; in each case assume P&Q, AndElim contradicts the "
     "negation, ~Intro. De Morgan."),
    # Ex falso quodlibet
    (17, "neg10", ["~P"], "P > Q", True, False,
     "Assume P; P and ~P give a contradiction; ~Elim derives Q; ->Intro."),
    (17, "neg11", ["P & ~P"], "Q", True, False,
     "AndElim gives P and ~P; ~Elim derives Q. Explosion."),
    (17, "neg12", ["P v Q"], "~P > Q", True, False,
     "Assume ~P; OrElim on P v Q: P with ~P gives Q by ~Elim, Q directly; "
     "->Intro."),
    (17, "neg13", ["P > Q", "P & ~Q"], "R", True, False,
     "AndElim gives P and ~Q; P>Q gives Q; Q and ~Q contradict; ~Elim gives R."),
    # neg14 has a biconditional premise -> skipped
    (17, "neg14", ["P v Q", "P <-> Q", "~(P & Q)"], "R", True, False, "BICOND"),
    # Indirect proofs
    (17, "neg15", ["~~P"], "P", True, True,
     "Assume ~P contradicting ~~P; ~Elim discharges ~P and gives P. "
     "Double-negation elimination -- classical."),
    (17, "neg16", [], "P v ~P", True, True,
     "Assume ~(P v ~P); assume P gives P v ~P contradiction -> ~P; ~P gives "
     "P v ~P, contradiction again; ~Elim. Excluded middle -- classical."),
    (17, "neg17", ["~(~P v ~Q)"], "P & Q", True, True,
     "Assume ~P -> ~P v ~Q contradicts premise -> ~Elim gives P; similarly Q; "
     "AndIntro. Uses ~Elim -- classical."),
    (17, "neg18", ["~(P & Q)"], "~P v ~Q", True, True,
     "Assume ~(~P v ~Q); assume ~P gives ~P v ~Q contradiction -> P; P and "
     "~(P&Q) give ~Q -> ~P v ~Q contradiction; ~Elim. De Morgan -- classical."),
    # Mixed problems
    (17, "neg19", ["~(P > Q)"], "P", True, True,
     "Assume ~P; ~P lets us derive Q from any P (problem 10) hence P>Q "
     "contradicting ~(P>Q); ~Elim gives P. Classical."),
    (17, "neg20", ["(P > Q) > P"], "P", True, True,
     "Assume ~P; assuming P with ~P gives Q, ->Intro gives P>Q, premise gives "
     "P contradicting ~P; ~Elim gives P. Peirce -- classical."),
    # neg21 has a biconditional -> skipped
    (17, "neg21", ["P <-> ~~Q"], "P <-> Q", True, False, "BICOND"),
    (17, "neg22", ["(P > Q) > Q"], "~Q > P", True, True,
     "Assume ~Q; assume ~P; assume P with ~P gives Q (->Intro P>Q), premise "
     "gives Q contradicting ~Q -> ~Elim gives P. Classical."),
    (18, "neg23", ["~P & ~Q"], "~(P v Q)", True, False,
     "Assume P v Q; OrElim, each disjunct contradicts ~P resp ~Q from premise; "
     "~Intro. De Morgan."),
    (18, "neg24", [], "P v (P > Q)", True, True,
     "Assume ~(P v (P>Q)); assume P -> P v (P>Q) contradiction -> ~P; from ~P "
     "derive P>Q -> P v (P>Q) contradiction; ~Elim. Classical."),
    (18, "neg25", [], "(P > Q) v (Q > R)", True, True,
     "Indirect: assume the negation; assuming Q gives P>Q hence the disjunction "
     "(contradiction -> ~Q); ~Q gives Q>R hence the disjunction again; ~Elim. "
     "Classical."),
    (18, "neg26", ["~P > Q", "R v ~Q", "P > (Q1 v Q2)", "~R & ~Q2"], "Q1", True, True,
     "From ~R&~Q2 get ~R and ~Q2; ~R with R v ~Q gives ~Q; assume ~P, premise "
     "gives Q contradiction -> P; P>(Q1 v Q2) gives Q1 v Q2; ~Q2 gives Q1. "
     "(2010 past paper). Classical."),
    (18, "neg27", ["P > (Q v R)"], "(P > Q) v (P > R)", True, True,
     "Indirect proof of a disjunction: assume ~((P>Q) v (P>R)); assume P, "
     "premise gives Q v R, each disjunct gives one conditional hence the "
     "disjunction (contradiction); ~Elim. Classical."),
    # neg28 conclusion is a biconditional -> skipped
    (18, "neg28", [], "~(P & Q) <-> (~P v ~Q)", True, True, "BICOND"),
    # Bonus challenge 1 (page 18)
    (18, "negbonus1a", ["~~~P"], "~P", True, False,
     "Without ~Elim: assume P and ~P, contradiction discharges ~P giving ~~P, "
     "which contradicts ~~~P; ~Intro discharges P giving ~P. Triple negation."),
    (18, "negbonus1b", ["P"], "~~P", True, False,
     "Without ~Intro is asked in the challenge; classically ~~P from P: assume "
     "~P, contradiction with P, ~Intro. (Standard ~Intro version.)"),
    # Bonus challenge 2 (page 18)
    (18, "negbonus2", ["~~P & ~~Q"], "P & Q", True, True,
     "Two indirect proofs: ~~P with assumed ~P gives a contradiction -> P; "
     "similarly Q; AndIntro. Uses ~Elim -- classical."),

    # --- 4.10 Additional challenges: Contraposition (page 25) --------------
    # These are stated as concrete propositional sequents (in a modified proof
    # system) but are ordinary classically-valid sequents; we keep the valid
    # concrete ones.  The metatheory items (3,4,9,10,11) are not sequents.
    (25, "addcontra1", ["~(Q > P)"], "~P", True, True,
     "Without ~Intro, via contraposition rule. Classically valid: from "
     "~(Q>P), assuming P gives Q>P, contradiction. Classical."),
    (25, "addcontra2", [], "~(P & ~P)", True, False,
     "Without ~Intro: ~(P & ~P) holds; assume P&~P, contradiction. "
     "(Stated for the contraposition exercise.)"),
    (25, "addcontra7", ["(Q > Q) > P"], "P", True, False,
     "In the contraposition system: Q>Q is a theorem, ->Elim gives P. "
     "Classically and intuitionistically valid."),
    (25, "addcontra8", ["~~P"], "P", True, True,
     "Double-negation elimination, target of the contraposition exercise. "
     "Classical."),
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(root, 'data', 'targets_ndpack.jsonl')

    stats = {
        'seen': 0, 'bicond': 0, 'too_many_atoms': 0, 'parse_or_quant': 0,
        'failed_validation': 0, 'emitted': 0, 'dup': 0,
    }
    records = []
    seen_thms = set()
    failures = []

    for (page, label, prem_strs, concl_str, has_sol, cls_only, proof) in PROBLEMS:
        stats['seen'] += 1
        # parse all parts; biconditional / quantifier raise ParseError
        try:
            raw_prems = [parse_formula(s) for s in prem_strs]
            raw_concl = parse_formula(concl_str)
        except ParseError as e:
            if 'BICOND' in str(e):
                stats['bicond'] += 1
            else:
                stats['parse_or_quant'] += 1
            continue

        # collect atoms by first appearance across premises then conclusion
        order = []
        for f in raw_prems:
            raw_atoms_in_order(f, order)
        raw_atoms_in_order(raw_concl, order)
        if len(order) > 4:
            stats['too_many_atoms'] += 1
            continue
        mapping = {name: OUR_ATOMS[i] for i, name in enumerate(order)}

        prems = [remap(f, mapping) for f in raw_prems]
        concl = remap(raw_concl, mapping)

        if not is_valid(prems, concl):
            stats['failed_validation'] += 1
            failures.append((page, label, prem_strs, concl_str))
            continue

        cprems = canon_premises(prems)
        thm = theorem_id(cprems, concl)
        if thm in seen_thms:
            stats['dup'] += 1
            continue
        seen_thms.add(thm)

        prompt = ' '.join(prompt_tokens(cprems, concl))
        for tok in prompt.split():
            if tok not in STOI:
                raise SystemExit(f"token {tok!r} not in vocab ({label})")

        n_atoms = len(order)
        n_prem = len(cprems)
        rec = {
            'thm': thm,
            'prompt': prompt,
            'name': f'ndpack_p{page}_{label}',
            'n_atoms': n_atoms,
            'n_premises': n_prem,
            'has_solution': bool(has_sol),
            'human_proof_text': proof if has_sol else '',
            'classical_only': bool(cls_only),
            'source_page': page,
        }
        records.append(rec)
        stats['emitted'] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        for rec in records:
            fh.write(json.dumps(rec) + '\n')

    # report-ready summary
    from collections import Counter
    atom_hist = Counter(r['n_atoms'] for r in records)
    prem_hist = Counter(r['n_premises'] for r in records)

    print("=== ndpack extraction ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  n_atoms histogram: {dict(sorted(atom_hist.items()))}")
    print(f"  n_premises histogram: {dict(sorted(prem_hist.items()))}")
    print(f"  with_solution: {sum(r['has_solution'] for r in records)}")
    print(f"  classical_only(True): {sum(r['classical_only'] for r in records)}")
    if failures:
        print("  VALIDATION FAILURES:")
        for f in failures:
            print(f"    {f}")
    print(f"wrote {out_path}")
    return records, stats, atom_hist, prem_hist, failures


if __name__ == '__main__':
    main()
