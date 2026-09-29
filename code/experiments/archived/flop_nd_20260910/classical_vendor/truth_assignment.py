"""Classically complete propositional certificate construction, independent of RL.

This module reads no data. It shares only the immutable generic Proof/serializer
and strict verifier with earlier work. It never imports a development evaluator.
Resources are explicit: charge each partial-formula evaluation, decision node,
and constructed proof node. The finite budget can truncate a complete method.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import sys

from classical_vendor.support import Proof, serialize, neg, BOT
from nd_data import sequent_parts
from nd_verify.verify import parse_formula

class ConstructionBudgetExceeded(Exception):pass

@dataclass(frozen=True)
class Countermodel:
    assignment:dict


def parse(text):
    f,end=parse_formula(text.split(),0)
    if end!=len(text.split()):raise ValueError('Unparsed formula suffix')
    return f


def atoms(formula):
    if formula[0]=='atom':return {formula[1]}
    return set().union(*(atoms(child) for child in formula[1:] if isinstance(child,tuple)))


class TruthAssignmentConstructor:
    def __init__(self,budget=1000000,atom_order='alphabetical'):
        if atom_order not in ('alphabetical','frequency'):raise ValueError('Unsupported fixed atom order')
        self.budget=budget;self.atom_order=atom_order;self.identities=count(1)
        self.counts={'partial_formula_evaluations':0,'decision_visits':0,'constructed_proof_nodes':0}
        self.valuation_leaves=0;self.decision_splits=0

    @property
    def operations(self):return sum(self.counts.values())

    def charge(self,kind):
        if self.operations>=self.budget:raise ConstructionBudgetExceeded
        self.counts[kind]+=1

    def node(self,formula,rule,args=()):
        self.charge('constructed_proof_nodes')
        return Proof(formula,rule,tuple(args),next(self.identities) if rule in ('AS','PR') else 0)

    def value(self,formula,assignment,cache):
        if formula in cache:return cache[formula]
        self.charge('partial_formula_evaluations');kind=formula[0]
        if kind=='atom':answer=assignment.get(formula[1])
        elif kind=='bot':answer=False
        elif kind=='not':
            a=self.value(formula[1],assignment,cache);answer=None if a is None else not a
        else:
            a=self.value(formula[1],assignment,cache);b=self.value(formula[2],assignment,cache)
            if kind=='and':answer=False if a is False or b is False else True if a is True and b is True else None
            elif kind=='or':answer=True if a is True or b is True else False if a is False and b is False else None
            elif kind=='imp':answer=True if a is False or b is True else False if a is True and b is False else None
            else:raise ValueError(f'Unsupported connective {kind}')
        cache[formula]=answer;return answer

    def signed(self,formula,assignment,literals,values,proofs):
        """Build formula when true, or its negation when false, by induction."""
        if formula in proofs:return proofs[formula]
        truth=self.value(formula,assignment,values)
        if truth is None:raise ValueError('A partial valuation cannot justify this signed formula')
        kind=formula[0]
        if kind=='atom':result=literals[formula[1]]
        elif kind=='bot':
            assumption=self.node(BOT,'AS');result=self.node(neg(BOT),'NEGI',(assumption,assumption))
        elif kind=='not':
            inner=self.signed(formula[1],assignment,literals,values,proofs)
            if truth:result=inner
            else:
                assumption=self.node(formula,'AS');contradiction=self.node(BOT,'NEGE',(inner,assumption));result=self.node(neg(formula),'NEGI',(assumption,contradiction))
        elif kind=='and':
            if truth:result=self.node(formula,'ANDI',tuple(self.signed(child,assignment,literals,values,proofs) for child in formula[1:]))
            else:
                side=1 if self.value(formula[1],assignment,values) is False else 2
                negative=self.signed(formula[side],assignment,literals,values,proofs);assumption=self.node(formula,'AS')
                child=self.node(formula[side],f'ANDE{side}',(assumption,));contradiction=self.node(BOT,'NEGE',(child,negative));result=self.node(neg(formula),'NEGI',(assumption,contradiction))
        elif kind=='or':
            if truth:
                side=1 if self.value(formula[1],assignment,values) is True else 2
                child=self.signed(formula[side],assignment,literals,values,proofs);result=self.node(formula,f'ORI{side}',(child,))
            else:
                left=self.signed(formula[1],assignment,literals,values,proofs);right=self.signed(formula[2],assignment,literals,values,proofs);assumption=self.node(formula,'AS')
                left_as=self.node(formula[1],'AS');left_bot=self.node(BOT,'NEGE',(left_as,left));right_as=self.node(formula[2],'AS');right_bot=self.node(BOT,'NEGE',(right_as,right))
                contradiction=self.node(BOT,'ORE',(assumption,left_as,left_bot,right_as,right_bot));result=self.node(neg(formula),'NEGI',(assumption,contradiction))
        elif kind=='imp':
            if truth:
                assumption=self.node(formula[1],'AS')
                if self.value(formula[2],assignment,values) is True:inner=self.signed(formula[2],assignment,literals,values,proofs)
                else:
                    negative=self.signed(formula[1],assignment,literals,values,proofs);contradiction=self.node(BOT,'NEGE',(assumption,negative));inner=self.node(formula[2],'BOTE',(contradiction,))
                result=self.node(formula,'IMPI',(assumption,inner))
            else:
                positive=self.signed(formula[1],assignment,literals,values,proofs);negative=self.signed(formula[2],assignment,literals,values,proofs);assumption=self.node(formula,'AS')
                consequent=self.node(formula[2],'IMPE',(assumption,positive));contradiction=self.node(BOT,'NEGE',(consequent,negative));result=self.node(neg(formula),'NEGI',(assumption,contradiction))
        else:raise ValueError(kind)
        expected=formula if truth else neg(formula)
        if result.formula!=expected:raise AssertionError('Signed derivation mismatch')
        proofs[formula]=result;return result

    def excluded_middle(self,atom):
        disjunction=('or',atom,neg(atom));negative=self.node(neg(disjunction),'AS');positive=self.node(atom,'AS')
        inner_or=self.node(disjunction,'ORI1',(positive,));inner_bot=self.node(BOT,'NEGE',(inner_or,negative));not_atom=self.node(neg(atom),'NEGI',(positive,inner_bot))
        outer_or=self.node(disjunction,'ORI2',(not_atom,));outer_bot=self.node(BOT,'NEGE',(outer_or,negative));double_negative=self.node(neg(neg(disjunction)),'NEGI',(negative,outer_bot));return self.node(disjunction,'DN',(double_negative,))

    def solve(self,prompt):
        premises_text,goal_text=sequent_parts(prompt);premises=tuple(self.node(parse(x),'PR') for x in premises_text);goal=parse(goal_text)
        all_formulas=tuple(p.formula for p in premises)+(goal,);names=set().union(*(atoms(x) for x in all_formulas))
        if self.atom_order=='frequency':
            def occurrences(formula,name):return int(formula==('atom',name))+sum(occurrences(c,name) for c in formula[1:] if isinstance(c,tuple))
            names=sorted(names,key=lambda n:(-sum(occurrences(f,n) for f in all_formulas),n))
        else:names=sorted(names)
        def visit(assignment,literals):
            self.charge('decision_visits');values={};proofs={}
            # False premise or true goal yields a checked local proof under this partial valuation.
            for premise in premises:
                if self.value(premise.formula,assignment,values) is False:
                    self.valuation_leaves+=1;negative=self.signed(premise.formula,assignment,literals,values,proofs);contradiction=self.node(BOT,'NEGE',(premise,negative));return self.node(goal,'BOTE',(contradiction,))
            if self.value(goal,assignment,values) is True:
                self.valuation_leaves+=1;return self.signed(goal,assignment,literals,values,proofs)
            unassigned=[n for n in names if n not in assignment]
            if not unassigned:
                self.valuation_leaves+=1;return Countermodel(dict(assignment))
            self.decision_splits+=1;name=unassigned[0];atom=('atom',name)
            left_as=self.node(atom,'AS');left=visit({**assignment,name:True},{**literals,name:left_as})
            if isinstance(left,Countermodel):return left
            right_as=self.node(neg(atom),'AS');right=visit({**assignment,name:False},{**literals,name:right_as})
            if isinstance(right,Countermodel):return right
            lemma=self.excluded_middle(atom);return self.node(goal,'ORE',(lemma,left_as,left,right_as,right))
        result=visit({},{});meta={'operations':self.operations,'operation_components':dict(self.counts),'decision_splits':self.decision_splits,'valuation_leaves':self.valuation_leaves,'atom_order':names,'symbolic_assistance_only':True,'no_learned_policy':True}
        if isinstance(result,Countermodel):return {**meta,'status':'countermodel','countermodel':result.assignment}
        return {**meta,'status':'certificate','body':serialize(premises,result)}
