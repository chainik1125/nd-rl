"""Sound rule/subgoal construction with counted search and swappable ordering.

The action space is shared by all policies. This is symbolic search assistance,
not unconstrained token generation. No dataset is read when this module imports.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
import sys

from classical_vendor.support import Proof, Search, serialize, proof_size, BudgetExceeded

BOT=("bot",)
def neg(f):return ("not",f)

@lru_cache(maxsize=10000)
def formula_size(f):return 1+sum(formula_size(c) for c in f[1:] if isinstance(c,tuple))

def projections(f,path=()):
    yield f,path
    if f[0]=="and":
        yield from projections(f[1],path+("ANDE1",))
        yield from projections(f[2],path+("ANDE2",))
    elif f[0]=="not" and f[1][0]=="not":
        yield from projections(f[1][1],path+("DN",))

@dataclass(frozen=True)
class Action:
    rule:str
    source:Proof|None=None
    side:int=0
    path:tuple=()

class OrderedSearch(Search):
    def __init__(self,budget=4096,ordering="heuristic",seed=0,max_depth=10):
        super().__init__(seconds=10**8,max_calls=10**9)
        if ordering not in ("heuristic","random","canonical"):
            raise ValueError(ordering)
        self.budget,self.ordering,self.rng,self.max_depth=budget,ordering,random.Random(seed),max_depth
        self.nodes=0
        self.counts={"subgoal_visits":0,"backward_action_attempts":0,"new_forward_derivations":0}
        self.action_counts={}

    def charge(self,kind):
        if self.nodes>=self.budget:raise BudgetExceeded
        self.nodes+=1;self.counts[kind]+=1

    def closure(self,context):
        facts={node.formula:node for node in context}
        def add(formula,rule,children):
            node=Proof(formula,rule,tuple(children))
            if formula not in facts or proof_size(node)<proof_size(facts[formula]):
                self.charge("new_forward_derivations")
                facts[formula]=node
                return True
            return False
        changed=True
        while changed:
            changed=False
            for f,node in list(facts.items()):
                if f[0]=="and":
                    changed|=add(f[1],"ANDE1",(node,));changed|=add(f[2],"ANDE2",(node,))
                if f[0]=="not" and f[1][0]=="not":changed|=add(f[1][1],"DN",(node,))
                if f[0]=="imp" and f[1] in facts:changed|=add(f[2],"IMPE",(node,facts[f[1]]))
                if neg(f) in facts:changed|=add(BOT,"NEGE",(node,facts[neg(f)]))
        return facts

    def actions(self,goal,facts):
        result=[]
        if goal[0]=="and":result.append(Action("ANDI"))
        elif goal[0]=="imp":result.append(Action("IMPI"))
        elif goal[0]=="not":result.append(Action("NEGI"))
        elif goal[0]=="or":result.extend((Action("ORI",side=1),Action("ORI",side=2)))
        for f,node in facts.items():
            if f[0]=="imp":
                paths=[path for target,path in projections(f[2]) if target==goal]
                if paths:result.append(Action("IMPE_PROJECT",node,path=min(paths,key=len)))
            if goal==BOT and f[0]=="not":result.append(Action("CONTRADICTION",node))
            if f[0]=="or":result.append(Action("ORE",node))
        if goal!=BOT and neg(goal) not in facts:result.append(Action("CLASSICAL"))
        return result

    def score(self,action,goal,facts):
        if action.rule in ("IMPI","NEGI","ANDI"):return (0,0)
        if action.rule=="IMPE_PROJECT":return (1,len(action.path)+formula_size(action.source.formula[1]))
        if action.rule=="CONTRADICTION":return (2,formula_size(action.source.formula[1]))
        if action.rule=="ORI":return (2,formula_size(goal[action.side]))
        if action.rule=="ORE":return (3,formula_size(action.source.formula))
        return (9,0)

    def order(self,actions,goal,facts):
        # Classical contradiction remains last for every comparison. Random
        # ordering is not disadvantaged by eagerly expanding classical detours.
        regular=[action for action in actions if action.rule!="CLASSICAL"]
        classical=[action for action in actions if action.rule=="CLASSICAL"]
        if self.ordering=="heuristic":regular.sort(key=lambda action:self.score(action,goal,facts))
        elif self.ordering=="random":self.rng.shuffle(regular)
        return regular+classical

    def prove(self,goal,context,depth,trail=frozenset()):
        self.charge("subgoal_visits")
        facts=self.closure(context)
        if goal in facts:return facts[goal]
        if BOT in facts:
            self.charge("new_forward_derivations")
            return Proof(goal,"BOTE",(facts[BOT],))
        state=(goal,frozenset(facts));cachekey=(state,depth)
        if depth<=0 or state in trail or cachekey in self.failed:return None
        trail=trail|{state};available=tuple(facts.values())
        def sub(target,context=available):return self.prove(target,context,depth-1,trail)
        for action in self.order(self.actions(goal,facts),goal,facts):
            self.charge("backward_action_attempts")
            self.action_counts[action.rule]=self.action_counts.get(action.rule,0)+1
            if action.rule=="ANDI":
                left=sub(goal[1])
                if left:
                    right=sub(goal[2],available+(left,))
                    if right:return Proof(goal,"ANDI",(left,right))
            elif action.rule in ("IMPI","NEGI"):
                assumption=self.ref(goal[1]);target=goal[2] if action.rule=="IMPI" else BOT
                inner=sub(target,available+(assumption,))
                if inner:return Proof(goal,action.rule,(assumption,inner))
            elif action.rule=="ORI":
                inner=sub(goal[action.side])
                if inner:return Proof(goal,"ORI1" if action.side==1 else "ORI2",(inner,))
            elif action.rule=="IMPE_PROJECT":
                antecedent=sub(action.source.formula[1])
                if antecedent:
                    current=Proof(action.source.formula[2],"IMPE",(action.source,antecedent))
                    for rule in action.path:
                        self.charge("new_forward_derivations")
                        formula=current.formula[1][1] if rule=="DN" else current.formula[1 if rule=="ANDE1" else 2]
                        current=Proof(formula,rule,(current,))
                    assert current.formula==goal
                    return current
            elif action.rule=="CONTRADICTION":
                positive=sub(action.source.formula[1])
                if positive:return Proof(BOT,"NEGE",(positive,action.source))
            elif action.rule=="ORE":
                left_as,right_as=self.ref(action.source.formula[1]),self.ref(action.source.formula[2])
                left=sub(goal,available+(left_as,))
                if left:
                    right=sub(goal,available+(right_as,))
                    if right:return Proof(goal,"ORE",(action.source,left_as,left,right_as,right))
            else:
                assumption=self.ref(neg(goal));contradiction=sub(BOT,available+(assumption,))
                if contradiction:return Proof(goal,"DN",(Proof(neg(neg(goal)),"NEGI",(assumption,contradiction)),))
        self.failed.add(cachekey)
        return None

    def solve(self,premise_formulas,goal):
        premises=[self.ref(formula,"PR") for formula in premise_formulas]
        proof=self.prove(goal,tuple(premises),self.max_depth)
        return serialize(premises,proof) if proof else None
