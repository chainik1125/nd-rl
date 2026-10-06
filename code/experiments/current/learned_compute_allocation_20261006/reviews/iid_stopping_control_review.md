---
written_on: 2026-10-06
written_by: agent:codex
---

# IID stopping control: mathematical review

Accepted the argument in `selection/followup_stopping_design.md` and its interpretation link in `stopping_interpretation.md`. This is a mathematical control justification, not a new experimental endpoint or an empirical finding.

Assumptions: a fixed known iid distribution of observed candidate payoffs Y in [0,1], a reliably observed incumbent x in [0,1], full recall with terminal payoff equal to the largest observed payoff, a fixed positive cost c for each new draw, risk-neutral expected utility, and a finite draw budget. The distribution and cost do not change with the remaining horizon. Conditional iid generation with unknown theorem difficulty does not by itself satisfy the known-distribution assumption.

Write g(x)=E[(Y−x)+]. For every positive remaining horizon, stopping is optimal iff g(x)≤c. If g(x)>c, one draw followed by stopping strictly improves expected utility. If g(x)≤c, every subsequent incumbent z≥x has g(z)≤c; induction gives continuation value −c+E[max(x,Y)]=x+g(x)−c≤x. At equality, stopping and continuing are both optimal, so choosing stopping is a tie convention. At zero horizon, stopping is compulsory even below the reservation threshold. If c≥E[Y], stopping is optimal throughout [0,1]; when equality holds, x=0 is a tie, while c>E[Y] gives strict preference for stopping everywhere.

The design correctly says finite horizon alone does not require count-dependent thresholds and explicitly disclaims empirical validity of the idealization. A genuine expected-value advantage from count dependence requires a departure such as changing or unknown distributions, imperfect confidence, selection effects or omitted state. A noisy finite-sample apparent advantage can also arise from estimation error or optimization differences, even without such a population advantage; the empirical criterion must retain its uncertainty and symmetric-fitting qualifications. This is a minor precision qualification, not a flaw in the stopping argument.

No code, inference, outcomes or registered protocol changed during this review.
