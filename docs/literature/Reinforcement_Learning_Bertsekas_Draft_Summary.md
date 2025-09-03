# Summary: Reinforcement Learning (Bertsekas Draft)

Concise overview of core concepts, algorithms, and theory in the Bertsekas RL draft, emphasizing the dynamic programming (DP) viewpoint and approximate dynamic programming (ADP).

## Problem formulations
- Discounted MDPs: infinite horizon with factor γ ∈ (0, 1); contraction mappings yield existence/uniqueness of value functions.
- Stochastic shortest path (SSP): episodic, proper policies terminate w.p.1; noncontractive DP operators require special handling.
- Average cost per stage: steady-state criterion; relative value iteration and bias functions.

## DP foundations
- Bellman operator: T(V)(s) = max_a { r(s,a) + γ E[V(s')] }.
- Contraction and monotonicity enable value iteration (VI) convergence in discounted settings.
- Policy evaluation and policy improvement: policy iteration (PI) alternates exact/approximate evaluation with greedy improvement.

## Rollout and multi-step lookahead
- Rollout: improve a base policy by 1-step (or multi-step) lookahead, simulating the base for the tail; guarantees policy improvement.
- Monte Carlo rollout approximates lookahead with sampling; trades computation for performance.

## Approximate dynamic programming (ADP)
- Function approximation for value functions (linear, nonlinear) induces the projected Bellman equation: ΠT(V) ≈ V.
- Stability and error analysis via projection operators, MSPBE, and norm-based bounds.
- Aggregation and abstraction: state aggregation, feature design, and performance guarantees under aggregation mappings.

## Temporal-difference learning
- TD(0)/TD(λ): semi-gradient methods for policy evaluation; λ controls bias–variance via eligibility traces.
- LSTD/LSTDQ: least-squares TD for linear architectures; data-efficient and stable under standard assumptions.
- LSPI: policy iteration with least-squares Q-evaluation; off-policy and batch-friendly.

## Q-learning and control
- Tabular Q-learning: off-policy, converges under Robbins–Monro stepsizes and sufficient exploration.
- Overestimation bias and remedies (double Q-learning); importance of exploration policies and learning-rate schedules.
- SARSA/on-policy methods as alternatives with different stability–performance trade-offs.

## Actor–critic and policy gradient
- Policy gradient theorem; compatible function approximation and natural gradients.
- Actor–critic: critic approximates value/advantage; actor updates parameters via gradient estimates with baselines to reduce variance.

## Off-policy learning and importance sampling
- Importance sampling for off-policy evaluation; high variance motivates per-decision ratios and control variates.
- Gradient TD (GTD, TDC) methods optimize MSPBE for stable off-policy evaluation with linear function approximation.

## Asynchronous and distributed DP/RL
- Gauss–Seidel/asynchronous VI and PI updates; convergence under fair visitation and contraction-like conditions.
- Real-time dynamic programming (RTDP) and prioritized sweeping for focused updates.

## Convergence, stability, and error bounds
- Tabular DP/RL: strong convergence guarantees from contraction properties (discounted) or special SSP/average-cost conditions.
- With function approximation: convergence depends on projection geometry; counterexamples (e.g., divergence with off-policy TD) motivate GTD/LSTD.
- Performance loss bounds: relate approximation error to suboptimality via greedy improvement and policy deviation analysis.

## Practical guidance
- Prefer PI/modified PI when policy evaluation can be solved efficiently; use VI with acceleration otherwise.
- Rollout with strong base policies offers robust anytime improvements; use sampling for expensive models.
- For linear FA, use LSTD/LSPI or GTD for stability; for nonlinear FA, adopt target networks, replay, and regularization.

## Relevance to this repository
- Your algorithms with quasi-hyperbolic preferences reuse standard δ-discounted evaluation for continuation values; all tabular convergence claims for TD/Q-learning apply to that continuation component.
- For experiments, compare β = 1 (baseline RL) vs β < 1 to isolate present-bias effects while holding δ-based learning constant.

## Key takeaways
- DP provides the backbone of RL: VI, PI, and rollout form a unifying framework.
- ADP/TD methods enable large-scale RL via approximation and sampling, with clear stability criteria in linear settings.
- Off-policy evaluation/control require care; use least-squares or gradient-TD families for stability and variance control.
