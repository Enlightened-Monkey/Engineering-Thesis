# Summary: Markov decision processes with quasi-hyperbolic discounting

This note summarizes the core ideas and results about Markov Decision Processes (MDPs) when the decision-maker has quasi-hyperbolic (β, δ) time preferences.

## Context and motivation
- Exponential discounting underpins standard dynamic programming: optimality is time-consistent and Bellman’s principle applies.
- Empirical evidence shows present-bias: people overweight immediate outcomes relative to all delayed ones.
- Quasi-hyperbolic discounting (Laibson’s (β, δ)-model) captures present-bias with minimal departure from exponential discounting and is widely used in economics and behavioral RL.

## Preference model
- Payoff stream (r0, r1, r2, …) evaluated at time 0 as: V0 = r0 + β ∑_{t≥1} δ^t r_t, with δ ∈ (0,1) and β ∈ (0,1].
- At each future time τ, the same form re-anchors at “now”: Vτ = rτ + β ∑_{t≥1} δ^t r_{τ+t}.
- Present-bias arises from β < 1. When β = 1, the model reduces to standard exponential discounting.

## Time inconsistency and agent types
- Time inconsistency: a plan made today may be overturned tomorrow because each future self re-optimizes with a new “now.”
- Two canonical behavioral types:
  - Naive: does not anticipate future self-control problems; plans under the false assumption that future selves will follow today’s plan.
  - Sophisticated: correctly anticipates future re-optimization and chooses a strategy that is a subgame-perfect equilibrium of the intrapersonal game among successive selves.

## MDP formulation with (β, δ)
- Environment: finite state space S, actions A, transition kernel P, bounded rewards r(s,a).
- Preferences: at each decision time, value of immediate reward is weighted by 1, while all strictly future rewards are down-weighted by β and then exponentially by δ.
- Consequence: standard Bellman optimality fails in general because the objective changes with the decision time.

## Equilibrium notion and main structural results (high level)
- Sophisticated agent: planning is an intrapersonal dynamic game. A (Markov) subgame-perfect equilibrium policy prescribes, for every state, an action that no future self wishes to deviate from given that all future selves will also follow the policy.
- Existence: In finite MDPs with bounded rewards, stationary Markov equilibrium policies exist under mild regularity conditions (e.g., continuity/compactness for more general cases). Randomization can be accommodated but is typically unnecessary when rewards are generic.
- Stationarity and Markov property: Despite time inconsistency, in time-homogeneous infinite-horizon settings, there often exist stationary Markov equilibria because each self faces an identical continuation problem once the current state is given.

## Computation: how to solve
- Finite horizon: backward induction on the intra-personal game. At each time and state, “today’s” self picks an action anticipating best responses of “tomorrow’s” selves.
- Infinite horizon: compute a fixed point of a best-response operator among successive selves. Two common computational views:
  1) Game-theoretic fixed point: iterate best responses until convergence to a stationary Markov equilibrium.
  2) State augmentation / nested DP: transform the problem into coupled Bellman-like recursions where “immediate” and “continuation” components are separated, then apply policy iteration/value iteration variants tailored to (β, δ).
- Naive agent: can be computed by solving a standard exponential MDP with discount δ for the continuation part, but re-planning occurs each step; performance must be evaluated under quasi-hyperbolic preferences, leading to potential large welfare losses relative to sophisticated equilibrium.

## Comparison with exponential discounting
- Bellman optimality principle does not hold globally; the optimal plan depends on the vantage point in time.
- Dynamic programming can be restored via equilibrium analysis (sophisticated case) or via augmented formulations that separate immediate utility from discounted continuation.
- Blackwell’s optimality and standard contraction arguments need adaptation; existence/uniqueness can be obtained via alternative fixed-point mappings under boundedness and compactness.

## Implications for reinforcement learning
- Policy evaluation and control must separate “immediate-self” and “future-self” values.
- Practical algorithms mirror two components: an immediate value head and a δ-discounted continuation value, mixed with weight β for strictly future rewards.
- Off-the-shelf TD(λ), Q-learning, and policy iteration can be adapted by modifying targets to implement (β, δ) aggregation and, for sophisticated agents, by embedding best-response or equilibrium updates.

## Limitations and modeling choices
- (β, δ) is a stylized present-bias model; richer hyperbolic forms may fit data better but complicate computation further.
- Sophisticated equilibrium presumes perfect foresight about future re-optimization; partial sophistication leads to different policies.
- In continuous or unbounded settings, existence and computation require additional technical conditions.

## Key takeaways
- Quasi-hyperbolic discounting introduces time inconsistency into MDPs, invalidating standard Bellman optimality.
- Sophisticated agents’ planning is an equilibrium of an intrapersonal game; stationary Markov equilibria exist broadly in finite settings.
- Computation reduces to solving a fixed point among successive selves or to coupled recursions separating immediate and continuation values.
- RL algorithms can implement (β, δ) by modifying targets and update rules; evaluation must respect present-bias.

## Notes for this repository
- The implementation in `src/algorithms/qh_policy_evaluation.py` and `src/algorithms/qh_qlearning.py` can follow the two-component target structure: immediate reward plus β-weighted δ-discounted continuation.
- Experiments should clarify whether they model naive vs sophisticated agents; results can differ markedly across the two.

