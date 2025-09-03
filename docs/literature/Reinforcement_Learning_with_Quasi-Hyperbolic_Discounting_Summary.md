# Summary: Reinforcement Learning with Quasi-Hyperbolic Discounting

This summary captures the key ideas, algorithms, and implications of incorporating quasi-hyperbolic (β, δ) time preferences into Reinforcement Learning (RL).

## Motivation
- Standard RL assumes exponential discounting (time-consistent preferences). Empirical behavior often shows present-bias: immediate rewards are overweighted relative to all delayed rewards.
- The quasi-hyperbolic model (β, δ) is a parsimonious way to introduce present-bias while preserving tractable structure for learning and control.

## Preference model and value definitions
- At any decision time t, a stream of rewards (r_t, r_{t+1}, …) is evaluated as: r_t + β ∑_{k≥1} δ^k r_{t+k}, with δ ∈ (0,1), β ∈ (0,1]. β < 1 captures present-bias; β = 1 recovers standard exponential discounting.
- For a fixed policy π and Markovian environment, define two objects:
  - Continuation (exponential) value W^π(s): satisfies W^π(s) = Eπ[r(s,a) + δ W^π(s') | s]. This is the standard δ-discounted value.
  - Quasi-hyperbolic value V^π(s): V^π(s) = Eπ[r(s,a) + β δ W^π(s') | s]. Thus V^π separates the “now” (weight 1) from all strictly future rewards (weight β, then δ-exponential).
- Action-value counterparts:
  - Q^C(s,a) = E[r(s,a) + δ max_{a'} Q^C(s',a')] (continuation control value under δ).
  - Q^I(s,a) = r(s,a) + β δ E[max_{a'} Q^C(s',a')] (immediate-self control objective).

## Agent types and control
- Naive agent: plans as if future selves will stick to the current plan (ignores future re-optimization). Typically selects actions using δ-based objectives (Q^C), but performance/evaluation exhibits present-bias via β.
- Sophisticated agent: anticipates future re-optimization and chooses a policy that is a subgame-perfect equilibrium among successive selves. In tabular control, this corresponds to choosing actions by maximizing Q^I while learning Q^C for bootstrapping.

## Algorithms
1) Policy evaluation (QH-TD(0))
- Learn W^π with standard TD(0): W(s_t) ← W(s_t) + α [r_t + δ W(s_{t+1}) − W(s_t)].
- Report/compute V on the fly: V_t(s_t) = r_t + β δ W(s_{t+1}).
- Eligibility traces (TD(λ)) extend directly by using δ in the trace decay.

2) Off-policy control (QH-Q-learning, sophisticated)
- Update continuation head like standard Q-learning:
  Q^C(s_t,a_t) ← Q^C(s_t,a_t) + α [r_t + δ max_{a'} Q^C(s_{t+1},a') − Q^C(s_t,a_t)].
- Define immediate-self objective for action selection:
  Q^I(s_t,a_t) = r_t + β δ max_{a'} Q^C(s_{t+1},a').
- Act greedily/ε-greedily with respect to Q^I; only Q^C is learned by TD to ensure stability.

3) On-policy control (QH-SARSA)
- Replace the max by the next action a_{t+1} actually taken in both the continuation update and the immediate-self objective.

4) Actor–critic (QH-AC)
- Critic learns W^π via standard δ-discounted TD; the actor’s objective uses advantages built from r_t + β δ W(s_{t+1}). Policy gradient uses returns G_t^{β,δ} = r_t + β ∑_{k≥1} δ^k r_{t+k}.

## Theoretical properties (tabular, high level)
- Continuation operator T_δ is a contraction with modulus δ; tabular TD and Q-learning for W or Q^C inherit standard convergence guarantees under usual conditions (sufficient exploration, Robbins–Monro stepsizes, bounded rewards).
- Immediate-self values (V, Q^I) are algebraic functions of the converged continuation solution; thus learnability hinges on the δ-problem.
- For sophisticated agents in stationary environments, stationary Markov equilibria exist; acting w.r.t. Q^I while learning Q^C corresponds to computing such equilibria in finite MDPs.

## Practical guidance
- β tunes present-bias intensity; set β ∈ (0,1]. β = 1 reduces to standard RL.
- Architectures with shared trunk + two heads (continuation head and immediate-self head) are effective in function approximation; only the continuation head needs bootstrapped TD targets.
- Use target networks and double Q-learning for Q^C to mitigate overestimation; β only rescales the bootstrap in Q^I.
- Evaluation and logging: report both δ-based returns (W) and (β, δ)-based returns (V) to reveal preference reversals and procrastination.

## Empirical phenomena highlighted
- Preference reversals and procrastination: agents plan to act later but change plans at the moment of acting.
- Precommitment demand: sophisticated agents may sacrifice current payoff to constrain future choices.
- Performance gaps: naive vs sophisticated agents yield different long-run rewards under the same (β, δ).

## Limitations and extensions
- Richer discounting (fully hyperbolic) may capture behavior better but is harder to learn stably.
- Partial sophistication (agents only partly anticipate future re-optimization) leads to policies between naive and sophisticated.
- Continuous spaces require function approximation; stability depends on standard deep RL caveats.

## Notes for this repository
- The decomposition used here aligns with `src/algorithms/qh_policy_evaluation.py` (learn W with TD, compute V from W) and `src/algorithms/qh_qlearning.py` (learn Q^C, act via Q^I).
- Tests in `src/tests/test_qh_algorithms.py` should distinguish β=1 (reduces to baseline RL) and β<1 cases, and verify convergence/evaluation consistency.

## Key takeaways
- Implement present-bias in RL by separating a δ-discounted continuation learner from an immediate-self objective weighted by β for strictly future rewards.
- In tabular settings, reuse standard TD/Q-learning for the continuation component and derive the quasi-hyperbolic objective from it for action selection and evaluation.
- The framework provides a clean bridge between behavioral time preferences and practical RL algorithms.
