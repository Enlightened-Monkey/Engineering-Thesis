# QH Policy Evaluation vs. Algorithm 1

This note maps Algorithm 1 (model-free policy evaluation for quasi-hyperbolic discounting) to the current implementation in `src/algorithms/qh_policy_evaluation.py` and shows how `src/experiments/InventoryMDP.py` uses it.

## Roles of policies
- `sampling_policy` (ν): behaviour policy used to generate transitions and importance weights.
- `mu_policy` (μ): evaluation policy for the **first** decision in the quasi-hyperbolic return.
- `phi_policy` (φ): evaluation policy for the continuation (after the first step).

## Algorithm (paper)
For each state update, Algorithm 1 does:
1. Sample `a ~ ν(.|s)`, observe `(r(s,a), s')`.
2. Sample `a' ~ φ(.|s')`, observe `r(s',a')`.
3. TD target: `r_target = r(s,a) - (1-σ)γ r(s',a') + γ W_n(s')`.
4. Fast timescale: `W_{n+1}(s) = W_n(s) + α_n * [ φ(a|s)/ν(a|s) * (r_target - W_n(s)) ]`.
5. Slow timescale: `V_{n+1}(s) = V_n(s) + β_n * [ μ(a|s)/ν(a|s) * (r_target - V_n(s)) ]`.
Robbins–Monro stepsizes must satisfy `β_n / α_n → 0`.

## Implementation mapping
- **TD target** (`update`): `r_target = reward - (1-α)*β*follow_reward + β*W[next_state]`. Here code symbols `alpha=σ`, `beta=γ` (note naming swap versus paper conventions).
- **Fast update** (`W`): `W[s] += eta_n * (weight_phi * r_target - W[s])`, where `weight_phi = phi_prob / sampling_prob = φ/ν`.
- **Slow update** (`J`): `J[s] += theta_n * (weight_mu * r_target - J[s])`, where `weight_mu = mu_prob / sampling_prob = μ/ν`.
- **Two timescales**: default exponents `eta_exponent=0.51`, `theta_exponent=0.6` ensure `theta_n / eta_n → 0` (slow J, fast W). Steps use Robbins–Monro `initial / (1+t)^exponent`.
- **Support adjustment**: `adjust_support=True` blends ν with μ and φ (`ensure_support`) using `mix_weight` to avoid zero probabilities in importance weights. This is an implementation guard not in the pseudocode but consistent with importance sampling practice.
- **Sampler**: `sampler(state, action)` is user-provided; for continuation reward, a follow-up action is drawn from φ at `next_state`.
- **Reset/terminal handling**: optional `terminal_function`, `reset_on_terminal` for episodic setups. Inventory MDP uses non-terminal loop, so no resets beyond state progression.

## Usage in `InventoryMDP.py`
- `sampling_policy_equil` is passed as ν for all scenarios:
  - State 0: `[1/3, 1/3, 1/3]`
  - State 1: `[0.5, 0.5, 0.0]`
  - State 2: `[1.0, 0.0, 0.0]`
- Scenarios:
  - `(μ*, π*)`: μ = optimal mu*, φ = optimal pi*.
  - `(μ*, π_u)`: μ = mu*, φ = sampling_policy_equil (acts as π_u).
  - `(μ_u, π*)`: μ = sampling_policy_equil (mu_u), φ = pi*.
- `reference_values` uses the analytic `V^{σ,γ}_ρ` from `calculate_qh_value_for_policy`; `reference_diff` tracks ‖W_n - V_ref‖₂ per iter.
- Iterations currently set to 100k in the script; values printed for analytical V and final J estimates.

## Naming correspondences
- Paper σ ↔ code `alpha` (present bias).
- Paper γ ↔ code `beta` (exponential discount).
- Paper V ↔ code `J` (quasi-hyperbolic value). Paper W ↔ code `W` (exponential baseline).
- Paper ν, μ, π ↔ code `sampling_policy`, `mu_policy`, `phi_policy`.

## Compliance assessment
- TD target, importance weights, and dual timescales match Algorithm 1.
- Importance sampling ratios use φ/ν for W and μ/ν for J as in lines 10–11 of the pseudocode.
- Additional `adjust_support` safeguard is a practical extension; setting `adjust_support=False` recovers the exact pseudocode behaviour when ν already covers μ and φ.
- Stepsizes satisfy Robbins–Monro and `θ/η → 0` by default. Custom schedules are supported.

## How to run (Inventory experiment)
```bash
cd /run/media/mwrona/Nowy/Engineering-Thesis
MPLBACKEND=Agg ./.venv/bin/python -m src.experiments.InventoryMDP
```
Outputs: optimal policies, Q-table for (μ*,π*), analytic V for each scenario, final J estimates, and `reproduced_fig1.png` convergence plot.

## Key takeaways
- The implementation is faithful to Algorithm 1 with correct importance weights and two-timescale updates.
- Behaviour policy ν is user-supplied; μ and φ can differ per scenario, and ν can be equal to one of them (as in `(μ*, π_u)` and `(μ_u, π*)`).
- Support adjustment prevents division by zero; disable if exact ratios on the original ν are required.
