"""Model-free policy evaluation aligned with Algorithm 1 (QH discounting)."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

TransitionOutcome = Union[Tuple[int, float], Tuple[int, float, bool], Tuple[int, float, bool, Dict[str, object]]]
TransitionSampler = Callable[[int, int], TransitionOutcome]
PolicyMatrix = np.ndarray
StepSchedule = Callable[[int], float]
ResetFn = Callable[[], int]
TerminalFn = Callable[[int], bool]


class QHPolicyEvaluation:
    r"""Two-timescale policy evaluation for quasi-hyperbolic discounting.

    This class implements Algorithm 1 from the slides/paper exactly, including the
    importance weighting factors and the TD target

    .. math::

        r_n^{\text{target}}(s) = r(s, a) - (1-\alpha) \beta r(s', a') + \beta W_n(s').

    The fast timescale ``(\eta_n)`` updates the exponential baseline ``W``, while the
    slow timescale ``(\theta_n)`` tracks the quasi-hyperbolic value ``J``. By default
    both stepsizes follow Robbins–Monro schedules such that ``\theta_n / \eta_n → 0``.
    """

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.8,
        beta: float = 0.95,
        theta_step: float = 0.05,
        eta_step: float = 0.2,
        *,
        theta_schedule: Optional[StepSchedule] = None,
        eta_schedule: Optional[StepSchedule] = None,
        theta_exponent: float = 0.9,
        eta_exponent: float = 0.6,
        min_probability: float = 1e-12,
    ) -> None:
        r"""Initialise the evaluator with Robbins–Monro-compliant stepsizes.

        Args:
            n_states: Number of states in the MDP.
            alpha: Present-bias parameter :math:`\alpha \in [0, 1]`.
            beta: Exponential discount factor :math:`\beta \in [0, 1)`.
            theta_step: Initial magnitude for the slow schedule :math:`(\theta_n)`.
            eta_step: Initial magnitude for the fast schedule :math:`(\eta_n)`.
            theta_schedule: Optional custom callable producing ``\theta_n``.
            eta_schedule: Optional custom callable producing ``\eta_n``.
            theta_exponent: Exponent for the default Robbins–Monro schedule of
                ``\theta_n`` (must be in ``(0.5, 1]``). Larger exponents decay faster.
            eta_exponent: Exponent for ``\eta_n`` (also in ``(0.5, 1]``). The defaults
                satisfy ``\theta_n / \eta_n \to 0`` because ``theta_exponent > eta_exponent``.
            min_probability: Numerical floor used in importance weights to avoid
                division by zero when ``\nu(a\mid s)`` is tiny.
        """

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1).")
        if theta_step <= 0 or eta_step <= 0:
            raise ValueError("Stepsizes must be positive.")
        for name, value in {"theta_exponent": theta_exponent, "eta_exponent": eta_exponent}.items():
            if not 0.5 < value <= 1.0:
                raise ValueError(f"{name} must belong to (0.5, 1].")
        if theta_exponent <= eta_exponent:
            raise ValueError("Require theta_exponent > eta_exponent to ensure θ_n/η_n → 0.")
        if min_probability <= 0.0:
            raise ValueError("min_probability must be positive.")

        self.n_states = int(n_states)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._min_probability = float(min_probability)

        self._eta_schedule = self._build_schedule(eta_schedule, eta_step, eta_exponent)
        self._theta_schedule = self._build_schedule(theta_schedule, theta_step, theta_exponent)
        self._iteration = 0

        self.W = np.zeros(self.n_states)
        self.J = np.zeros(self.n_states)

    # ------------------------------------------------------------------
    # Step-size utilities
    # ------------------------------------------------------------------
    @staticmethod
    def robbins_monro_schedule(initial: float, exponent: float = 0.6, offset: float = 1.0) -> StepSchedule:
        """Return a power-law sequence satisfying Robbins–Monro conditions.

        The resulting schedule obeys ``sum_n s_n = ∞``, ``sum_n s_n^2 < ∞`` provided
        ``0.5 < exponent ≤ 1``. The ``offset`` prevents division by zero.
        """

        if initial <= 0:
            raise ValueError("initial must be positive for a Robbins–Monro schedule.")
        if not 0.5 < exponent <= 1.0:
            raise ValueError("exponent must belong to (0.5, 1].")
        if offset <= 0:
            raise ValueError("offset must be positive.")

        def schedule(t: int) -> float:
            return initial / ((offset + float(t)) ** exponent)

        return schedule

    @staticmethod
    def constant_schedule(value: float) -> StepSchedule:
        """Return a constant stepsize helper (does *not* satisfy Robbins–Monro)."""

        if value <= 0:
            raise ValueError("value must be positive.")
        return lambda _t: value

    def _build_schedule(
        self,
        schedule: Optional[StepSchedule],
        base_value: float,
        exponent: float,
    ) -> StepSchedule:
        if schedule is not None:
            return schedule
        return self.robbins_monro_schedule(base_value, exponent=exponent)

    def _next_stepsizes(self) -> Tuple[float, float]:
        eta_n = float(self._eta_schedule(self._iteration))
        theta_n = float(self._theta_schedule(self._iteration))
        self._iteration += 1
        return eta_n, theta_n

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        follow_reward: float,
        sampling_prob: float,
        mu_prob: float,
        phi_prob: float,
    ) -> None:
        r"""Apply one iteration of Algorithm 1 for state ``s``.

        Args:
            state: Current state ``s``.
            action: Action sampled from the behaviour policy ``ν``.
            reward: Reward ``r(s, a)``.
            next_state: Successor state ``s'``.
            follow_reward: Reward ``r(s', a')`` from evaluating action ``a'`` sampled
                from ``φ_s(·|s')``.
            sampling_prob: Behaviour probability ``ν(a|s)``.
            mu_prob: Evaluation probability ``μ(a|s)``.
            phi_prob: Continuation probability ``φ_s(a|s)``.
        """

        eta_n, theta_n = self._next_stepsizes()

        r_target = reward - (1.0 - self.alpha) * self.beta * follow_reward + self.beta * self.W[next_state]

        denom = max(self._min_probability, sampling_prob)
        weight_phi = phi_prob / denom
        weight_mu = mu_prob / denom

        self.W[state] += eta_n * (weight_phi * r_target - self.W[state])
        self.J[state] += theta_n * (weight_mu * r_target - self.J[state])

    # ------------------------------------------------------------------
    # Sampling driver
    # ------------------------------------------------------------------
    def evaluate_policy(
        self,
        sampler: TransitionSampler,
        sampling_policy: PolicyMatrix,
        mu_policy: PolicyMatrix,
        phi_policy: PolicyMatrix,
        *,
        n_iterations: int = 1000,
        initial_state: int = 0,
        rng: Optional[np.random.Generator] = None,
        reference_values: Optional[np.ndarray] = None,
        adjust_support: bool = True,
        support_mix: float = 1e-2,
        reset_on_terminal: bool = True,
        reset_fn: Optional[ResetFn] = None,
        terminal_function: Optional[TerminalFn] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Execute Algorithm 1 with the provided policies and sampler.

        Args:
            sampler: Callable returning ``(next_state, reward)`` for ``(state, action)``.
            sampling_policy: Behaviour policy :math:`ν` as an ``(n_states, n_actions)`` array.
            mu_policy: Evaluation policy :math:`μ` for the first decision.
            phi_policy: Continuation policy :math:`φ_s`.
            n_iterations: Number of stochastic approximation updates to run.
            initial_state: Starting state for the simulation.
            rng: Optional NumPy generator for reproducibility.
            reference_values: Optional exponential value function to monitor
                ``\|W_n - V^β_{φ_s}\|_2`` each iteration.
            adjust_support: When ``True`` ensure ``ν`` covers the support of both
                evaluation policies by injecting their mass where needed.
            support_mix: Mixing coefficient added from evaluation policies into ``ν``.
            reset_on_terminal: Whether to draw a fresh state after reaching a terminal state.
            reset_fn: Optional callable producing the reset state (defaults to ``initial_state``).
            terminal_function: Optional predicate marking terminal states when the sampler
                does not return a ``done`` flag.

        Returns:
            Dictionary containing histories and the final ``W``/``J`` estimates.
        """

        nu = self._validate_policy(sampling_policy, "sampling")
        mu = self._validate_policy(mu_policy, "μ")
        phi = self._validate_policy(phi_policy, "φ_s")
        self._assert_same_shape(nu, mu, phi)

        if adjust_support:
            if not 0.0 < support_mix < 1.0:
                raise ValueError("support_mix must lie in (0, 1).")
            nu = self.ensure_support(nu, mu, mix_weight=support_mix)
            nu = self.ensure_support(nu, phi, mix_weight=support_mix)

        rng = np.random.default_rng() if rng is None else rng
        n_actions = nu.shape[1]
        state = int(initial_state)

        states = np.zeros(n_iterations, dtype=int)
        diff_history = None
        if reference_values is not None:
            reference = np.asarray(reference_values, dtype=float)
            if reference.shape != (self.n_states,):
                raise ValueError("reference_values must match (n_states,).")
            diff_history = np.zeros(n_iterations)

        reset_callable: ResetFn = (reset_fn if reset_fn is not None else lambda: initial_state)

        def is_terminal(state_id: int) -> bool:
            return bool(terminal_function(state_id)) if terminal_function is not None else False

        for t in range(n_iterations):
            states[t] = state
            action = int(rng.choice(n_actions, p=nu[state]))
            next_state, reward, done = self._sample_transition(sampler, state, action)

            follow_reward = 0.0
            if not done:
                follow_action = int(rng.choice(n_actions, p=phi[next_state]))
                _, follow_reward, _ = self._sample_transition(sampler, next_state, follow_action)

            done = done or is_terminal(next_state)

            self.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                follow_reward=follow_reward,
                sampling_prob=nu[state, action],
                mu_prob=mu[state, action],
                phi_prob=phi[state, action],
            )

            if diff_history is not None:
                diff_history[t] = np.linalg.norm(self.W - reference)

            if done and reset_on_terminal:
                state = int(reset_callable())
            else:
                state = next_state

        result: Dict[str, np.ndarray] = {
            "states": states,
            "W": self.W.copy(),
            "J": self.J.copy(),
        }
        if diff_history is not None:
            result["reference_diff"] = diff_history
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_policy(self, policy: PolicyMatrix, name: str) -> PolicyMatrix:
        arr = np.asarray(policy, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != self.n_states:
            raise ValueError(f"{name} policy must have shape (n_states, n_actions).")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} policy must be non-negative.")
        row_sums = arr.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError(f"Each state in {name} policy must have positive mass.")
        arr = arr / row_sums
        return arr

    @staticmethod
    def _assert_same_shape(*policies: PolicyMatrix) -> None:
        shapes = {policy.shape for policy in policies}
        if len(shapes) != 1:
            raise ValueError("All policies must share the same shape.")

    @staticmethod
    def ensure_support(
        base_policy: PolicyMatrix,
        reference_policy: PolicyMatrix,
        *,
        mix_weight: float = 1e-2,
    ) -> PolicyMatrix:
        """Blend ``reference`` into ``base`` to guarantee overlapping support."""

        if not 0.0 < mix_weight < 1.0:
            raise ValueError("mix_weight must lie in (0, 1).")

        adjusted = (1.0 - mix_weight) * base_policy + mix_weight * reference_policy
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        return adjusted

    @staticmethod
    def _sample_transition(
        sampler: TransitionSampler,
        state: int,
        action: int,
    ) -> Tuple[int, float, bool]:
        outcome = sampler(state, action)
        if not isinstance(outcome, tuple):
            raise TypeError("Transition sampler must return a tuple.")

        if len(outcome) == 2:
            next_state, reward = outcome
            done = False
        elif len(outcome) >= 3:
            next_state, reward, done = outcome[:3]
        else:
            raise ValueError("Sampler output must contain at least (next_state, reward).")

        return int(next_state), float(reward), bool(done)

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Return norms useful for monitoring convergence."""

        return {
            "W_norm": float(np.linalg.norm(self.W)),
            "J_norm": float(np.linalg.norm(self.J)),
            "difference_norm": float(np.linalg.norm(self.J - self.W)),
            "iteration": float(self._iteration),
        }

    def reset(self) -> None:
        """Clear value estimates and iteration counter."""

        self.W.fill(0.0)
        self.J.fill(0.0)
        self._iteration = 0