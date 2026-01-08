"""Torch/CUDA backend for QH policy evaluation (vectorized, model-based).

This is an *optional* backend intended for cases where you know the MDP model:
- transition tensor P with shape [S, A, S]
- expected reward matrix R with shape [S, A]

It implements a deterministic, fully vectorized analogue of the synchronous
sweep driver in `QHPolicyEvaluation.evaluate_policy`, but using expected values
instead of single-sample transitions. This makes it highly parallel and GPU
friendly.

Important:
    This is not a stochastic approximation run (no sampling noise). It is meant
    as a parallel baseline or a fast reference computation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .torch_utils import _require_torch, resolve_device


class QHPolicyEvaluationTorch:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.8,
        beta: float = 0.95,
        theta_step: float = 0.05,
        eta_step: float = 0.2,
        theta_exponent: float = 0.9,
        eta_exponent: float = 0.6,
        device: Optional[str] = None,
        dtype: str = "float32",
    ) -> None:
        torch = _require_torch()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1).")
        if theta_step <= 0 or eta_step <= 0:
            raise ValueError("Stepsizes must be positive.")
        if not 0.5 < theta_exponent <= 1.0 or not 0.5 < eta_exponent <= 1.0:
            raise ValueError("Exponents must belong to (0.5, 1].")
        if theta_exponent <= eta_exponent:
            raise ValueError("Require theta_exponent > eta_exponent.")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.device = resolve_device(device)
        self.dtype = getattr(torch, dtype)

        self.theta_step = float(theta_step)
        self.eta_step = float(eta_step)
        self.theta_exponent = float(theta_exponent)
        self.eta_exponent = float(eta_exponent)
        self._iteration = 0

        # Store values per (state, action) pair for stochastic approximation
        self.W = torch.zeros(self.n_states, self.n_actions, device=self.device, dtype=self.dtype)
        self.J = torch.zeros(self.n_states, self.n_actions, device=self.device, dtype=self.dtype)

    def _stepsizes(self, t: int) -> Tuple[float, float]:
        eta = self.eta_step / ((1.0 + float(t)) ** self.eta_exponent)
        theta = self.theta_step / ((1.0 + float(t)) ** self.theta_exponent)
        return float(eta), float(theta)

    def update(self, state: int, action_sample: int, action_policy: int, reward: float, 
               next_state: int, eta_n: float, theta_n: float) -> None:
        """Apply single-transition stochastic approximation update.
        
        This implements the two-timescale TD updates for policy evaluation:
        - W tracks the continuation value under the policy
        - J tracks the quasi-hyperbolic value starting with the policy action
        
        Args:
            state: Current state index
            action_sample: Action sampled from behavior policy (for visiting state-action pairs)
            action_policy: Action from the evaluated policy
            reward: Observed reward
            next_state: Next state index
            eta_n: Step size for W update (faster timescale)
            theta_n: Step size for J update (slower timescale)
        """
        torch = _require_torch()
        
        # Current values at sampled (s,a)
        w_prev = self.W[state, action_sample]
        j_prev = self.J[state, action_sample]
        
        # Max W over next state actions (for continuation value)
        max_w_next = torch.max(self.W[next_state, :])
        
        # W update: standard beta-discounted TD
        # W(s,a) = E[r + beta * max_a' W(s',a')]
        td_error_w = reward + self.beta * max_w_next - w_prev
        w_new = w_prev + eta_n * td_error_w
        self.W[state, action_sample] = w_new
        
        # J update: quasi-hyperbolic value for the policy action
        # J(s,a) = E[r + alpha*beta*W(s',a_policy) | a_policy ~ policy(s)]
        # We use the policy action at the current state
        w_policy_next = self.W[next_state, action_policy]
        
        qh_target = reward + self.alpha * self.beta * w_policy_next
        td_error_j = qh_target - j_prev
        j_new = j_prev + theta_n * td_error_j
        self.J[state, action_sample] = j_new

    def evaluate_policy_expected_model(
        self,
        P,
        R,
        *,
        mu_policy,
        phi_policy,
        n_iterations: int = 1000,
        reference_values: Optional[np.ndarray] = None,
        reference_kind: str = "W",
    ) -> Dict[str, np.ndarray]:
        """Vectorized policy evaluation using the known model (P, R).

        Args:
            P: transition tensor [S, A, S]
            R: reward matrix [S, A]
            mu_policy: evaluation policy μ [S, A]
            phi_policy: continuation policy φ [S, A]
            n_iterations: number of sweeps
            reference_values: optional reference vector for convergence history
            reference_kind: 'W' or 'J'

        Returns:
            dict with 'W', 'J' and optionally 'reference_diff'. Values are NumPy arrays.
        """

        torch = _require_torch()

        P = torch.as_tensor(P, device=self.device, dtype=self.dtype)
        R = torch.as_tensor(R, device=self.device, dtype=self.dtype)
        mu = torch.as_tensor(mu_policy, device=self.device, dtype=self.dtype)
        phi = torch.as_tensor(phi_policy, device=self.device, dtype=self.dtype)

        if P.ndim != 3 or P.shape[0] != self.n_states or P.shape[2] != self.n_states:
            raise ValueError("P must have shape [S, A, S].")
        if R.shape[0] != self.n_states or R.shape[1] != P.shape[1]:
            raise ValueError("R must have shape [S, A] with same A as P.")
        if mu.shape != R.shape or phi.shape != R.shape:
            raise ValueError("mu_policy and phi_policy must have shape [S, A].")

        # Precompute follow_reward(s') = E_{a'~phi(s')}[R(s',a')]
        follow_reward = torch.sum(phi * R, dim=1)  # [S]

        diff_history = None
        if reference_values is not None:
            if reference_kind not in {"W", "J"}:
                raise ValueError("reference_kind must be either 'W' or 'J'.")
            reference = torch.as_tensor(reference_values, device=self.device, dtype=self.dtype)
            if reference.shape != (self.n_states,):
                raise ValueError("reference_values must match (n_states,).")
            diff_history = torch.zeros(n_iterations, device=self.device, dtype=self.dtype)
        else:
            reference = None

        for t in range(n_iterations):
            eta_n, theta_n = self._stepsizes(self._iteration)
            self._iteration += 1

            # Compute expectations under P for all (s,a):
            #   Ew = sum_{s'} P[s,a,s'] * W[s']
            #   Efr = sum_{s'} P[s,a,s'] * follow_reward[s']
            Ew = torch.einsum("sas,s->sa", P, self.W)
            Efr = torch.einsum("sas,s->sa", P, follow_reward)

            # r_target expectation per (s,a)
            g = R + self.beta * Ew - (1.0 - self.alpha) * self.beta * Efr  # [S,A]

            # Match the exact structure of evaluate_policy():
            # W target uses phi at current state; J target uses mu at current state.
            W_target = torch.sum(phi * g, dim=1)  # [S]
            J_target = torch.sum(mu * g, dim=1)  # [S]

            eta = torch.tensor(eta_n, device=self.device, dtype=self.dtype)
            theta = torch.tensor(theta_n, device=self.device, dtype=self.dtype)

            self.W = self.W + eta * (W_target - self.W)
            self.J = self.J + theta * (J_target - self.J)

            if diff_history is not None and reference is not None:
                current = self.W if reference_kind == "W" else self.J
                diff_history[t] = torch.linalg.norm(current - reference)

        result: Dict[str, np.ndarray] = {
            "W": self.W.detach().cpu().numpy(),
            "J": self.J.detach().cpu().numpy(),
        }
        if diff_history is not None:
            result["reference_diff"] = diff_history.detach().cpu().numpy()
        return result


def analytic_reference_WJ_numpy(
    P: np.ndarray,
    R: np.ndarray,
    *,
    alpha: float,
    beta: float,
    mu_policy: np.ndarray,
    phi_policy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form references consistent with `evaluate_policy_expected_model`.

    W solves: (I - beta P_phi) W = r_phi - (1-alpha) beta P_phi r_phi
    J is:     J = r_mu - (1-alpha) beta P_mu r_phi + beta P_mu W

    Where:
        P_pi[s,s'] = sum_a pi[s,a] P[s,a,s']
        r_pi[s]    = sum_a pi[s,a] R[s,a]
    """

    S, A = R.shape
    if P.shape != (S, A, S):
        raise ValueError("P must have shape [S,A,S].")

    P_phi = np.einsum("sa,sas->ss", phi_policy, P)
    P_mu = np.einsum("sa,sas->ss", mu_policy, P)
    r_phi = np.sum(phi_policy * R, axis=1)
    r_mu = np.sum(mu_policy * R, axis=1)

    bPphi = beta * P_phi
    rhs_W = r_phi - (1.0 - alpha) * beta * (P_phi @ r_phi)
    W = np.linalg.solve(np.eye(S) - bPphi, rhs_W)

    J = r_mu - (1.0 - alpha) * beta * (P_mu @ r_phi) + beta * (P_mu @ W)
    return W, J
