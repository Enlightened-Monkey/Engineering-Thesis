"""Torch/CUDA backend for quasi-hyperbolic Q-learning.

This is an *optional* backend. It is useful when you can generate transitions in
batches and want the core table updates (W, Q) to run on GPU.

It does NOT try to batch a scalar `sampler(state, action)` for you. Instead,
use `update_batch()` with tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .torch_utils import _require_torch, resolve_device

StepSchedule = Callable[[int], float]


@dataclass
class TorchSchedules:
    eta_schedule: StepSchedule
    theta_schedule: StepSchedule


class QHQLearningTorch:
    """Quasi-Hyperbolic Q-learning with Torch tensors (CPU/CUDA).

    Mirrors the update equations from `src/algorithms/qh_qlearning.py`, but runs
    them in parallel for a batch of transitions.

    Update equations (per sample i):
        W[s,a] <- W[s,a] + eta_n * (r + beta * max_a' W[s',a'] - W[s,a])
        Q[s,a] <- Q[s,a] + theta_n * ((1-alpha)*r + alpha*W_new - Q[s,a])

    Notes:
        - Step sizes are taken from a GLOBAL iteration counter incremented once
          per `update_batch()` call.
        - If you need per-(s,a) visit-based schedules, keep using the NumPy
          implementation.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.8,
        beta: float = 0.95,
        theta_step: float = 0.1,
        eta_step: Optional[float] = None,
        theta_power: float = 0.8,
        eta_power: float = 0.6,
        step_offset: float = 10.0,
        init_value: float = 0.0,
        device: Optional[str] = None,
        dtype: str = "float32",
        theta_schedule: Optional[StepSchedule] = None,
        eta_schedule: Optional[StepSchedule] = None,
    ) -> None:
        torch = _require_torch()

        if eta_step is None:
            eta_step = theta_step

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        if step_offset <= 0:
            raise ValueError("step_offset must be positive")
        if theta_power <= 0.5 or eta_power <= 0.5:
            raise ValueError("Robbins-Monro exponents must exceed 0.5")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._iteration = 0

        self.device = resolve_device(device)
        self.dtype = getattr(torch, dtype)

        self._schedules = TorchSchedules(
            eta_schedule=self._build_schedule(
                eta_schedule, base_value=float(eta_step), exponent=float(eta_power), offset=float(step_offset)
            ),
            theta_schedule=self._build_schedule(
                theta_schedule,
                base_value=float(theta_step),
                exponent=float(theta_power),
                offset=float(step_offset),
            ),
        )

        fill = torch.tensor(float(init_value), device=self.device, dtype=self.dtype)
        self.W = fill.repeat(self.n_states, self.n_actions).reshape(self.n_states, self.n_actions).clone()
        self.Q = fill.repeat(self.n_states, self.n_actions).reshape(self.n_states, self.n_actions).clone()
        
        # Cache for optimization
        self._cached_mask = None
        self._cached_neg_inf = None

    @staticmethod
    def _robbins_monro_schedule(initial: float, exponent: float, offset: float) -> StepSchedule:
        def schedule(t: int) -> float:
            return initial / ((offset + float(t)) ** exponent)

        return schedule

    def _build_schedule(
        self,
        schedule: Optional[StepSchedule],
        *,
        base_value: float,
        exponent: float,
        offset: float,
    ) -> StepSchedule:
        if schedule is not None:
            return schedule
        return self._robbins_monro_schedule(base_value, exponent=exponent, offset=offset)

    def _next_stepsizes(self) -> Tuple[float, float]:
        eta_n = float(self._schedules.eta_schedule(self._iteration))
        theta_n = float(self._schedules.theta_schedule(self._iteration))
        self._iteration += 1
        return eta_n, theta_n

    def update_batch(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones=None,
        *,
        available_actions_mask=None,
    ) -> Tuple[float, float]:
        """Apply one batched update (optimized for performance).

        Args:
            states: int tensor [B]
            actions: int tensor [B]
            rewards: float tensor [B]
            next_states: int tensor [B]
            dones: optional bool tensor [B]
            available_actions_mask: optional bool tensor [S, A] indicating which
                actions are valid at each next_state. Invalid actions are ignored
                in the max over a'.

        Returns:
            (eta_n, theta_n) used for this batch.
        
        Optimizations:
            - Minimizes tensor allocations by reusing scalars
            - Uses in-place operations where safe
            - Caches frequently accessed tensors (mask, neg_inf)
            - Optimized tensor conversions
        """

        torch = _require_torch()
        eta_n, theta_n = self._next_stepsizes()
        
        # Optimization: Reuse scalars instead of creating new tensors each time
        eta_scalar = float(eta_n)
        theta_scalar = float(theta_n)

        # Optimization: More efficient tensor conversion using torch.as_tensor (no copy if possible)
        # and explicit dtype/device specification
        states = torch.as_tensor(states, device=self.device, dtype=torch.long)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=self.dtype)
        next_states = torch.as_tensor(next_states, device=self.device, dtype=torch.long)
        
        if dones is None:
            dones = torch.zeros(len(states), dtype=torch.bool, device=self.device)
        else:
            dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool)

        # W[s,a] and Q[s,a] for the sampled pairs
        w_prev = self.W[states, actions]
        q_prev = self.Q[states, actions]

        # Compute max_a' W[s',a'] with optional feasibility mask
        w_next_all = self.W[next_states]  # [B, A]

        # Optimization: Cache mask and neg_inf as instance variables if mask is provided
        if available_actions_mask is not None:
            # Convert mask once and cache it
            if not hasattr(self, '_cached_mask') or self._cached_mask is None:
                self._cached_mask = torch.as_tensor(available_actions_mask, device=self.device, dtype=torch.bool)
            if not hasattr(self, '_cached_neg_inf') or self._cached_neg_inf is None:
                self._cached_neg_inf = torch.tensor(-1.0e30, device=self.device, dtype=self.dtype)
            
            mask_next = self._cached_mask[next_states]  # [B, A]
            w_next_all = torch.where(mask_next, w_next_all, self._cached_neg_inf)

        max_w_next = torch.max(w_next_all, dim=1).values
        # Optimization: Use scalar 0.0 instead of torch.zeros_like
        max_w_next = torch.where(dones, torch.tensor(0.0, device=self.device, dtype=self.dtype), max_w_next)

        # Optimization: Compute updates with scalar multiplication (faster than tensor ops)
        td_error_w = rewards + self.beta * max_w_next - w_prev
        w_new = w_prev + eta_scalar * td_error_w

        # Commit W updates (in-place)
        self.W[states, actions] = w_new

        # Optimization: Use scalar operations for quasi-hyperbolic target
        qh_target = (1.0 - self.alpha) * rewards + self.alpha * w_new
        td_error_q = qh_target - q_prev
        q_new = q_prev + theta_scalar * td_error_q

        # Commit Q updates (in-place)
        self.Q[states, actions] = q_new
        return eta_n, theta_n

    def clear_cache(self) -> None:
        """Clear cached tensors to free memory.
        
        Call this if you need to change the action mask or free GPU memory.
        The cache will be rebuilt on the next update_batch call if needed.
        """
        self._cached_mask = None
        self._cached_neg_inf = None

    def update(self, state: int, action: int, reward: float, next_state: int, eta_n: float, theta_n: float) -> None:
        """Apply single-transition update with external step sizes.
        
        This is a convenience wrapper for single transitions when you want to
        provide your own step sizes (e.g., using local visit counters).
        
        Args:
            state: Current state index
            action: Action taken
            reward: Observed reward
            next_state: Next state index
            eta_n: Step size for W update
            theta_n: Step size for Q update
        """
        torch = _require_torch()
        
        # Convert to tensors
        s = torch.tensor([state], device=self.device, dtype=torch.long)
        a = torch.tensor([action], device=self.device, dtype=torch.long)
        r = torch.tensor([reward], device=self.device, dtype=self.dtype)
        s_next = torch.tensor([next_state], device=self.device, dtype=torch.long)
        
        # Current values
        w_prev = self.W[state, action]
        q_prev = self.Q[state, action]
        
        # Max W over next state actions
        max_w_next = torch.max(self.W[next_state, :])
        
        # W update (beta-discounted)
        td_error_w = r[0] + self.beta * max_w_next - w_prev
        w_new = w_prev + eta_n * td_error_w
        self.W[state, action] = w_new
        
        # Q update (quasi-hyperbolic)
        qh_target = (1.0 - self.alpha) * r[0] + self.alpha * w_new
        td_error_q = qh_target - q_prev
        q_new = q_prev + theta_n * td_error_q
        self.Q[state, action] = q_new
