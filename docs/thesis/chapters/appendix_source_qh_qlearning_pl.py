"""
Quasi-Hyperbolic Q-Learning Algorithm

Implementation of Q-learning algorithm for Markov Decision Processes
with quasi-hyperbolic discounting for precommitted agents.

Based on:
- Jaskiewicz, A. & Nowak, A.S. (2021). Markov decision processes with quasi-hyperbolic discounting
- Eshwar, S. et al. (2024). Reinforcement learning with quasi-hyperbolic discounting
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

StepSchedule = Callable[[int], float]
TransitionOutcome = Union[
    Tuple[int, float],
    Tuple[int, float, bool],
    Tuple[int, float, bool, Dict[str, object]],
]
TransitionSampler = Callable[[int, int], TransitionOutcome]


class QHQLearning:
    """
    Quasi-Hyperbolic Q-Learning algorithm for precommitted agents.

    This implementation follows the theoretical framework where the agent
    commits to a policy at the beginning and follows it throughout the process.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.8,
        beta: float = 0.95,
        theta_step: float = 0.1,
        eta_step: Optional[float] = None,
        theta_power: float = 0.8,
        eta_power: float = 0.6,
        init_value: float = 0.0,
        # New (policy-evaluation-style) schedule hooks:
        theta_schedule: Optional[StepSchedule] = None,
        eta_schedule: Optional[StepSchedule] = None,
        step_offset: float = 10.0,
        **legacy_kwargs: Any,
    ):
        r"""
        Initialize QH Q-Learning algorithm.

        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions in the MDP
            alpha: Present-bias parameter ($0 <= \alpha <= 1$)
            beta: Exponential discount factor ($0 <= \beta < 1$)
            theta_step: Initial learning rate magnitude for the slow timescale ($\theta_n$)
            eta_step: Initial learning rate magnitude for the fast timescale ($\eta_n$).
                Defaults to ``theta_step`` when not provided.
            theta_power: Exponent for Robbins--Monro schedule of $\theta_n$
                (must be $>0.5$ and greater than ``eta_power`` for two-timescale separation)
            eta_power: Exponent for Robbins--Monro schedule of $\eta_n$ (must be $>0.5$)
            init_value: Initial fill value for both ``W`` and ``Q`` tables.
            theta_schedule: Optional custom callable producing $\theta_n$ from iteration index.
                If provided, overrides (theta_step, theta_power, step_offset) for that schedule.
            eta_schedule: Optional custom callable producing $\eta_n$ from iteration index.
                If provided, overrides (eta_step, eta_power, step_offset) for that schedule.
            step_offset: Offset used in default Robbins--Monro schedule:
                ``step / (offset + t) ** power``.
                Kept at 10.0 to preserve the previous behavior of smaller initial steps.
        """
        alpha_lr_legacy = legacy_kwargs.pop("alpha_lr", None)
        learning_rate_legacy = legacy_kwargs.pop("learning_rate", None)

        legacy_lr = alpha_lr_legacy if alpha_lr_legacy is not None else learning_rate_legacy
        if legacy_lr is not None:
            warnings.warn(
                "Legacy learning-rate arguments are deprecated; use 'theta_step' and 'eta_step'.",
                DeprecationWarning,
                stacklevel=2,
            )
            theta_step = legacy_lr

        if eta_step is None:
            eta_step = theta_step

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        if step_offset <= 0:
            raise ValueError("step_offset must be positive.")
        if theta_power <= 0.5 or eta_power <= 0.5:
            raise ValueError("Robbins-Monro exponents must exceed 0.5 for square-summable schedules")
        if theta_power <= eta_power:
            # Match the intent from policy evaluation: theta decays faster (smaller) than eta.
            warnings.warn(
                "theta_power should be strictly greater than eta_power to ensure two-timescale separation.",
                UserWarning,
                stacklevel=2,
            )

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.theta_step = float(theta_step)
        self.eta_step = float(eta_step)
        self.theta_power = float(theta_power)
        self.eta_power = float(eta_power)
        self.step_offset = float(step_offset)
        self.init_value = float(init_value)
        self._iteration = 0

        # Build schedules like in qh_policy_evaluation.py
        self._eta_schedule = self._build_schedule(
            eta_schedule, base_value=self.eta_step, exponent=self.eta_power, offset=self.step_offset
        )
        self._theta_schedule = self._build_schedule(
            theta_schedule, base_value=self.theta_step, exponent=self.theta_power, offset=self.step_offset
        )

        # Track visits per (s,a) pair for diagnostics / persistence.
        # Step sizes are generated from the GLOBAL iteration counter (see _next_step_sizes).
        self._visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)

        # Initialize Q-functions
        self.W = np.full((n_states, n_actions), self.init_value)  # Auxiliary Q-function W (exponential baseline)
        self.Q = np.full((n_states, n_actions), self.init_value)  # Quasi-hyperbolic Q

    @staticmethod
    def _normalize_available_actions(
        n_actions: int,
        available_actions: Optional[Union[Sequence[int], Callable[[int], Sequence[int]]]],
    ) -> Callable[[int], Sequence[int]]:
        if available_actions is None:

            def _all_actions(_state: int) -> Sequence[int]:
                return range(n_actions)

            return _all_actions

        if callable(available_actions):
            return available_actions

        action_list = list(available_actions)

        def _fixed_actions(_state: int) -> Sequence[int]:
            return action_list

        return _fixed_actions

    @staticmethod
    def _validate_action_indices(actions: Sequence[int], *, n_actions: int) -> None:
        if len(actions) == 0:
            raise ValueError("available_actions(state) returned an empty action set.")
        for action in actions:
            if not 0 <= int(action) < n_actions:
                raise ValueError(f"Invalid action index {action} for n_actions={n_actions}.")

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        done: bool = False,
        available_actions: Optional[Callable[[int], Sequence[int]]] = None,
    ) -> None:
        r"""One step of Algorithm 2 with Robbins--Monro schedules and terminal masking.

        The fast sequence :math:`(\eta_n)` drives the auxiliary baseline ``W`` using
        :math:`W_{n+1}(s,a) = W_n(s,a) + \eta_n [r + \beta \max_{a'} W_n(s', a') - W_n(s,a)]`.
        The slow sequence :math:`(\theta_n)` updates the quasi-hyperbolic value ``Q`` via
        :math:`Q_{n+1}(s,a) = Q_n(s,a) + \theta_n [(1-\alpha) r + \alpha W_{n+1}(s,a) - Q_n(s,a)]`.

        Args:
            state: Current state
            action: Action taken
            reward: Received reward
            next_state: Next state
            done: Whether ``next_state`` is terminal (prevents bootstrapping)
        """
        eta_n, theta_n = self._next_step_sizes(state, action)
        self._update_core(
            state,
            action,
            reward,
            next_state,
            eta_n=eta_n,
            theta_n=theta_n,
            done=done,
            available_actions=available_actions,
        )

    def _update_core(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        eta_n: float,
        theta_n: float,
        done: bool,
        available_actions: Optional[Callable[[int], Sequence[int]]],
    ) -> None:
        """Shared update logic parameterized by (eta_n, theta_n)."""

        # Snapshot W_n(s, a) before the fast update
        w_prev = self.W[int(state), int(action)]

        # Fast timescale (eta_n): exponential baseline W
        if done:
            max_w_next = 0.0
        else:
            if available_actions is None:
                next_actions = list(range(self.n_actions))
            else:
                next_actions = list(available_actions(int(next_state)))
            self._validate_action_indices(next_actions, n_actions=self.n_actions)
            max_w_next = float(np.max(self.W[int(next_state), next_actions]))

        td_error_w = float(reward) + self.beta * max_w_next - float(w_prev)
        w_new = float(w_prev) + float(eta_n) * td_error_w
        self.W[int(state), int(action)] = w_new

        # Slow timescale (theta_n): quasi-hyperbolic Q
        qh_target = (1.0 - self.alpha) * float(reward) + self.alpha * w_new
        td_error_q = qh_target - float(self.Q[int(state), int(action)])
        self.Q[int(state), int(action)] += float(theta_n) * td_error_q

    def _update_fixed_steps(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        eta_n: float,
        theta_n: float,
        done: bool = False,
        available_actions: Optional[Callable[[int], Sequence[int]]] = None,
    ) -> None:
        r"""Update using externally provided (eta_n, theta_n) without advancing n.

        Used by the sweep / generative-model driver to match the pseudocode where
        a single pair (eta_n, theta_n) applies to all (s,a) updates inside one
        sweep iteration n.
        """

        self._record_visit(int(state), int(action))
        self._update_core(
            int(state),
            int(action),
            float(reward),
            int(next_state),
            eta_n=float(eta_n),
            theta_n=float(theta_n),
            done=bool(done),
            available_actions=available_actions,
        )

    def _next_sweep_step_sizes(self) -> tuple[float, float]:
        """Advance the outer sweep counter by one and return (eta_n, theta_n)."""

        self._iteration += 1
        t = self._iteration
        eta_n = float(self._eta_schedule(t))
        theta_n = float(self._theta_schedule(t))
        return eta_n, theta_n

    def _record_visit(self, state: int, action: int) -> None:
        self._visit_counts[int(state), int(action)] += 1

    def _next_step_sizes(self, state: int, action: int) -> tuple[float, float]:
        """Generate the next pair of Robbins--Monro step sizes.

        This matches the step-size semantics in `qh_policy_evaluation.py`:
        a GLOBAL iteration counter drives both schedules.

        Visit counts per (s,a) are still tracked, but they do not influence
        the step sizes.
        """

        self._record_visit(state, action)
        self._iteration += 1
        t = self._iteration

        eta_n = float(self._eta_schedule(t))
        theta_n = float(self._theta_schedule(t))
        return eta_n, theta_n

    @staticmethod
    def _sample_transition(sampler: TransitionSampler, state: int, action: int) -> tuple[int, float, bool]:
        """Normalize a generative-model transition sampler output."""

        outcome = sampler(state, action)
        if not isinstance(outcome, tuple):
            raise TypeError("Transition sampler must return a tuple.")
        if len(outcome) == 2:
            next_state, reward = outcome
            done = False
        elif len(outcome) == 3:
            next_state, reward, done = outcome
        elif len(outcome) == 4:
            next_state, reward, done, _info = outcome
        else:
            raise ValueError("Transition sampler must return 2-4 values.")
        return int(next_state), float(reward), bool(done)

    def get_policy(
        self,
        *,
        available_actions: Optional[Union[Sequence[int], Callable[[int], Sequence[int]]]] = None,
    ) -> np.ndarray:
        """
        Extract the optimal policy from Q-functions.

        Returns:
            Policy array where policy[s] gives the optimal action in state s
        """
        actions_fn = self._normalize_available_actions(self.n_actions, available_actions)
        policy = np.zeros(self.n_states, dtype=int)
        for state in range(self.n_states):
            actions = list(actions_fn(int(state)))
            self._validate_action_indices(actions, n_actions=self.n_actions)
            q_values = self.Q[int(state), actions]
            policy[int(state)] = int(actions[int(np.argmax(q_values))])
        return policy

    def get_value_function(
        self,
        *,
        available_actions: Optional[Union[Sequence[int], Callable[[int], Sequence[int]]]] = None,
    ) -> np.ndarray:
        """
        Extract value function from Q-functions.

        Returns:
            Value function array
        """
        actions_fn = self._normalize_available_actions(self.n_actions, available_actions)
        values = np.zeros(self.n_states, dtype=float)
        for state in range(self.n_states):
            actions = list(actions_fn(int(state)))
            self._validate_action_indices(actions, n_actions=self.n_actions)
            values[int(state)] = float(np.max(self.Q[int(state), actions]))
        return values

    def _build_schedule(
        self, custom_schedule: Optional[StepSchedule], base_value: float, exponent: float, offset: float
    ) -> StepSchedule:
        """Construct a step size schedule, either constant or decaying.

        Args:
            custom_schedule: Optional user-provided schedule function
            base_value: Base value for the schedule
            exponent: Exponent for decay (Robbins--Monro style)
            offset: Offset for decay

        Returns:
            Callable schedule function
        """
        if custom_schedule is not None:
            return custom_schedule

        # Default Robbins--Monro style schedule
        def schedule(iteration: int) -> float:
            return base_value / (offset + iteration) ** exponent

        return schedule

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the agent state."""

        return {
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
            "alpha_bias": float(self.alpha),
            "beta_discount": float(self.beta),
            "theta_step": float(self.theta_step),
            "eta_step": float(self.eta_step),
            "theta_power": float(self.theta_power),
            "eta_power": float(self.eta_power),
            "W": self.W,
            "Q": self.Q,
            "iteration": int(self._iteration),
            "visit_counts": self._visit_counts,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the agent parameters from a snapshot."""
        alpha_bias = state.get("alpha_bias")

        beta_discount = state.get("beta_discount")

        if alpha_bias is None or beta_discount is None:
            raise ValueError("State dict must contain 'alpha_bias' and 'beta_discount'.")

        if "theta_step" in state:
            theta_step = float(state["theta_step"])
        elif "alpha_lr" in state:
            theta_step = float(state["alpha_lr"])
        else:
            theta_step = float(state.get("alpha", self.theta_step))

        eta_step = float(state.get("eta_step", theta_step))

        theta_power = float(state.get("theta_power", 0.8))
        eta_power = float(state.get("eta_power", 0.6))

        self.alpha = float(alpha_bias)
        self.beta = float(beta_discount)
        self.theta_step = theta_step
        self.eta_step = eta_step
        self.theta_power = theta_power
        self.eta_power = eta_power
        # Backward compatibility: older checkpoints may include exploration-related
        # keys (like 'epsilon'); sweep-based training ignores them.

        if "W" not in state or "Q" not in state:
            raise ValueError("State dict must contain 'W' and 'Q' arrays.")
        self.W = np.array(state["W"], copy=True)
        self.Q = np.array(state["Q"], copy=True)
        self._iteration = int(state.get("iteration", 0))

        # Load visit counts with backward compatibility
        if "visit_counts" in state:
            self._visit_counts = np.array(state["visit_counts"], copy=True)
        else:
            # For older saves without visit counts, initialize to zeros
            self._visit_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

    def save(self, path: Path | str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Persist agent parameters to a compressed ``.npz`` file."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = self.state_dict()
        if metadata is not None:
            payload["metadata_json"] = np.array(json.dumps(metadata))

        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        return_metadata: bool = False,
    ) -> "QHQLearning" | tuple["QHQLearning", Optional[Dict[str, Any]]]:
        """Restore an agent from disk.

        Args:
            path: Location of the saved ``.npz`` file.
            return_metadata: When ``True`` the metadata dict is returned
                alongside the agent.

        Returns:
            ``QHQLearning`` instance, optionally accompanied by metadata.
        """

        source = Path(path)
        with np.load(source, allow_pickle=True) as data:
            n_states = int(data["n_states"])
            n_actions = int(data["n_actions"])

            if "alpha_bias" not in data or "beta_discount" not in data:
                raise ValueError("Saved agent is missing required keys 'alpha_bias'/'beta_discount'.")
            alpha_bias = float(data["alpha_bias"])
            beta_discount = float(data["beta_discount"])

            if "theta_step" in data:
                theta_step = float(data["theta_step"])
            elif "alpha_lr" in data:
                theta_step = float(data["alpha_lr"])
            else:
                theta_step = float(data.get("alpha", 0.1))

            eta_step = float(data.get("eta_step", theta_step))
            theta_power = float(data.get("theta_power", 0.8))
            eta_power = float(data.get("eta_power", 0.6))

            agent = cls(
                n_states=n_states,
                n_actions=n_actions,
                alpha=alpha_bias,
                beta=beta_discount,
                theta_step=theta_step,
                eta_step=eta_step,
                theta_power=theta_power,
                eta_power=eta_power,
            )

            if "W" not in data or "Q" not in data:
                raise ValueError("Saved agent is missing required arrays 'W'/'Q'.")
            agent.W = np.array(data["W"], copy=True)
            agent.Q = np.array(data["Q"], copy=True)

            if "iteration" in data:
                agent._iteration = int(data["iteration"])

            # Load visit counts with backward compatibility
            if "visit_counts" in data:
                agent._visit_counts = np.array(data["visit_counts"], copy=True)
            else:
                # For older saves without visit counts, initialize to zeros
                agent._visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)

            metadata = None
            if "metadata_json" in data:
                metadata_json = data["metadata_json"]
                metadata = json.loads(metadata_json.item() if hasattr(metadata_json, "item") else str(metadata_json))

        if return_metadata:
            return agent, metadata
        return agent


def train_qh_qlearning_sweep(
    sampler: TransitionSampler,
    agent: QHQLearning,
    *,
    n_iterations: int,
    states: Optional[Sequence[int]] = None,
    available_actions: Optional[Union[Sequence[int], Callable[[int], Sequence[int]]]] = None,
    actions: Optional[Union[Sequence[int], Callable[[int], Sequence[int]]]] = None,
) -> Dict[str, Any]:
    """Train QH Q-Learning with a generative model via full sweeps over SxA.

    This matches the common pseudocode setting where, for each iteration, the
    algorithm visits every state-action pair and samples a transition
    ``(s', r, done) ~ q(.|s,a)`` from a generative model.

    Args:
        sampler: Callable implementing the generative model.
        agent: QHQLearning agent to update.
        n_iterations: Number of full sweeps.
        states: Optional iterable of state indices (defaults to ``range(agent.n_states)``).
        actions: Either
            - an iterable of action indices used for every state (defaults to ``range(agent.n_actions)``), or
            - a callable ``actions(state) -> iterable[int]`` returning the available actions for that state.

    Returns:
        Dict with learned arrays and basic counters.
    """

    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    state_list = list(range(agent.n_states)) if states is None else list(states)

    if actions is not None and available_actions is not None:
        raise ValueError("Pass only one of 'available_actions' or legacy 'actions'.")

    actions_source = available_actions if available_actions is not None else actions
    available_actions_fn = agent._normalize_available_actions(agent.n_actions, actions_source)

    for _ in range(n_iterations):
        # One (eta_n, theta_n) per sweep iteration n (matches pseudocode).
        eta_n, theta_n = agent._next_sweep_step_sizes()
        for state in state_list:
            state_actions = list(available_actions_fn(int(state)))
            agent._validate_action_indices(state_actions, n_actions=agent.n_actions)
            for action in state_actions:
                next_state, reward, done = agent._sample_transition(sampler, int(state), int(action))
                agent._update_fixed_steps(
                    int(state),
                    int(action),
                    reward,
                    next_state,
                    eta_n=eta_n,
                    theta_n=theta_n,
                    done=done,
                    available_actions=available_actions_fn,
                )

    return {
        "W": agent.W,
        "Q": agent.Q,
        "iteration": int(agent._iteration),
        "visit_counts": agent._visit_counts,
    }
    def _robins_monro_schedule(n: int, a: float, b: float, c: float, kappa: float) -> float:
        return a / ((n + b) ** kappa) + c

    def reset(self) -> None:
        self.Q[:, :] = 0.0

    def train(
        self,
        sampler: Callable[[int, int], Tuple[int, float]],
        *,
        n_sweeps: int = 200_000,
        epsilon: float = 0.1,
        step: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.6),
        record_every: int = 1_000,
    ) -> QLearningRun:
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be > 0")

        Q_history: List[Array] = []

        for sweep in range(n_sweeps):
            step_n = float(self._robins_monro_schedule(sweep, *step))

            for s in range(self.n_states):
                a = self._epsilon_greedy_action(self.rng, self.Q[s], epsilon)
                sp, r = sampler(s, a)

                ap = int(np.argmax(self.Q[int(sp)]))
                spp, rp = sampler(int(sp), ap)

                target = float(r) - (1.0 - self.alpha) * self.beta * float(rp) + self.beta * float(np.max(self.Q[int(sp)]))
                self.Q[s, a] = self.Q[s, a] + step_n * (target - self.Q[s, a])

            if record_every > 0 and (sweep % record_every == 0 or sweep == n_sweeps - 1):
                Q_history.append(self.Q.copy())

        return QLearningRun(Q_history=Q_history)
