"""
Quasi-Hyperbolic Q-Learning Algorithm

Implementation of Q-learning algorithm for Markov Decision Processes 
with quasi-hyperbolic discounting for precommitted agents.

Based on:
- Jaśkiewicz, A. & Nowak, A.S. (2021). Markov decision processes with quasi-hyperbolic discounting
- Eshwar, S. et al. (2024). Reinforcement learning with quasi-hyperbolic discounting
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

class QHQLearning:
    """
    Quasi-Hyperbolic Q-Learning algorithm for precommitted agents.
    
    This implementation follows the theoretical framework where the agent
    commits to a policy at the beginning and follows it throughout the process.
    """
    
    def __init__(self, 
                 n_states: int, 
                 n_actions: int,
                 alpha: float = 0.8,
                 beta: float = 0.95,
                 theta_step: float = 0.1,
                 eta_step: Optional[float] = None,
                 epsilon: float = 0.1,
                 **legacy_kwargs: Any):
        r"""
        Initialize QH Q-Learning algorithm.
        
        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions in the MDP  
            alpha: Present-bias parameter ($0 \leq \alpha \leq 1$)
            beta: Exponential discount factor ($0 \leq \beta < 1$)
            theta_step: Learning rate for the slow timescale ($\theta_n$)
            eta_step: Learning rate for the fast timescale ($\eta_n$). Defaults to \texttt{theta\_step} when not provided.
            epsilon: Exploration rate for epsilon-greedy policy
        """
        sigma_legacy = legacy_kwargs.pop("sigma", None)
        gamma_legacy = legacy_kwargs.pop("gamma", None)
        alpha_lr_legacy = legacy_kwargs.pop("alpha_lr", None)
        learning_rate_legacy = legacy_kwargs.pop("learning_rate", None)

        if sigma_legacy is not None:
            warnings.warn(
                "Parameter 'sigma' is deprecated; use 'alpha' for the present-bias value.",
                DeprecationWarning,
                stacklevel=2,
            )
            alpha = sigma_legacy

        if gamma_legacy is not None:
            warnings.warn(
                "Parameter 'gamma' is deprecated; use 'beta' for the exponential discount factor.",
                DeprecationWarning,
                stacklevel=2,
            )
            beta = gamma_legacy

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

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.theta_step = theta_step
        self.eta_step = eta_step
        self.epsilon = epsilon
        
        # Initialize Q-functions
        self.W = np.zeros((n_states, n_actions))  # Auxiliary Q-function W
        self.Q = np.zeros((n_states, n_actions))  # Quasi-hyperbolic Q-function
        
    def get_action(self, state: int, exploration: bool = True) -> int:
        """
        Get action using epsilon-greedy policy based on QH Q-values.
        
        Args:
            state: Current state
            exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            Selected action
        """
        if exploration and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        r"""One step of Algorithm 2 from the slides.

        The fast sequence :math:`(\eta_n)` drives the auxiliary baseline ``W`` using
        :math:`W_{n+1}(s,a) = W_n(s,a) + \eta_n [r + \beta \max_{a'} W_n(s', a') - W_n(s,a)]`.
        The slow sequence :math:`(\theta_n)` updates the quasi-hyperbolic value ``Q`` via
        :math:`Q_{n+1}(s,a) = Q_n(s,a) + \theta_n [(1-\alpha) r + \alpha W_{n+1}(s,a) - Q_n(s,a)]`.

        Args:
            state: Current state
            action: Action taken
            reward: Received reward
            next_state: Next state
        """
        # Snapshot W_n(s, a) before the fast update
        w_prev = self.W[state, action]

        # Fast timescale (\eta_n): exponential baseline W
        max_w_next = np.max(self.W[next_state])
        td_error_w = reward + self.beta * max_w_next - w_prev
        w_new = w_prev + self.eta_step * td_error_w
        self.W[state, action] = w_new
        
        # Slow timescale (\theta_n): quasi-hyperbolic Q
        qh_target = (1.0 - self.alpha) * reward + self.alpha * w_new
        td_error_q = qh_target - self.Q[state, action]
        self.Q[state, action] += self.theta_step * td_error_q
    
    def get_policy(self) -> np.ndarray:
        """
        Extract the optimal policy from Q-functions.
        
        Returns:
            Policy array where policy[s] gives the optimal action in state s
        """
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        Extract value function from Q-functions.
        
        Returns:
            Value function array
        """
        return np.max(self.Q, axis=1)

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
            "epsilon": float(self.epsilon),
            "W": self.W,
            "Q": self.Q,
            # Backward compatibility payload
            "sigma": float(self.alpha),
            "gamma": float(self.beta),
            "alpha": float(self.theta_step),
            "q_exp": self.W,
            "q_qh": self.Q,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the agent parameters from a snapshot."""
        alpha_bias = state.get("alpha_bias")
        if alpha_bias is None:
            alpha_bias = state.get("sigma")
        if alpha_bias is None and "theta_step" in state and "alpha" in state:
            # New-format save but alias missing; treat alpha as bias only when theta_step present
            alpha_bias = state.get("alpha")

        beta_discount = state.get("beta_discount")
        if beta_discount is None:
            beta_discount = state.get("beta")
        if beta_discount is None:
            beta_discount = state.get("gamma")

        if alpha_bias is None or beta_discount is None:
            raise ValueError("State dict missing required discount parameters.")

        if "theta_step" in state:
            theta_step = float(state["theta_step"])
        elif "alpha_lr" in state:
            theta_step = float(state["alpha_lr"])
        else:
            theta_step = float(state.get("alpha", self.theta_step))

        eta_step = float(state.get("eta_step", theta_step))

        self.alpha = float(alpha_bias)
        self.beta = float(beta_discount)
        self.theta_step = theta_step
        self.eta_step = eta_step
        self.epsilon = float(state["epsilon"])

        self.W = np.array(state.get("W", state.get("q_exp")), copy=True)
        self.Q = np.array(state.get("Q", state.get("q_qh")), copy=True)

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
            epsilon = float(data["epsilon"])

            alpha_bias = data.get("alpha_bias")
            if alpha_bias is None and "sigma" in data:
                alpha_bias = data["sigma"]
            if alpha_bias is None and "alpha" in data and "theta_step" in data:
                alpha_bias = data["alpha"]
            alpha_bias = float(alpha_bias) if alpha_bias is not None else 0.8

            beta_discount = data.get("beta_discount")
            if beta_discount is None and "beta" in data:
                beta_discount = data["beta"]
            if beta_discount is None and "gamma" in data:
                beta_discount = data["gamma"]
            beta_discount = float(beta_discount) if beta_discount is not None else 0.95

            if "theta_step" in data:
                theta_step = float(data["theta_step"])
            elif "alpha_lr" in data:
                theta_step = float(data["alpha_lr"])
            else:
                theta_step = float(data.get("alpha", 0.1))

            eta_step = float(data.get("eta_step", theta_step))

            agent = cls(n_states=n_states,
                        n_actions=n_actions,
                        alpha=alpha_bias,
                        beta=beta_discount,
                        theta_step=theta_step,
                        eta_step=eta_step,
                        epsilon=epsilon)

            agent.W = np.array(data.get("W", data.get("q_exp")), copy=True)
            agent.Q = np.array(data.get("Q", data.get("q_qh")), copy=True)

            metadata = None
            if "metadata_json" in data:
                metadata_json = data["metadata_json"]
                metadata = json.loads(metadata_json.item() if hasattr(metadata_json, "item") else str(metadata_json))

        if return_metadata:
            return agent, metadata
        return agent


def train_qh_qlearning(env, agent: QHQLearning, n_episodes: int = 1000) -> Dict:
    """
    Train QH Q-Learning agent in environment.
    
    Args:
        env: Environment (should have step, reset methods)
        agent: QHQLearning agent
        n_episodes: Number of training episodes
        
    Returns:
        Training statistics
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # Decay exploration rate
        if episode % 100 == 0:
            agent.epsilon *= 0.95
    
    return {
        'episode_rewards': episode_rewards,
        'final_policy': agent.get_policy(),
        'final_values': agent.get_value_function()
    }