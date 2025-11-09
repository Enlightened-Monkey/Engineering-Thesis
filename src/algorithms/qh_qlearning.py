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
                 sigma: float = 0.8,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        """
        Initialize QH Q-Learning algorithm.
        
        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions in the MDP  
            sigma: Present-bias parameter (0 <= sigma <= 1)
            gamma: Standard exponential discount factor (0 <= gamma < 1)
            alpha: Learning rate
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize Q-functions
        self.q_exp = np.zeros((n_states, n_actions))  # Exponential Q-function
        self.q_qh = np.zeros((n_states, n_actions))   # Quasi-hyperbolic Q-function
        
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
            return np.argmax(self.q_qh[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-functions using QH discounting rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Received reward
            next_state: Next state
        """
        
        # Update exponential Q-function
        max_q_exp_next = np.max(self.q_exp[next_state])
        td_error_exp = reward + self.gamma * max_q_exp_next - self.q_exp[state, action]
        self.q_exp[state, action] += self.alpha * td_error_exp
        
        # Update quasi-hyperbolic Q-function
        max_q_qh_next = np.max(self.q_qh[next_state])
        qh_target = reward + self.sigma * self.gamma * max_q_qh_next
        td_error_qh = qh_target - self.q_qh[state, action]
        self.q_qh[state, action] += self.alpha * td_error_qh
    
    def get_policy(self) -> np.ndarray:
        """
        Extract the optimal policy from Q-functions.
        
        Returns:
            Policy array where policy[s] gives the optimal action in state s
        """
        return np.argmax(self.q_qh, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        Extract value function from Q-functions.
        
        Returns:
            Value function array
        """
        return np.max(self.q_qh, axis=1)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the agent state."""

        return {
            "n_states": int(self.n_states),
            "n_actions": int(self.n_actions),
            "sigma": float(self.sigma),
            "gamma": float(self.gamma),
            "alpha": float(self.alpha),
            "epsilon": float(self.epsilon),
            "q_exp": self.q_exp,
            "q_qh": self.q_qh,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load the agent parameters from a snapshot."""

        self.sigma = float(state["sigma"])
        self.gamma = float(state["gamma"])
        self.alpha = float(state["alpha"])
        self.epsilon = float(state["epsilon"])

        self.q_exp = np.array(state["q_exp"], copy=True)
        self.q_qh = np.array(state["q_qh"], copy=True)

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
            sigma = float(data["sigma"])
            gamma = float(data["gamma"])
            alpha = float(data["alpha"])
            epsilon = float(data["epsilon"])

            agent = cls(n_states=n_states,
                        n_actions=n_actions,
                        sigma=sigma,
                        gamma=gamma,
                        alpha=alpha,
                        epsilon=epsilon)

            agent.q_exp = np.array(data["q_exp"], copy=True)
            agent.q_qh = np.array(data["q_qh"], copy=True)

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