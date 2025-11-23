"""
Unit tests for quasi-hyperbolic discounting algorithms.

Run with: python -m pytest src/tests/
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.qh_qlearning import QHQLearning
from algorithms.qh_policy_evaluation import QHPolicyEvaluation
from models.mdp_environments import InventoryMDP, GridWorldMDP, PoleBalancingMDP
from utils.analysis_tools import calculate_qh_return, validate_qh_parameters

class TestQHQLearning:
    """Test cases for QH Q-Learning algorithm."""
    
    def test_initialization(self):
        """Test proper initialization of QH Q-Learning."""
        agent = QHQLearning(n_states=5, n_actions=3, alpha=0.8, beta=0.95)
        
        assert agent.n_states == 5
        assert agent.n_actions == 3
        assert agent.alpha == 0.8
        assert agent.beta == 0.95
        assert agent.W.shape == (5, 3)
        assert agent.Q.shape == (5, 3)
        
    def test_action_selection(self):
        """Test action selection methods."""
        agent = QHQLearning(n_states=3, n_actions=2)
        
        # Test deterministic action selection
        agent.epsilon = 0.0
        agent.Q[0, 1] = 1.0  # Make action 1 optimal for state 0
        action = agent.get_action(0, exploration=False)
        assert action == 1
        
    def test_q_function_update(self):
        """Test Q-function update mechanism."""
        agent = QHQLearning(n_states=3, n_actions=2, alpha=1.0)  # Fast learning
        
        initial_q = agent.Q[0, 0]
        agent.update(state=0, action=0, reward=1.0, next_state=1)
        
        # Q-function should have changed
        assert agent.Q[0, 0] != initial_q
        
    def test_policy_extraction(self):
        """Test policy extraction from Q-functions."""
        agent = QHQLearning(n_states=2, n_actions=3)
        
        # Set up Q-values to make action 2 optimal for both states
        agent.Q[0, 2] = 1.0
        agent.Q[1, 2] = 1.0
        
        policy = agent.get_policy()
        assert len(policy) == 2
        assert policy[0] == 2
        assert policy[1] == 2


class TestPolicyEvaluation:
    """Test cases for QH policy evaluation."""
    
    def test_initialization(self):
        """Test proper initialization."""
        evaluator = QHPolicyEvaluation(n_states=4, alpha=0.7, beta=0.9)
        
        assert evaluator.n_states == 4
        assert evaluator.alpha == 0.7
        assert evaluator.beta == 0.9
        assert len(evaluator.W) == 4
        assert len(evaluator.J) == 4
        
    def test_value_update(self):
        """Test value function updates."""
        evaluator = QHPolicyEvaluation(n_states=3, theta_step=1.0, eta_step=1.0)
        
        initial_j = evaluator.J[0]
        evaluator.update(
            state=0,
            action=0,
            reward=1.0,
            next_state=1,
            follow_reward=0.5,
            sampling_prob=1.0,
            mu_prob=1.0,
            phi_prob=1.0,
        )
        
        # Value function should have changed
        assert evaluator.J[0] != initial_j

    def test_evaluate_policy_loop(self):
        """Ensure the full Algorithm 1 driver runs without errors."""

        evaluator = QHPolicyEvaluation(n_states=2, alpha=0.5, beta=0.9)

        transitions = {
            (0, 0): (0, 0.0),
            (0, 1): (1, 1.0),
            (1, 0): (0, 0.5),
            (1, 1): (1, 1.5),
        }

        def sampler(state: int, action: int):
            return transitions[(state, action)]

        policy = np.full((2, 2), 0.5)

        result = evaluator.evaluate_policy(
            sampler=sampler,
            sampling_policy=policy,
            mu_policy=policy,
            phi_policy=policy,
            n_iterations=10,
        )

        assert result['W'].shape == (2,)
        assert result['J'].shape == (2,)
        assert len(result['states']) == 10


class TestMDPEnvironments:
    """Test cases for MDP environments."""
    
    def test_inventory_mdp(self):
        """Test inventory MDP environment."""
        env = InventoryMDP(max_inventory=10, max_order=5)
        
        assert env.n_states == 11  # 0 to 10
        assert env.n_actions == 6   # 0 to 5
        
        # Test reset
        initial_state = env.reset()
        assert 0 <= initial_state <= 10
        
        # Test step
        next_state, reward, done, info = env.step(action=2)
        assert 0 <= next_state <= 10
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    def test_gridworld_mdp(self):
        """Test GridWorld MDP environment."""
        env = GridWorldMDP(width=4, height=3)
        
        assert env.n_states == 12  # 4 × 3
        assert env.n_actions == 4  # up, down, left, right
        
        # Test state/position conversion
        assert env._pos_to_state(0, 0) == 0
        assert env._pos_to_state(3, 2) == 11
        assert env._state_to_pos(0) == (0, 0)
        assert env._state_to_pos(11) == (3, 2)

    def test_pole_balancing_mdp(self):
        """Test pole balancing MDP environment dynamics and discretization."""
        env = PoleBalancingMDP()

        expected_states = (
            env.n_position_bins
            * env.n_velocity_bins
            * env.n_angle_bins
            * env.n_ang_velocity_bins
            * env.n_length_bins
        )

        assert env.n_states == expected_states
        assert env.n_actions == 3

        state = env.reset()
        assert 0 <= state < env.n_states

        next_state, reward, done, info = env.step(1)

        assert 0 <= next_state < env.n_states
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'theta' in info
        assert 'wind_force' in info


class TestAnalysisTools:
    """Test cases for analysis utilities."""
    
    def test_qh_return_calculation(self):
        """Test QH return calculation."""
        rewards = [1.0, 2.0, 3.0]
        sigma = 0.8
        gamma = 0.9
        
        qh_return = calculate_qh_return(rewards, sigma, gamma)
        
        # Manual calculation: 1.0 + 0.8 * (0.9 * 2.0 + 0.9^2 * 3.0)
        expected = 1.0 + 0.8 * (0.9 * 2.0 + 0.81 * 3.0)
        assert abs(qh_return - expected) < 1e-10
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        assert validate_qh_parameters(0.8, 0.95) == True
        assert validate_qh_parameters(1.2, 0.95) == False  # sigma > 1
        assert validate_qh_parameters(0.8, 1.1) == False   # gamma >= 1
        assert validate_qh_parameters(-0.1, 0.95) == False # sigma < 0
        
    def test_empty_rewards_qh_return(self):
        """Test QH return calculation with empty rewards."""
        qh_return = calculate_qh_return([], 0.8, 0.9)
        assert qh_return == 0.0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_agent_environment_interaction(self):
        """Test agent training in environment."""
        env = GridWorldMDP(width=3, height=3)
        agent = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.8,
            beta=0.95,
            epsilon=0.1
        )
        
        # Run a few training steps
        state = env.reset()
        for _ in range(10):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            
            if done:
                state = env.reset()
            else:
                state = next_state
        
        # Agent should have learned something (Q-values changed)
        assert np.any(agent.Q != 0)
        
    def test_sigma_parameter_effects(self):
        """Test that different sigma values produce different behaviors."""
        env = GridWorldMDP(width=3, height=3)
        
        # Create agents with different sigma values
        agent1 = QHQLearning(env.n_states, env.n_actions, alpha=1.0)  # No bias
        agent2 = QHQLearning(env.n_states, env.n_actions, alpha=0.5)  # Strong bias
        
        # Train both agents briefly
        for agent in [agent1, agent2]:
            state = env.reset()
            for _ in range(50):
                action = agent.get_action(state, exploration=False)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state if not done else env.reset()
        
        # Agents should develop different policies (at least potentially)
        # This is a weak test since we can't guarantee different policies
        # with such short training, but it checks the mechanism works
        policy1 = agent1.get_policy()
        policy2 = agent2.get_policy()
        
        assert len(policy1) == len(policy2) == env.n_states


if __name__ == "__main__":
    pytest.main([__file__])