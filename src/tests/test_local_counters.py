"""Tests for visit counters and global step sizes in QH Q-Learning.

The implementation tracks local visit counts per (s,a) pair for diagnostics and
persistence, but step sizes follow a GLOBAL Robbins--Monro schedule driven by the
total iteration count (matching `qh_policy_evaluation.py`).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.qh_qlearning import QHQLearning


class TestLocalCounters:
    """Test cases for visit counts and step-size schedule."""
    
    def test_visit_counts_initialization(self):
        """Test that visit counts are initialized to zeros."""
        agent = QHQLearning(n_states=3, n_actions=2, alpha=0.8, beta=0.95)
        
        assert hasattr(agent, '_visit_counts')
        assert agent._visit_counts.shape == (3, 2)
        assert np.all(agent._visit_counts == 0)
    
    def test_visit_counts_increment(self):
        """Test that visit counts increment for the specific (s,a) pair."""
        agent = QHQLearning(n_states=3, n_actions=2, alpha=0.8, beta=0.95)
        
        # Initial state: all counts should be zero
        assert agent._visit_counts[0, 0] == 0
        assert agent._visit_counts[1, 1] == 0
        
        # Update (s=0, a=0) three times
        for _ in range(3):
            agent.update(state=0, action=0, reward=1.0, next_state=1)
        
        # Check that only (0, 0) was incremented
        assert agent._visit_counts[0, 0] == 3
        assert agent._visit_counts[0, 1] == 0
        assert agent._visit_counts[1, 0] == 0
        assert agent._visit_counts[1, 1] == 0
        
        # Update (s=1, a=1) once
        agent.update(state=1, action=1, reward=0.5, next_state=2)
        
        # Check counts
        assert agent._visit_counts[0, 0] == 3
        assert agent._visit_counts[1, 1] == 1
    
    def test_step_sizes_follow_global_iteration(self):
        """Step sizes depend on global iteration, not local visit counts."""
        agent = QHQLearning(
            n_states=3, 
            n_actions=2, 
            alpha=0.8, 
            beta=0.95,
            theta_step=1.0,
            eta_step=1.0,
            theta_power=0.8,
            eta_power=0.6,
            step_offset=10.0,
        )

        # Drive the global iteration counter up using only (0,0)
        for _ in range(100):
            agent.update(state=0, action=0, reward=1.0, next_state=1)

        # Now visit (1,1) for the first time: local count is 1, but global t is large.
        eta_n, theta_n = agent._next_step_sizes(1, 1)

        t = agent._iteration
        expected_eta = 1.0 / ((agent.step_offset + t) ** agent.eta_power)
        expected_theta = 1.0 / ((agent.step_offset + t) ** agent.theta_power)

        assert eta_n == pytest.approx(expected_eta)
        assert theta_n == pytest.approx(expected_theta)
    
    def test_step_size_decay(self):
        """Test that step sizes decay as global iteration increases."""
        agent = QHQLearning(
            n_states=2, 
            n_actions=2, 
            alpha=0.8, 
            beta=0.95,
            theta_step=1.0,
            eta_step=1.0,
            theta_power=0.8,
            eta_power=0.6
        )
        
        # Note: _next_step_sizes increments global iteration.
        eta_1, theta_1 = agent._next_step_sizes(0, 0)
        
        # Visit the same pair multiple times
        for _ in range(9):
            agent._next_step_sizes(0, 0)
        
        # Get step sizes after 10 visits
        eta_10, theta_10 = agent._next_step_sizes(0, 0)
        
        # Step sizes should decrease with more visits
        assert eta_10 < eta_1
        assert theta_10 < theta_1
    
    def test_state_persistence_with_visit_counts(self):
        """Test that visit counts are saved and loaded correctly."""
        agent = QHQLearning(n_states=3, n_actions=2, alpha=0.8, beta=0.95)
        
        # Perform some updates to create non-zero visit counts
        agent.update(state=0, action=0, reward=1.0, next_state=1)
        agent.update(state=0, action=0, reward=1.0, next_state=1)
        agent.update(state=1, action=1, reward=0.5, next_state=2)
        
        # Save state
        state_dict = agent.state_dict()
        
        # Verify visit_counts is in state_dict
        assert 'visit_counts' in state_dict
        assert np.array_equal(state_dict['visit_counts'], agent._visit_counts)
        
        # Create new agent and load state
        new_agent = QHQLearning(n_states=3, n_actions=2, alpha=0.8, beta=0.95)
        new_agent.load_state_dict(state_dict)
        
        # Verify visit counts match
        assert np.array_equal(new_agent._visit_counts, agent._visit_counts)
        assert new_agent._visit_counts[0, 0] == 2
        assert new_agent._visit_counts[1, 1] == 1
    
    def test_backward_compatibility_load(self):
        """Test loading old state dicts without visit_counts."""
        agent = QHQLearning(n_states=3, n_actions=2, alpha=0.8, beta=0.95)
        
        # Create old-style state dict without visit_counts
        old_state = {
            'n_states': 3,
            'n_actions': 2,
            'alpha_bias': 0.8,
            'beta_discount': 0.95,
            'theta_step': 0.1,
            'eta_step': 0.1,
            'theta_power': 0.8,
            'eta_power': 0.6,
            # Historical field from epsilon-greedy variants; ignored by sweep-based code.
            'epsilon': 0.1,
            'W': np.zeros((3, 2)),
            'Q': np.zeros((3, 2)),
            'iteration': 100
        }
        
        # Should load without error and initialize visit_counts to zeros
        agent.load_state_dict(old_state)
        
        assert hasattr(agent, '_visit_counts')
        assert agent._visit_counts.shape == (3, 2)
        assert np.all(agent._visit_counts == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
