"""
MDP Models for Quasi-Hyperbolic Discounting Experiments

This module contains various MDP environments used to test and validate
the quasi-hyperbolic discounting algorithms.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod

class MDPEnvironment(ABC):
    """Abstract base class for MDP environments."""
    
    @abstractmethod
    def reset(self) -> int:
        """Reset environment and return initial state."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Take action and return (next_state, reward, done, info)."""
        pass
    
    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of states in the MDP."""
        pass
    
    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions in the MDP."""
        pass


class InventoryMDP(MDPEnvironment):
    """
    Inventory management MDP.
    
    States: inventory levels (0 to max_inventory)
    Actions: order quantities (0 to max_order)
    
    This is the main application model from the thesis where QH discounting
    leads to time-inconsistent optimal policies.
    """
    
    def __init__(self,
                 max_inventory: int = 20,
                 max_order: int = 10,
                 holding_cost: float = 1.0,
                 ordering_cost: float = 2.0,
                 shortage_cost: float = 5.0,
                 demand_prob: Optional[np.ndarray] = None):
        """
        Initialize inventory MDP.
        
        Args:
            max_inventory: Maximum inventory level
            max_order: Maximum order quantity
            holding_cost: Cost per unit held in inventory
            ordering_cost: Fixed cost per order
            shortage_cost: Cost per unit of unmet demand
            demand_prob: Probability distribution over demand (default: Poisson)
        """
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.shortage_cost = shortage_cost
        
        # Default demand distribution (truncated Poisson)
        if demand_prob is None:
            poisson_param = 3.0
            self.demand_prob = np.array([
                np.exp(-poisson_param) * (poisson_param ** k) / np.math.factorial(k)
                for k in range(max_inventory + 1)
            ])
            self.demand_prob /= self.demand_prob.sum()
        else:
            self.demand_prob = demand_prob
            
        self.state = 0  # Current inventory level
        
    @property
    def n_states(self) -> int:
        return self.max_inventory + 1
    
    @property
    def n_actions(self) -> int:
        return self.max_order + 1
    
    def reset(self) -> int:
        """Reset to random initial inventory level."""
        self.state = np.random.randint(0, self.max_inventory + 1)
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Take ordering action and advance one time step.
        
        Args:
            action: Order quantity (0 to max_order)
            
        Returns:
            (next_state, reward, done, info)
        """
        # Current inventory before ordering
        inventory_before = self.state
        
        # Order arrives immediately (simple model)
        inventory_after_order = min(inventory_before + action, self.max_inventory)
        
        # Sample demand
        demand = np.random.choice(len(self.demand_prob), p=self.demand_prob)
        
        # Meet demand
        sold = min(inventory_after_order, demand)
        shortage = max(0, demand - inventory_after_order)
        inventory_after_demand = inventory_after_order - sold
        
        # Calculate reward (negative costs)
        holding_cost = self.holding_cost * inventory_after_demand
        ordering_cost = self.ordering_cost * action if action > 0 else 0
        shortage_cost = self.shortage_cost * shortage
        
        reward = -holding_cost - ordering_cost - shortage_cost
        
        # Update state
        self.state = inventory_after_demand
        
        # Episode continues indefinitely (done=False)
        return self.state, reward, False, {
            'inventory_before': inventory_before,
            'order_quantity': action,
            'demand': demand,
            'sold': sold,
            'shortage': shortage
        }


class GridWorldMDP(MDPEnvironment):
    """
    Simple GridWorld environment for testing algorithms.
    
    Agent navigates in a grid to reach goal states while avoiding obstacles.
    """
    
    def __init__(self, width: int = 5, height: int = 5, 
                 goal_reward: float = 10.0, step_cost: float = -0.1):
        self.width = width
        self.height = height
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        
        # Define goal states (top-right corner)
        self.goal_states = {(width-1, height-1)}
        
        # Define obstacles (optional - can be extended)
        self.obstacles = set()
        
        self.state = 0  # Linear state index
        
    @property
    def n_states(self) -> int:
        return self.width * self.height
    
    @property
    def n_actions(self) -> int:
        return 4  # up, down, left, right
    
    def _pos_to_state(self, x: int, y: int) -> int:
        """Convert (x, y) position to linear state index."""
        return y * self.width + x
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert linear state index to (x, y) position."""
        return state % self.width, state // self.width
    
    def reset(self) -> int:
        """Reset to random starting position (not goal or obstacle)."""
        while True:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            if (x, y) not in self.goal_states and (x, y) not in self.obstacles:
                self.state = self._pos_to_state(x, y)
                break
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Take movement action.
        
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        x, y = self._state_to_pos(self.state)
        
        # Apply action
        if action == 0 and y > 0:  # up
            y -= 1
        elif action == 1 and y < self.height - 1:  # down
            y += 1
        elif action == 2 and x > 0:  # left
            x -= 1
        elif action == 3 and x < self.width - 1:  # right
            x += 1
        
        # Check for obstacles
        if (x, y) in self.obstacles:
            x, y = self._state_to_pos(self.state)  # Stay in place
        
        new_state = self._pos_to_state(x, y)
        
        # Calculate reward
        if (x, y) in self.goal_states:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_cost
            done = False
        
        self.state = new_state
        
        return self.state, reward, done, {'position': (x, y)}