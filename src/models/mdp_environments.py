"""
MDP Models for Quasi-Hyperbolic Discounting Experiments

This module contains various MDP environments used to test and validate
the quasi-hyperbolic discounting algorithms.
"""

import math
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
                math.exp(-poisson_param) * (poisson_param ** k) / math.factorial(k)
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


class InventoryControlMDP(MDPEnvironment):
    """Finite inventory control problem with explicit cost structure.

    This environment encodes the benchmark described in
    *Reinforcement Learning with Quasi-Hyperbolic Discounting* where:

    - State space: inventory level ``S = {0, …, M}``
    - Action space: order quantity ``A = {0, …, M}``
    - Demand: finite-valued random variable with configurable support
    - Immediate reward: selling revenue minus procurement and holding costs

    The default parameters follow the numerical example from the paper:

    ``M = 2``, procurement cost ``c = 5``, holding cost ``h = 2``,
    selling price ``p = 9``, demand support ``{0, 1, 2}`` with
    probabilities ``(0.2, 0.3, 0.5)``.
    """

    def __init__(self,
                 max_inventory: int = 2,
                 procurement_cost: float = 5.0,
                 holding_cost: float = 2.0,
                 selling_price: float = 9.0,
                 demand_support: Optional[np.ndarray] = None,
                 demand_prob: Optional[np.ndarray] = None,
                 initial_state: int = 0):
        self.max_inventory = int(max_inventory)
        self.procurement_cost = float(procurement_cost)
        self.holding_cost = float(holding_cost)
        self.selling_price = float(selling_price)
        self.initial_state = int(initial_state)

        if demand_support is None:
            self.demand_support = np.arange(self.max_inventory + 1)
        else:
            self.demand_support = np.asarray(demand_support, dtype=int)

        if demand_prob is None:
            if len(self.demand_support) < self.max_inventory + 1:
                raise ValueError(
                    "Default demand probabilities require support {0,…,M}."
                )
            demand_prob = np.array([0.2, 0.3, 0.5], dtype=float)
            if len(self.demand_support) != len(demand_prob):
                # Pad or trim to match support size when M != 2
                demand_prob = np.ones(len(self.demand_support), dtype=float)
        else:
            demand_prob = np.asarray(demand_prob, dtype=float)

        if demand_prob.shape[0] != self.demand_support.shape[0]:
            raise ValueError("Demand support and probabilities must align in length.")

        total_prob = demand_prob.sum()
        if total_prob <= 0:
            raise ValueError("Demand probabilities must sum to a positive value.")

        self.demand_prob = demand_prob / total_prob

        if not 0 <= self.initial_state <= self.max_inventory:
            raise ValueError("Initial state must be within [0, max_inventory].")

        self.state = self.initial_state

    @property
    def n_states(self) -> int:
        return self.max_inventory + 1

    @property
    def n_actions(self) -> int:
        return self.max_inventory + 1

    def reset(self) -> int:
        self.state = self.initial_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if not 0 <= action <= self.max_inventory:
            raise ValueError(f"Action {action} outside admissible range [0, {self.max_inventory}].")

        inventory_pre_order = self.state
        inventory_post_order = min(inventory_pre_order + action, self.max_inventory)

        demand = int(np.random.choice(self.demand_support, p=self.demand_prob))
        sales = min(inventory_post_order, demand)
        next_state = max(inventory_post_order - demand, 0)

        holding_inventory = next_state
        revenue = self.selling_price * sales
        procurement_cost = self.procurement_cost * action
        holding_cost = self.holding_cost * holding_inventory
        reward = revenue - procurement_cost - holding_cost

        self.state = next_state

        return self.state, reward, False, {
            "inventory_pre_order": inventory_pre_order,
            "inventory_post_order": inventory_post_order,
            "demand": demand,
            "sales": sales,
            "holding_inventory": holding_inventory,
            "procurement_cost": procurement_cost,
            "holding_cost": holding_cost,
            "revenue": revenue,
            "immediate_reward": reward,
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


class PoleBalancingMDP(MDPEnvironment):
    """Physics-based pole balancing environment with quasi-hyperbolic MDP interface."""

    def __init__(self,
                 min_x: float = -2.4,
                 max_x: float = 2.4,
                 max_speed: float = 2.5,
                 force_mag: float = 10.0,
                 wind_force_max: float = 3.0,
                 wind_turbulence: float = 0.5,
                 time_step: float = 0.02,
                 max_time: float = 10.0,
                 angle_reward_threshold: float = np.deg2rad(12),
                 angle_failure: float = np.deg2rad(45),
                 length_range: Tuple[float, float] = (0.5, 2.0),
                 mass_per_meter: float = 0.5,
                 n_position_bins: int = 7,
                 n_velocity_bins: int = 7,
                 n_angle_bins: int = 11,
                 n_ang_velocity_bins: int = 7,
                 n_length_bins: int = 3,
                 fall_penalty: float = 10.0,
                 success_bonus: float = 2.0):
        """
        Initialize pole balancing environment.

        Args:
            min_x: Minimum cart position.
            max_x: Maximum cart position.
            max_speed: Maximum cart speed (absolute value).
            force_mag: Magnitude of control force applied to cart.
            wind_force_max: Maximum wind force magnitude (absolute value).
            wind_turbulence: Standard deviation of wind noise per step.
            time_step: Simulation time step in seconds.
            max_time: Maximum episode duration in seconds.
            angle_reward_threshold: Angle within which rewards stay positive.
            angle_failure: Angle at which the pole is considered to have fallen.
            length_range: Tuple specifying min and max pole length.
            mass_per_meter: Mass per meter of pole length.
            n_position_bins: Number of discrete bins for cart position.
            n_velocity_bins: Number of discrete bins for cart velocity.
            n_angle_bins: Number of discrete bins for pole angle.
            n_ang_velocity_bins: Number of discrete bins for pole angular velocity.
            n_length_bins: Number of discrete bins for pole length.
            fall_penalty: Penalty applied when the pole falls.
            success_bonus: Bonus when the agent balances for the full duration.
        """

        self.min_x = min_x
        self.max_x = max_x
        self.max_speed = max_speed
        self.force_mag = force_mag
        self.wind_force_max = wind_force_max
        self.wind_turbulence = wind_turbulence
        self.time_step = time_step
        self.max_time = max_time
        self.angle_reward_threshold = angle_reward_threshold
        self.angle_failure = angle_failure
        self.length_range = length_range
        self.mass_per_meter = mass_per_meter
        self.fall_penalty = fall_penalty
        self.success_bonus = success_bonus

        # Discretization bins
        self.n_position_bins = n_position_bins
        self.n_velocity_bins = n_velocity_bins
        self.n_angle_bins = n_angle_bins
        self.n_ang_velocity_bins = n_ang_velocity_bins
        self.n_length_bins = n_length_bins

        self.position_edges = np.linspace(min_x, max_x, n_position_bins + 1)[1:-1]
        self.velocity_edges = np.linspace(-max_speed, max_speed, n_velocity_bins + 1)[1:-1]
        self.angle_edges = np.linspace(-angle_failure, angle_failure, n_angle_bins + 1)[1:-1]
        self.ang_velocity_max = np.pi * 2  # approximate bound
        self.ang_velocity_edges = np.linspace(-self.ang_velocity_max,
                                              self.ang_velocity_max,
                                              n_ang_velocity_bins + 1)[1:-1]
        self.length_edges = np.linspace(length_range[0], length_range[1], n_length_bins + 1)[1:-1]

        self.bin_sizes = [
            n_position_bins,
            n_velocity_bins,
            n_angle_bins,
            n_ang_velocity_bins,
            n_length_bins
        ]
        self._n_states = int(np.prod(self.bin_sizes))

        # Dynamics state variables
        self.gravity = 9.81
        self.masscart = 1.0

        self._continuous_state = np.zeros(4)
        self._length = float(length_range[0])
        self._length_index = 0
        self._wind_force = 0.0
        self._time_elapsed = 0.0

        self.masspole = self.mass_per_meter * self._length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self._length

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return 3  # push left, noop, push right

    def reset(self) -> int:
        """Reset environment with random initial conditions."""
        x = np.random.uniform(self.min_x * 0.25, self.max_x * 0.25)
        x_dot = np.random.uniform(-0.5, 0.5)
        theta = np.random.uniform(-np.deg2rad(6), np.deg2rad(6))
        theta_dot = np.random.uniform(-0.5, 0.5)

        self._length = np.random.uniform(*self.length_range)
        self._length_index = self._digitize(self._length, self.length_edges, self.n_length_bins)
        self.masspole = self.mass_per_meter * self._length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self._length

        self._continuous_state = np.array([x, x_dot, theta, theta_dot], dtype=float)
        self._time_elapsed = 0.0
        self._wind_force = np.random.uniform(-self.wind_force_max, self.wind_force_max)

        return self._encode_state()

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Advance the simulation by one time step given an action."""
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Action {action} is invalid for PoleBalancingMDP")

        force = {
            0: -self.force_mag,
            1: 0.0,
            2: self.force_mag
        }[action]

        # Wind dynamics
        wind_noise = np.random.normal(0.0, self.wind_turbulence)
        self._wind_force = np.clip(self._wind_force + wind_noise,
                                   -self.wind_force_max,
                                   self.wind_force_max)
        applied_force = force + self._wind_force

        x, x_dot, theta, theta_dot = self._continuous_state

        # Equations of motion (cart-pole dynamics)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        temp = (applied_force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x = x + self.time_step * x_dot
        x_dot = np.clip(x_dot + self.time_step * x_acc, -self.max_speed, self.max_speed)
        theta = theta + self.time_step * theta_dot
        theta_dot = np.clip(theta_dot + self.time_step * theta_acc,
                            -self.ang_velocity_max,
                            self.ang_velocity_max)

        self._continuous_state = np.array([x, x_dot, theta, theta_dot], dtype=float)
        self._time_elapsed += self.time_step

        fell = bool(abs(theta) > self.angle_failure or x < self.min_x or x > self.max_x)
        timed_out = bool(self._time_elapsed >= self.max_time)
        done = fell or timed_out

        reward = self._compute_reward(theta, x, fell, timed_out)

        info = {
            'x': x,
            'x_dot': x_dot,
            'theta': theta,
            'theta_dot': theta_dot,
            'length': self._length,
            'wind_force': self._wind_force,
            'time_elapsed': self._time_elapsed,
            'terminated_reason': 'fall' if fell else ('timeout' if timed_out else None)
        }

        return self._encode_state(), reward, done, info

    def _digitize(self, value: float, edges: np.ndarray, n_bins: int) -> int:
        """Digitize a continuous value into discrete bins."""
        if n_bins == 1:
            return 0
        idx = int(np.digitize([value], edges)[0])
        return int(np.clip(idx, 0, n_bins - 1))

    def _encode_state(self) -> int:
        x, x_dot, theta, theta_dot = self._continuous_state
        indices = [
            self._digitize(x, self.position_edges, self.n_position_bins),
            self._digitize(x_dot, self.velocity_edges, self.n_velocity_bins),
            self._digitize(theta, self.angle_edges, self.n_angle_bins),
            self._digitize(theta_dot, self.ang_velocity_edges, self.n_ang_velocity_bins),
            self._length_index
        ]

        index = indices[0]
        for dim in range(1, len(indices)):
            index = index * self.bin_sizes[dim] + indices[dim]
        return int(index)

    def _compute_reward(self, theta: float, x: float, fell: bool, timed_out: bool) -> float:
        """Compute reward based on pole angle and episode status."""
        angle_error = abs(theta)
        if angle_error <= self.angle_reward_threshold:
            upright_reward = 1.0 - angle_error / max(self.angle_reward_threshold, 1e-6)
        else:
            excess = angle_error - self.angle_reward_threshold
            scale = max(self.angle_failure - self.angle_reward_threshold, 1e-6)
            upright_reward = 1.0 - (self.angle_reward_threshold / max(self.angle_reward_threshold, 1e-6))
            upright_reward -= excess / scale
            upright_reward = max(upright_reward, -1.0)

        position_penalty = 0.1 * (abs(x) / self.max_x) ** 2
        reward = upright_reward - position_penalty

        if fell:
            reward -= self.fall_penalty
        elif timed_out:
            reward += self.success_bonus

        return float(reward)