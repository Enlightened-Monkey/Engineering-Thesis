"""Two-state MDP (Fig. 1b) with five stationary policies.

This experiment module provides:
1) Closed-form (matrix) evaluation of the policies under quasi-hyperbolic
   discounting using the simplified Bellman-style equation

   V(s) = E[ r(s,a,s') + (alpha * beta) * V(s') ]

2) Model-free policy evaluation (Algorithm 1) via
   :class:`src.algorithms.qh_policy_evaluation.QHPolicyEvaluation`.

3) QH Q-learning (Algorithm 2) via :class:`src.algorithms.qh_qlearning.QHQLearning`
   trained with the sweep (generative-model) driver.

Nomenclature follows the thesis:
- alpha (α): present-bias parameter, in [0, 1]
- beta  (β): exponential discount factor, in [0, 1)

MDP detail: the reward for (state=2, action=a2) depends on the realised next
state. In particular, when it transitions 2 -> 2 under action a2, the reward is
5 (not 20). For 2 -> 1 under action a2, the reward is 20.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

PolicyDict = Dict[int, np.ndarray]
QTable = Dict[Tuple[int, str], float]


@dataclass(frozen=True)
class TwoStateNinePoliciesMDP:
	"""Two-state MDP with a1/a2 actions and transition-dependent reward."""

	alpha: float = 0.8
	beta: float = 0.95

	def __post_init__(self) -> None:
		if not 0.0 <= self.alpha <= 1.0:
			raise ValueError("alpha must lie in [0, 1].")
		if not 0.0 <= self.beta < 1.0:
			raise ValueError("beta must lie in [0, 1).")

	@property
	def states(self) -> List[int]:
		return [1, 2]

	@property
	def actions(self) -> List[str]:
		return ["a1", "a2"]

	@property
	def transitions(self) -> Dict[Tuple[int, str], Dict[int, float]]:
		# q(s'|s,a)
		return {
			(1, "a1"): {2: 1.0},
			(1, "a2"): {1: 0.5, 2: 0.5},
			(2, "a1"): {1: 1.0},
			(2, "a2"): {1: 0.5, 2: 0.5},
		}

	def reward(self, state: int, action: str, next_state: int) -> float:
		"""Reward r(s,a,s')."""

		state = int(state)
		next_state = int(next_state)
		action = str(action)

		if (state, action) == (2, "a2"):
			return 5.0 if next_state == 2 else 20.0

		rewards_sa = {
			(1, "a1"): 0.0,
			(1, "a2"): 2.0,
			(2, "a1"): 15.0,
			(2, "a2"): 20.0,  # Only used for the non-2->2 branch (handled above)
		}
		return float(rewards_sa[(state, action)])

	# ------------------------------------------------------------------
	# Policies
	# ------------------------------------------------------------------
	def define_policies(self) -> Dict[str, PolicyDict]:
		"""Return a small set of stationary policies.

		Each policy is represented as ``{state: [P(a1|s), P(a2|s)]}``.
		"""

		# Zgodnie z założeniem eksperymentu notebookowego trzymamy tylko:
		# - 4 polityki deterministyczne
		# - 1 politykę mieszaną 50/50 w obu stanach
		return {
			"always_a1": {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
			"always_a2": {1: np.array([0.0, 1.0]), 2: np.array([0.0, 1.0])},
			"a1_in_s1_a2_in_s2": {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
			"a2_in_s1_a1_in_s2": {1: np.array([0.0, 1.0]), 2: np.array([1.0, 0.0])},
			"fifty_fifty_both": {1: np.array([0.5, 0.5]), 2: np.array([0.5, 0.5])},
		}

	@staticmethod
	def policy_to_matrix(policy: PolicyDict) -> np.ndarray:
		"""Convert a policy dict with states {1,2} into a (2,2) matrix."""

		mat = np.zeros((2, 2), dtype=float)
		mat[0] = np.asarray(policy[1], dtype=float)
		mat[1] = np.asarray(policy[2], dtype=float)
		mat /= mat.sum(axis=1, keepdims=True)
		return mat

	# ------------------------------------------------------------------
	# Closed-form evaluation
	# ------------------------------------------------------------------
	def evaluate_policy_closed_form(self, policy: PolicyDict) -> Dict[int, float]:
		"""Evaluate a stationary policy via a 2x2 linear system."""

		pi_1 = np.asarray(policy[1], dtype=float)
		pi_2 = np.asarray(policy[2], dtype=float)
		alpha_beta = self.alpha * self.beta

		# Transition matrix under policy: P[s, s'] with s in {1,2}
		P = np.zeros((2, 2), dtype=float)

		# From state 1
		P[0, 0] = pi_1[1] * 0.5
		P[0, 1] = pi_1[0] * 1.0 + pi_1[1] * 0.5

		# From state 2
		P[1, 0] = pi_2[0] * 1.0 + pi_2[1] * 0.5
		P[1, 1] = pi_2[1] * 0.5

		# Expected immediate reward per state under policy
		r1_a1 = sum(self.transitions[(1, "a1")].get(sp, 0.0) * self.reward(1, "a1", sp) for sp in self.states)
		r1_a2 = sum(self.transitions[(1, "a2")].get(sp, 0.0) * self.reward(1, "a2", sp) for sp in self.states)
		r2_a1 = sum(self.transitions[(2, "a1")].get(sp, 0.0) * self.reward(2, "a1", sp) for sp in self.states)
		r2_a2 = sum(self.transitions[(2, "a2")].get(sp, 0.0) * self.reward(2, "a2", sp) for sp in self.states)

		r1 = float(pi_1[0]) * float(r1_a1) + float(pi_1[1]) * float(r1_a2)
		r2 = float(pi_2[0]) * float(r2_a1) + float(pi_2[1]) * float(r2_a2)

		A = np.eye(2) - alpha_beta * P
		b = np.array([r1, r2], dtype=float)
		V = np.linalg.solve(A, b)

		return {1: float(V[0]), 2: float(V[1])}

	def evaluate_all_policies_closed_form(self) -> Dict[str, QTable]:
		"""Compute Q(s,a) for each policy in :meth:`define_policies` (closed form)."""

		alpha_beta = self.alpha * self.beta
		results: Dict[str, QTable] = {}

		for policy_name, policy in self.define_policies().items():
			V = self.evaluate_policy_closed_form(policy)

			q_values: QTable = {}
			for s in self.states:
				for a in self.actions:
					q_sa = 0.0
					for sp, prob in self.transitions[(s, a)].items():
						q_sa += float(prob) * (self.reward(s, a, sp) + alpha_beta * float(V[sp]))
					q_values[(s, a)] = float(q_sa)

			results[policy_name] = q_values

		return results

	# ------------------------------------------------------------------
	# Sampler for model-free algorithms
	# ------------------------------------------------------------------
	def make_sampler(self, rng: np.random.Generator):
		"""Return sampler(state, action) -> (next_state, reward, done) for 0-indexed states/actions."""

		transitions = self.transitions

		def sampler(state: int, action: int):
			s_mdp = int(state) + 1
			a_label = ["a1", "a2"][int(action)]
			dist = transitions[(s_mdp, a_label)]
			sp_mdp = int(rng.choice(list(dist.keys()), p=list(dist.values())))
			r = float(self.reward(s_mdp, a_label, sp_mdp))
			next_state = sp_mdp - 1
			done = False
			return int(next_state), float(r), bool(done)

		return sampler


def main() -> None:
	import pandas as pd

	from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation
	from src.algorithms.qh_qlearning import QHQLearning, train_qh_qlearning_sweep

	mdp = TwoStateNinePoliciesMDP(alpha=0.8, beta=0.95)

	# ==================== CLOSED-FORM (TABLE) ====================
	print("=" * 80)
	print("CLOSED-FORM EVALUATION: 5 POLICIES")
	print("=" * 80)

	closed_form = mdp.evaluate_all_policies_closed_form()
	data = {
		name: [
			q[(1, "a1")],
			q[(1, "a2")],
			q[(2, "a1")],
			q[(2, "a2")],
		]
		for name, q in closed_form.items()
	}

	df = pd.DataFrame(data, index=["(1, a1)", "(1, a2)", "(2, a1)", "(2, a2)"])
	print(f"\nQ-values under quasi-hyperbolic discounting (alpha={mdp.alpha}, beta={mdp.beta}):")
	print(df.to_string())

	# ==================== POLICY EVALUATION (ALG. 1) ====================
	print("\n" + "=" * 80)
	print("MODEL-FREE POLICY EVALUATION (ALGORITHM 1)")
	print("=" * 80)

	rng = np.random.default_rng(123)
	sampler = mdp.make_sampler(rng)
	evaluator = QHPolicyEvaluation(n_states=2, alpha=mdp.alpha, beta=mdp.beta)

	# Keep this modest for notebook runtime; increase if you want tighter convergence.
	n_sweeps = 2_000_000

	for name, policy in mdp.define_policies().items():
		evaluator.reset()
		pi = mdp.policy_to_matrix(policy)
		res = evaluator.evaluate_policy(
			sampler=sampler,
			sampling_policy=pi,
			mu_policy=pi,
			phi_policy=pi,
			n_iterations=n_sweeps,
		)
		print(f"\nPolicy: {name}")
		print("  W:", np.round(res["W"], 4))
		print("  J:", np.round(res["J"], 4))

	# ==================== QH Q-LEARNING (ALG. 2, SWEEP) ====================
	print("\n" + "=" * 80)
	print("QH Q-LEARNING (ALGORITHM 2, SWEEP TRAINING)")
	print("=" * 80)

	agent = QHQLearning(
		n_states=2,
		n_actions=2,
		alpha=0.5,
		beta=0.9,
		theta_step=0.1,
		eta_step=0.1,
	)

	n_updates = 1_000_000
	updates_per_sweep = agent.n_states * agent.n_actions
	sweeps = max(1, int(n_updates // updates_per_sweep))

	train_qh_qlearning_sweep(sampler=sampler, agent=agent, n_iterations=sweeps)

	policy_q = agent.get_policy()
	policy_w = np.argmax(agent.W, axis=1)

	print(f"\nAfter {sweeps} sweeps:")
	print("Q-table:")
	for s in range(2):
		s_mdp = s + 1
		print(f"  State {s_mdp}: Q(a1)={agent.Q[s, 0]:9.4f}, Q(a2)={agent.Q[s, 1]:9.4f}")

	print("W-table:")
	for s in range(2):
		s_mdp = s + 1
		print(f"  State {s_mdp}: W(a1)={agent.W[s, 0]:9.4f}, W(a2)={agent.W[s, 1]:9.4f}")

	print("\nGreedy policy (first-step, argmax_a Q):", policy_q)
	print("Greedy continuation policy (argmax_a W):", policy_w)


if __name__ == "__main__":
	main()
