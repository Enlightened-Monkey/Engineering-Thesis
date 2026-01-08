"""Algorithms module for quasi-hyperbolic discounting."""

from .qh_qlearning import QHQLearning, train_qh_qlearning_sweep
from .qh_policy_evaluation import QHPolicyEvaluation

# Optional Torch/CUDA backends.
try:  # pragma: no cover
	from .qh_policy_evaluation_torch import QHPolicyEvaluationTorch
	from .qh_qlearning_torch import QHQLearningTorch

	__all__ = [
		"QHQLearning",
		"train_qh_qlearning_sweep",
		"QHPolicyEvaluation",
		"QHPolicyEvaluationTorch",
		"QHQLearningTorch",
	]
except Exception:  # pragma: no cover
	__all__ = ["QHQLearning", "train_qh_qlearning_sweep", "QHPolicyEvaluation"]