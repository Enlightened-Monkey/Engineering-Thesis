import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_policy_eval_torch_matches_closed_form_cpu():
    from src.algorithms.qh_policy_evaluation_torch import (
        QHPolicyEvaluationTorch,
        analytic_reference_WJ_numpy,
    )

    # Tiny MDP: S=2, A=2
    S, A = 2, 2
    P = np.zeros((S, A, S), dtype=float)
    # action 0: stay
    P[:, 0, :] = np.array([[1.0, 0.0], [0.0, 1.0]])
    # action 1: toggle
    P[:, 1, :] = np.array([[0.0, 1.0], [1.0, 0.0]])

    R = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    alpha = 0.35
    beta = 0.95

    # Simple stochastic policies
    mu = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    phi = np.array([[0.4, 0.6], [0.9, 0.1]], dtype=float)

    W_ref, J_ref = analytic_reference_WJ_numpy(P, R, alpha=alpha, beta=beta, mu_policy=mu, phi_policy=phi)

    ev = QHPolicyEvaluationTorch(
        n_states=S,
        alpha=alpha,
        beta=beta,
        # fairly aggressive steps to converge fast
        eta_step=0.8,
        theta_step=0.2,
        eta_exponent=0.6,
        theta_exponent=0.9,
        device="cpu",
    )

    out = ev.evaluate_policy_expected_model(P, R, mu_policy=mu, phi_policy=phi, n_iterations=2000)
    assert out["W"].shape == (S,)
    assert out["J"].shape == (S,)

    # Loose tolerance: schedule is diminishing, so convergence is asymptotic.
    assert np.linalg.norm(out["W"] - W_ref) < 1e-2
    assert np.linalg.norm(out["J"] - J_ref) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_policy_eval_torch_runs_on_cuda():
    from src.algorithms.qh_policy_evaluation_torch import QHPolicyEvaluationTorch

    S, A = 2, 2
    P = np.zeros((S, A, S), dtype=float)
    P[:, 0, :] = np.array([[1.0, 0.0], [0.0, 1.0]])
    P[:, 1, :] = np.array([[0.0, 1.0], [1.0, 0.0]])
    R = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    mu = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    phi = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    ev = QHPolicyEvaluationTorch(n_states=S, alpha=0.5, beta=0.9, device="cuda")
    out = ev.evaluate_policy_expected_model(P, R, mu_policy=mu, phi_policy=phi, n_iterations=10)
    assert out["W"].shape == (S,)
    assert out["J"].shape == (S,)
