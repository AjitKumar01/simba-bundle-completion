"""Unit and Theoretical Verification Suite for SIMBA Operators (`simba/tests/test_audit.py`).

Verifies all 7 audit & review criteria:
1. Exact row-stochasticity of `SimplicialGraphOperator.P_rw` (every active row sums to 1.0, zero diagonal).
2. Pure Simplicial Incidence Surplus (`score_simplicial_incidence_only`):
   - Strictly 0.0 on single-seed queries (|S| = 1).
   - Distinguishes two hypergraphs with identical pairwise co-occurrence C = B^T B.
   - Unweighted concordance `score_concordance_e2` satisfies R(w; tau) = [A^2 / (A + tau)] * (1 - sum_s p_s^2)
     and is jointly degree-1 homogeneous in (w, tau): R(lambda * w; lambda * tau) = lambda * R(w; tau).
3. Redundancy-Corrected Quadratic Concordance (`score_rao_concordance`, Rao 1982):
   - Vanishes identically (R_{-1} == 0) when query seeds are mutually redundant (three identical shirts, G_{s,s'}=1).
   - Strictly positive when query seeds are complementary/orthogonal (shirt + trousers + shoes, G_{s,s'}=0).
4. Exact Conjugate Normal-Normal Posterior (`conjugate_posterior` / `apply_eb_shrinkage`) & Constrained OLS Variance Estimator (`fit_variance_components`):
   - Matches the claimed posterior theta_hat = (1 - u_c) * Y + u_c * mu_0 to machine precision on the
     reviewer's numerical test: Cold neg prior -> -1.00, Cold pos prior -> 1.00, Warm (f=9, kappa_f=4) -> 3.08.
   - Recovers raw unrounded OLS estimates (`raw_sigma0_sq`, `raw_sigma_eps_sq`) and truncated plug-in `kappa_f`
     invariantly across widely varying spreads of metadata prior means mu_0.
   - Explicitly reports `identifiable=False` (with `raw_sigma0_sq=None, raw_sigma_eps_sq=None, kappa_f=0.0`)
     when all warm items have equal frequency (Var(1/f_c) == 0).
5. Strict Test-Batch Invariance:
   - Using validation-frozen `prior_scale`, adding unrelated queries to a test batch produces zero change
     (0.000000 difference) in the original query's scores.
6. Validation-tolerance backward elimination (`prune_weights_val`) removes redundant optional channels.
7. Every evaluator excludes positive shrinkage candidates when variance components are non-identifiable.
"""
import ast
import os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simba.operators import (
    SimplicialGraphOperator,
    DualSpectrumOntology,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)


def test_row_stochastic_diffusion():
    train_bundles = [
        [0, 1, 2],
        [1, 2, 3],
        [0, 2, 4],
        [2, 3, 4, 5],
    ]
    n_items = 7  # item 6 is cold (freq == 0)
    op = SimplicialGraphOperator(train_bundles, n_items=n_items, svd_dim=4)
    P_rw = op.P_rw.toarray()
    assert np.allclose(np.diag(P_rw), 0.0), "P_rw must have zero diagonal"
    row_sums = P_rw.sum(axis=1)
    assert np.allclose(row_sums[:6], 1.0, atol=1e-6), f"Warm rows must sum to 1.0, got {row_sums[:6]}"
    assert np.isclose(row_sums[6], 0.0), f"Cold item row must sum to 0.0, got {row_sums[6]}"
    ones_warm = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
    prop = op.propagate_rw(ones_warm)
    assert np.allclose(prop[:6], 1.0, atol=1e-6)


def test_pure_incidence_surplus_and_joint_homogeneity():
    train_bundles = [
        [0, 1, 2],
        [3, 4],
        [4, 5],
        [3, 5],
    ]
    op = SimplicialGraphOperator(train_bundles, n_items=6, svd_dim=3, gamma_vol=0.0)
    delta_single, _ = op.score_simplicial_incidence_only([[0]], beta_clique=2.0)
    assert np.allclose(delta_single, 0.0, atol=1e-7), "Delta_inc must be 0.0 on single-seed queries"

    delta_pair, _ = op.score_simplicial_incidence_only([[0, 1], [3, 4]], beta_clique=2.0)
    assert delta_pair[2, 0] > 0.0, "True 3-simplex [0, 1, 2] must have positive incidence surplus"
    assert np.isclose(delta_pair[5, 1], 0.0, atol=1e-7), "Pairwise triangle [3,4],[4,5],[3,5] must have zero incidence surplus"

    tau = 0.05
    r_e2 = op.score_concordance_e2([[0, 1]], tau=tau)[:, 0]
    w_s = op.W_pair[:, [0, 1]].toarray()
    A = w_s.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_s = np.where(A[:, None] > 0, w_s / A[:, None], 0.0)
        gini_simpson = np.where(A > 0, (A ** 2 / (A + tau)) * (1.0 - np.square(p_s).sum(axis=1)), 0.0)
    assert np.allclose(r_e2, gini_simpson, atol=1e-6)

    w_test = np.array([0.01, 0.01], dtype=np.float64)
    A1 = w_test.sum(); p1 = w_test / A1
    R1 = (A1 ** 2 / (A1 + tau)) * (1.0 - np.square(p1).sum())
    A2 = (2.0 * w_test).sum(); p2 = (2.0 * w_test) / A2
    R2_joint = (A2 ** 2 / (A2 + 2.0 * tau)) * (1.0 - np.square(p2).sum())
    assert np.isclose(R2_joint, 2.0 * R1, atol=1e-12)


def test_rao_concordance_redundancy_vs_complementarity():
    train_bundles = [
        [0, 5], [1, 5], [2, 5],
        [0, 6], [3, 6], [4, 6],
    ]
    op = SimplicialGraphOperator(train_bundles, n_items=7, svd_dim=3)
    cat_lbl = np.array([0, 0, 0, 1, 2, 3, 3], dtype=int)
    dense_feat = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)

    rao = op.score_rao_concordance(
        [[0, 1, 2], [0, 3, 4]], dense_feat=dense_feat, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.0
    )
    assert np.isclose(rao[5, 0], 0.0, atol=1e-7), f"Redundant seeds must get 0 Rao concordance, got {rao[5, 0]}"
    assert rao[6, 1] > 0.0, f"Complementary seeds must get positive Rao concordance, got {rao[6, 1]}"


def test_exact_conjugate_posterior_and_variance_identification():
    Y_ex = np.array([-2.0, -2.0, 4.0], dtype=np.float32)
    mu0_ex = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    freq_ex = np.array([0.0, 0.0, 9.0], dtype=np.float32)
    post_np = EmpiricalBayesRouter.apply_eb_shrinkage(Y_ex, mu0_ex, freq_ex, kappa_f=4.0)
    assert np.allclose(post_np, [-1.0, 1.0, 40.0 / 13.0], atol=1e-6), f"Got {post_np}"
    assert np.allclose(np.round(post_np, 2), [-1.00, 1.00, 3.08])

    post_pt = EmpiricalBayesRouter.apply_eb_shrinkage(
        torch.tensor(Y_ex), torch.tensor(mu0_ex), torch.tensor(freq_ex), kappa_f=4.0
    ).numpy()
    assert np.allclose(post_pt, [-1.0, 1.0, 40.0 / 13.0], atol=1e-6)

    rng = np.random.default_rng(42)
    N_sim, Q_sim = 2000, 200
    f_sim = rng.choice([1, 2, 4, 8, 16, 32, 64], size=N_sim).astype(np.float64)
    eta_sim = rng.normal(0.0, 1.0, size=(N_sim, Q_sim))
    eps_sim = rng.normal(0.0, np.sqrt(4.0 / f_sim)[:, None], size=(N_sim, Q_sim))

    mu0_narrow = rng.normal(0.0, 0.2, size=(N_sim, Q_sim))
    Y_narrow = mu0_narrow + eta_sim + eps_sim
    est_narrow = EmpiricalBayesRouter.fit_variance_components(f_sim, Y_narrow, mu0_narrow)

    mu0_wide = rng.normal(0.0, 50.0, size=(N_sim, Q_sim))
    Y_wide = mu0_wide + eta_sim + eps_sim
    est_wide = EmpiricalBayesRouter.fit_variance_components(f_sim, Y_wide, mu0_wide)

    assert est_narrow["identifiable"] is True and est_wide["identifiable"] is True
    assert np.isclose(est_narrow["raw_sigma0_sq"], est_wide["raw_sigma0_sq"], atol=1e-10)
    assert np.isclose(est_narrow["raw_sigma_eps_sq"], est_wide["raw_sigma_eps_sq"], atol=1e-10)
    assert np.isclose(est_narrow["kappa_f"], est_wide["kappa_f"], atol=1e-10)
    assert 3.5 <= est_narrow["kappa_f"] <= 4.5, f"Expected kappa_f ~ 4.0, got {est_narrow['kappa_f']}"

    # Equal-frequency warm items must report identifiable=False (no arbitrary slope)
    f_equal = np.array([5.0, 5.0, 5.0, 5.0, 0.0], dtype=np.float64)
    Y_eq = rng.normal(0.0, 1.0, size=(5, 10))
    mu0_eq = rng.normal(0.0, 1.0, size=(5, 10))
    est_eq = EmpiricalBayesRouter.fit_variance_components(f_equal, Y_eq, mu0_eq)
    assert est_eq["identifiable"] is False
    assert est_eq["raw_sigma0_sq"] is None and est_eq["raw_sigma_eps_sq"] is None
    assert est_eq["kappa_f"] == 0.0


def test_batch_invariance_with_frozen_prior_scale():
    train_bundles = [[0, 1, 2], [1, 2, 3], [0, 3, 4], [2, 4, 5]]
    op = SimplicialGraphOperator(train_bundles, n_items=7, svd_dim=4)
    freq_t = torch.tensor(op.freq, dtype=torch.float32)[:, None]

    q0 = [[0, 1]]
    q_batch = [[0, 1], [2, 3], [0, 2, 4]]

    def score_queries(qs, frozen_scale=1.75, kf=1.5):
        p_sc, d2_sc = op.score_pairwise_and_diff2(qs)
        inc_sc, d2_inc = op.score_simplicial_incidence_only(qs, beta_clique=1.5)
        rao_sc = op.score_rao_concordance(qs, gamma=0.0)
        Z = {
            "pair": zcols(torch.tensor(p_sc)),
            "diff2_pair": zcols(torch.tensor(d2_sc)),
            "simp": zcols(torch.tensor(p_sc + inc_sc)),
            "diff2": zcols(torch.tensor(d2_sc + d2_inc)),
            "rao": zcols(torch.tensor(rao_sc)),
            "cat": zcols(torch.arange(7, dtype=torch.float32)[:, None].repeat(1, len(qs))),
        }
        Y = Z["simp"] + 0.5 * Z["diff2"] + 0.25 * Z["rao"] + 0.5 * Z["cat"]
        return EmpiricalBayesRouter.apply_eb_shrinkage(
            Y, Z, freq_t, kappa_f=kf, w={"cat": 0.5}, prior_keys=["cat"], prior_scale=frozen_scale
        )

    s_single = score_queries(q0)[:, 0].numpy()
    s_batch_q0 = score_queries(q_batch)[:, 0].numpy()
    max_diff = float(np.max(np.abs(s_single - s_batch_q0)))
    assert max_diff < 1e-6, f"Adding unrelated queries changed original query scores by {max_diff}"


def test_nonidentifiable_evaluator_searches_disable_shrinkage():
    # Execute each production search's candidate expression without importing the
    # evaluators, whose module-level code starts dataset loading and training.
    root = Path(__file__).resolve().parents[1]
    for filename, target in [("fullrank_eval.py", "kf"),
                             ("bundlerec_eval.py", "kf_cand"),
                             ("dunnhumby_eval.py", "kf_cand")]:
        tree = ast.parse((root / "simba" / filename).read_text())
        searches = [node for node in ast.walk(tree)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == target]
        assert len(searches) == 1, f"Expected one posterior search in {filename}"
        expression = compile(ast.Expression(searches[0].iter), filename, "eval")
        for freq in [np.zeros(4), np.array([0., 0., 0., 2.]), np.array([5., 5., 5., 0.])]:
            Y = np.arange(8, dtype=float).reshape(4, 2)
            prior = np.zeros_like(Y)
            vc = EmpiricalBayesRouter.fit_variance_components(freq, Y, prior)
            context = {"EmpiricalBayesRouter": EmpiricalBayesRouter,
                       "vc": vc, "kf_est": vc["kappa_f"]}
            candidates = tuple(eval(expression, {"__builtins__": {}}, context))
            assert candidates == (0.0,), f"{filename} re-enabled unidentified shrinkage: {candidates}"
            posterior = EmpiricalBayesRouter.conjugate_posterior(Y, prior, freq, candidates[0])
            np.testing.assert_array_equal(posterior, Y)
        # Preserve the existing tuning choices when the components are identifiable.
        vc = {"identifiable": True, "kappa_f": 1.75}
        candidates = tuple(eval(expression, {"__builtins__": {}},
                                {"EmpiricalBayesRouter": EmpiricalBayesRouter,
                                 "vc": vc, "kf_est": vc["kappa_f"]}))
        assert 0.0 in candidates and 1.75 in candidates and any(k > 0 for k in candidates)


def test_validation_tolerance_pruning():
    Zv = {
        "simp": torch.tensor([[2.0], [1.0], [0.0]]),
        "bridge": torch.tensor([[2.0], [1.0], [0.0]]),
    }
    w = {"simp": 1.0, "bridge": 0.5}
    pruned = EmpiricalBayesRouter.prune_weights_val(
        w, Zv, eval_ndcg_fn=lambda f: 0.50, optional_keys=("bridge",), val_eps=5e-4
    )
    assert "bridge" not in pruned and "simp" in pruned


if __name__ == "__main__":
    test_row_stochastic_diffusion()
    test_pure_incidence_surplus_and_joint_homogeneity()
    test_rao_concordance_redundancy_vs_complementarity()
    test_exact_conjugate_posterior_and_variance_identification()
    test_batch_invariance_with_frozen_prior_scale()
    test_validation_tolerance_pruning()
    test_nonidentifiable_evaluator_searches_disable_shrinkage()
    print("All 7 SIMBA theoretical, numerical, batch-invariance, & evaluator-policy tests passed!", flush=True)
