"""Core Mathematical Operators for SIMBA (Simplicial-Spectral Empirical-Bayes Bundle Architecture).

Implements three distinct, identifiable estimators on top of a unified pairwise baseline:
1. Pure Simplicial Hyperedge Incidence Surplus (`score_simplicial_incidence_only`):
   Computes ONLY the excess (k >= 3)-way joint co-occurrence surplus Delta_inc(c | S) from historical
   bundles sharing >= 2 seeds with S (|b cap S| >= 2). Identically zero on single-seed queries (|S| = 1)
   and on queries with max_b |b cap S| <= 1. Non-zero overlap is a necessary condition for Delta_inc
   to alter rankings; whether that surplus improves retrieval on active queries is empirical.
2. Redundancy-Corrected Quadratic Seed Concordance (`score_rao_concordance` / `score_concordance_e2`):
   Computes the normalized second-order elementary symmetric seed-agreement score weighted by Rao's (1982)
   quadratic diversity kernel K_S(s, s'; gamma):
       R_gamma(c | S; tau) = [A_c^2 / (A_c + tau)] * sum_{s != s'} K_S(s, s'; gamma) * p_{c,s} * p_{c,s'}
   Jointly degree-1 homogeneous in (w, tau): R_gamma(lambda * w; lambda * tau) = lambda * R_gamma(w; tau).
3. Exposure-Modulated Conjugate Normal-Normal Empirical-Bayes Posterior (`EmpiricalBayesRouter`):
   For standardized latent compatibility theta_{S,c} | mu_0 ~ N(mu_0(c|S), sigma_0^2) and exposure-noisy
   observation Y(c|S) | theta_{S,c}, f_c ~ N(theta_{S,c}, sigma_eps^2 / f_c):
   - Estimates (sigma_0^2, sigma_eps^2, kappa_f = sigma_eps^2 / sigma_0^2) via constrained OLS regression
     of the mean squared residual E[(Y - mu_0)^2 | f_c] = sigma_0^2 + sigma_eps^2 * (1 / f_c) on 1 / f_c
     when Var(1 / f_c) > 0 (returning both raw unclipped OLS estimates and truncated plug-in estimates,
     and explicitly reporting `identifiable=False` when warm frequencies have zero variance).
   - Computes the exact conjugate posterior mean using validation-frozen prior scaling:
         theta_hat_{S,c} = (1 - u_c) * Y(c|S) + u_c * mu_0(c|S),   where u_c = kappa_f / (f_c + kappa_f).
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans


def zcols(X):
    """Column-wise z-normalization over the item catalogue (supports both torch.Tensor and np.ndarray)."""
    if isinstance(X, torch.Tensor):
        return (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-9)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)


class SimplicialGraphOperator:
    """Operates on the bundle-item hypergraph incidence matrix B in {0,1}^{|B| x |I|}.

    Provides:
    - `score_pairwise_and_diff2`: First-order additive pairwise score s_pair(c | S) and its 2-hop diffusion.
    - `score_simplicial_incidence_only`: Pure higher-order incidence surplus Delta_inc(c | S) (zero for |S| <= 1).
    - `score_concordance_e2` & `score_rao_concordance`: Second-order pairwise seed concordance R_gamma(c | S; tau).
    """
    def __init__(self, train_bundles, n_items: int, svd_dim: int = 128,
                 alpha_deg: float = 0.5, gamma_vol: float = 0.5,
                 beta_clique: float = 1.5, lambda_cov: float = 0.0,
                 bundle_weights: np.ndarray = None):
        self.n = n_items
        self.alpha_deg = alpha_deg
        self.gamma_vol = gamma_vol
        self.beta_clique = beta_clique
        self.lambda_cov = lambda_cov

        br, bc, bw = [], [], []
        b_iter = list(train_bundles.values()) if isinstance(train_bundles, dict) else list(train_bundles)
        n_b = len(b_iter)
        for bi, items in enumerate(b_iter):
            w_b = float(bundle_weights[bi]) if bundle_weights is not None else 1.0
            for it in dict.fromkeys(items):
                br.append(bi)
                bc.append(it)
                bw.append(w_b)
        self.n_bundles = max(n_b, 1)
        self.br = br
        self.bc = bc
        self.B = sp.csr_matrix((np.asarray(bw, np.float32), (br, bc)), shape=(self.n_bundles, n_items))
        self.B_bin = sp.csr_matrix((np.ones(len(br), np.float32), (br, bc)), shape=(self.n_bundles, n_items))
        self.C = (self.B.T @ self.B).tocsr()
        self.C.setdiag(0)
        self.C.eliminate_zeros()
        self.C_bin = self.C.copy()
        self.C_bin.data[:] = 1.0

        self.freq = np.asarray(self.B.sum(0)).ravel().astype(np.float32)
        self.bsize = np.asarray(self.B_bin.sum(1)).ravel().astype(np.float32)
        self.inv_sqrt_f = (1.0 / np.power(self.freq + 5.0, self.alpha_deg)).astype(np.float32)
        self.inv_bsize = (1.0 / (self.bsize + 1.0)).astype(np.float32)
        self.inv_vol_pow = (1.0 / np.power(self.bsize + 1.0, self.gamma_vol)).astype(np.float32)
        self.diag_corr = (self.inv_sqrt_f ** 2) * np.asarray(self.B.T @ self.inv_bsize).ravel()

        # Exact zero-diagonal row-stochastic bipartite transition matrix P_rw
        inv_f_raw = np.where(self.freq > 0, 1.0 / np.maximum(self.freq, 1e-9), 0.0).astype(np.float32)
        inv_b_raw = np.where(self.bsize > 0, 1.0 / np.maximum(self.bsize, 1e-9), 0.0).astype(np.float32)
        W_0 = (sp.diags(inv_f_raw) @ self.B.T @ sp.diags(inv_b_raw) @ self.B).tocsr()
        W_0.setdiag(0.0)
        W_0.eliminate_zeros()
        row_sums = np.asarray(W_0.sum(1)).ravel().astype(np.float32)
        inv_row = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-12), 0.0).astype(np.float32)
        self.P_rw = (sp.diags(inv_row) @ W_0).tocsr()

        # Pairwise IDF-normalized transition operator W_pair
        self.W_pair = (self.B.T @ sp.diags(self.inv_vol_pow) @ self.B @ sp.diags(self.inv_sqrt_f)).tocsr()
        self.W_pair.setdiag(0.0)
        self.W_pair.eliminate_zeros()
        self.W_pair_sq = self.W_pair.copy()
        self.W_pair_sq.data = np.square(self.W_pair_sq.data)

        # PPMI + Truncated-SVD collaborative spectral factors
        freq_c = np.asarray(self.C.sum(1)).ravel() + 1e-9
        totc = float(self.C.sum())
        Cc = self.C.tocoo()
        pv = np.maximum(0.0, np.log((Cc.data * totc) / (freq_c[Cc.row] * freq_c[Cc.col]) + 1e-12)).astype(np.float32)
        self.PPMI = sp.csr_matrix((pv, (Cc.row, Cc.col)), shape=self.C.shape)
        k_svd = min(svd_dim, max(1, n_items - 2))
        if self.PPMI.nnz > 0 and k_svd >= 1:
            U = TruncatedSVD(n_components=k_svd, random_state=0).fit_transform(self.PPMI).astype(np.float32)
            self.Ucsvd = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
        else:
            self.Ucsvd = np.zeros((n_items, max(1, k_svd)), dtype=np.float32)

    def propagate_rw(self, X_mat: np.ndarray) -> np.ndarray:
        """Exact zero-diagonal row-stochastic Markov transition P_rw @ X_mat."""
        if X_mat.ndim == 1:
            return np.asarray(self.P_rw @ X_mat).ravel().astype(np.float32)
        return np.asarray(self.P_rw @ X_mat).astype(np.float32)

    def propagate(self, X_mat: np.ndarray) -> np.ndarray:
        """Factored 1-step zero-diagonal symmetrically normalized bipartite transition."""
        if X_mat.ndim == 1:
            X_mat = X_mat[:, None]
            squeeze = True
        else:
            squeeze = False
        Z1 = self.B @ (self.inv_sqrt_f[:, None] * X_mat)
        Z2 = self.B.T @ (self.inv_bsize[:, None] * Z1)
        out = self.inv_sqrt_f[:, None] * Z2 - self.diag_corr[:, None] * X_mat
        return out.ravel().astype(np.float32) if squeeze else out.astype(np.float32)

    def build_indicator(self, bundles) -> sp.csr_matrix:
        nb = len(bundles)
        Ind = sp.lil_matrix((self.n, nb), dtype=np.float32)
        for j, s in enumerate(bundles):
            Ind[s, j] = 1.0
        return Ind.tocsr()

    def score_pairwise_and_diff2(self, bundles):
        """Matched purely pairwise linear baseline (beta_clique=0, beta_e2=0)."""
        Ind = self.build_indicator(bundles)
        Ind_w = Ind.multiply(self.inv_sqrt_f[:, None]).tocsr()
        ov_w = np.asarray((self.B @ Ind_w).todense()) * self.inv_vol_pow[:, None]
        sc_pair = np.asarray(self.B.T @ ov_w).astype(np.float32)
        sc_diff2 = self.propagate(sc_pair)
        return sc_pair, sc_diff2

    def score_simplicial_incidence_only(self, bundles, beta_clique: float = None):
        """Mechanism I (Pure Higher-Order Simplicial Incidence Surplus Delta_inc):
        Computes ONLY the higher-order hyperedge surplus:
            Delta_inc(c | S) = sum_{b ni c} (|b|+1)^{-gamma_vol} (sum_{s in b cap S} (f_s+5)^{-alpha_deg}) * beta_clique * max(0, |b cap S| - 1).
        Strictly 0.0 whenever |S| <= 1 or when no training bundle contains >= 2 seeds from S.
        Returns (delta_inc, diff2_delta_inc).
        """
        bc = self.beta_clique if beta_clique is None else beta_clique
        Ind = self.build_indicator(bundles)
        Ind_w = Ind.multiply(self.inv_sqrt_f[:, None]).tocsr()
        ov_raw = np.asarray((self.B_bin @ Ind).todense())
        ov_w = np.asarray((self.B @ Ind_w).todense()) * self.inv_vol_pow[:, None]
        surplus_act = ov_w * (bc * np.maximum(0.0, ov_raw - 1.0))
        delta_inc = np.asarray(self.B.T @ surplus_act).astype(np.float32)
        diff2_delta = self.propagate(delta_inc)
        return delta_inc, diff2_delta

    def score_concordance_e2(self, bundles, tau: float = 0.05) -> np.ndarray:
        """Mechanism IIa (Unweighted Gini-Simpson Pairwise Seed Concordance):
        For per-seed support w_s >= 0, A = sum_s w_s, and p_s = w_s / A:
            R_0(c | S; tau) = 2 * e_2(c | S) / (A + tau) = [A^2 / (A + tau)] * (1 - sum_{s in S} p_s^2).
        Jointly degree-1 homogeneous in (w, tau): R_0(lambda * w; lambda * tau) = lambda * R_0(w; tau).
        Captures 2nd-order pairwise seed agreement (not full m-way product conjunction).
        """
        Ind = self.build_indicator(bundles)
        e1 = np.asarray((self.W_pair @ Ind).todense()).astype(np.float32)
        s2 = np.asarray((self.W_pair_sq @ Ind).todense()).astype(np.float32)
        two_e2 = np.maximum(0.0, np.square(e1) - s2)
        return (two_e2 / (e1 + tau)).astype(np.float32)

    def score_rao_concordance(self, bundles, dense_feat: np.ndarray = None,
                              cat_lbl: np.ndarray = None, gamma: float = -1.0,
                              alpha_dense: float = 0.20, tau: float = 0.05) -> np.ndarray:
        """Mechanism IIb (Redundancy-Corrected Quadratic Concordance via Rao's Quadratic Entropy, Rao 1982):
            R_gamma(c | S; tau) = [A_c^2 / (A_c + tau)] * sum_{s != s' in S} K_S(s, s'; gamma) * p_{c,s} * p_{c,s'}
        where G_{s, s'} in [0, 1] is the seed-seed redundancy kernel (category + feature similarity).
        """
        nb = len(bundles)
        out = np.zeros((self.n, nb), dtype=np.float32)
        W_csc = self.W_pair.tocsc()
        for j, s in enumerate(bundles):
            k = len(s)
            if k < 2:
                continue
            W_sub = W_csc[:, s].toarray().astype(np.float32)
            if dense_feat is not None and alpha_dense > 0.0:
                W_sub = W_sub + alpha_dense * np.maximum(0.0, dense_feat @ dense_feat[s].T)
            A_c = W_sub.sum(axis=1)
            G_S = np.eye(k, dtype=np.float32)
            if dense_feat is not None and cat_lbl is not None:
                sim_f = np.maximum(0.0, dense_feat[s] @ dense_feat[s].T)
                same_c = (cat_lbl[s][:, None] == cat_lbl[s][None, :]).astype(np.float32)
                G_S = 0.5 * same_c + 0.5 * sim_f
            elif dense_feat is not None:
                G_S = np.maximum(0.0, dense_feat[s] @ dense_feat[s].T)
            elif cat_lbl is not None:
                G_S = (cat_lbl[s][:, None] == cat_lbl[s][None, :]).astype(np.float32)
            np.fill_diagonal(G_S, 1.0)

            if gamma < 0.0:
                K_S = np.maximum(0.0, 1.0 + gamma * G_S)
            else:
                K_S = 1.0 + gamma * G_S
            np.fill_diagonal(K_S, 0.0)
            quad = np.sum((W_sub @ K_S) * W_sub, axis=1)
            out[:, j] = quad / (A_c + tau)
        return out

    def score_simplicial_and_diff2(self, bundles, beta_clique: float = None, beta_e2: float = 0.35):
        """Returns combined (s_simp, s_diff2) = (s_pair + delta_inc + beta_e2 * r_e2, diff2)."""
        sc_pair, _ = self.score_pairwise_and_diff2(bundles)
        delta_inc, _ = self.score_simplicial_incidence_only(bundles, beta_clique=beta_clique)
        sc_ho = sc_pair + delta_inc
        if beta_e2 > 0.0:
            sc_ho = sc_ho + beta_e2 * self.score_concordance_e2(bundles, tau=0.05)
        if self.lambda_cov > 0.0:
            Ind = self.build_indicator(bundles)
            n_seeds = np.asarray((self.C_bin @ Ind).todense())
            sc_ho = (sc_ho * (1.0 + self.lambda_cov * n_seeds)).astype(np.float32)
        sc_diff2 = self.propagate(sc_ho)
        return sc_ho, sc_diff2


class DualSpectrumOntology:
    """Decomposes taxonomic co-occurrence into:
    - Off-diagonal complementarity PMI (P_comp, diag=0)
    - Signed on-diagonal self-exclusion (`score_slot_saturation`, an additive pairwise diagonal correction)
    - On-diagonal homophily (P_homo)
    - Soft fuzzy sub-type complementarity (M_soft @ P_s).
    """
    def __init__(self, train_bundles, cat_lbl: np.ndarray, n_items: int,
                 alb_lbl: np.ndarray = None, cluster_feat: np.ndarray = None,
                 n_soft_clusters: int = 400, soft_tau: float = 0.05, device=torch.device("cpu")):
        self.n = n_items
        self.cat_lbl = cat_lbl
        self.alb_lbl = alb_lbl
        self.device = device
        self.NCc = int(cat_lbl.max()) + 1

        b_iter = list(train_bundles.values()) if isinstance(train_bundles, dict) else list(train_bundles)
        pr, pc = [], []
        for bi, items in enumerate(b_iter):
            for it in dict.fromkeys(items):
                pr.append(bi)
                pc.append(cat_lbl[it])
        Tc = sp.csr_matrix((np.ones(len(pr), np.float32), (pr, pc)), shape=(max(len(b_iter), 1), self.NCc))
        Tc_arr = Tc.toarray()
        has_cat = (Tc_arr >= 1).sum(0).astype(np.float32)
        dup_cat = (Tc_arr >= 2).sum(0).astype(np.float32)
        self.dup_rate = ((dup_cat + 0.5) / (has_cat + 1.0)).astype(np.float32)

        Ctc = (Tc.T @ Tc).tocsr()
        Ctc.setdiag(0)
        Ctc.eliminate_zeros()
        fcc = np.asarray(Tc.sum(0)).ravel() + 1e-9
        totcc = float(fcc.sum())
        coc = Ctc.tocoo()
        pvc = np.maximum(0.0, np.log((coc.data * totcc) / (fcc[coc.row] * fcc[coc.col]) + 1e-12)).astype(np.float32)
        self.Pcat = sp.csr_matrix((pvc, (coc.row, coc.col)), shape=(self.NCc, self.NCc))

        I_art = sp.csr_matrix((np.ones(n_items, np.float32), (np.arange(n_items), cat_lbl)), shape=(n_items, self.NCc))
        art_deg = np.asarray(I_art.sum(0)).ravel() + 5.0
        self.I_art_w = I_art @ sp.diags(1.0 / np.sqrt(art_deg))
        if alb_lbl is not None:
            NCalb = int(alb_lbl.max()) + 1
            I_alb = sp.csr_matrix((np.ones(n_items, np.float32), (np.arange(n_items), alb_lbl)), shape=(n_items, NCalb))
            alb_deg = np.asarray(I_alb.sum(0)).ravel() + 2.0
            self.I_alb_w = I_alb @ sp.diags(1.0 / np.sqrt(alb_deg))
        else:
            self.I_alb_w = None

        self.MP_t = None
        self.Msoft_t = None
        if cluster_feat is not None:
            km = MiniBatchKMeans(n_clusters=n_soft_clusters, random_state=0, n_init=3, batch_size=4096).fit(cluster_feat)
            Cn = km.cluster_centers_.astype(np.float32)
            Cn /= (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-8)
            Ml = cluster_feat @ Cn.T / soft_tau
            Ml -= Ml.max(1, keepdims=True)
            Msoft = np.exp(Ml)
            Msoft /= (Msoft.sum(1, keepdims=True) + 1e-9)
            Msoft = Msoft.astype(np.float32)
            SB = np.zeros((len(b_iter), n_soft_clusters), np.float32)
            bc_flat = [it for items in b_iter for it in items]
            for bi, items in enumerate(b_iter):
                SB[bi] = Msoft[items].sum(0)
            cooc_s = SB.T @ SB - (Msoft[bc_flat].T @ Msoft[bc_flat])
            np.fill_diagonal(cooc_s, 0.0)
            cooc_s = np.maximum(cooc_s, 0.0)
            cf_s = Msoft.sum(0)
            tot_s = cf_s.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                Ps = np.maximum(0.0, np.log((cooc_s * tot_s) / (np.outer(cf_s, cf_s) + 1e-9) + 1e-12)).astype(np.float32)
            self.MP_t = torch.tensor((Msoft @ Ps).astype(np.float32), device=device)
            self.Msoft_t = torch.tensor(Msoft, device=device)

    def score_slot_saturation(self, bundles, thresh: float = 0.15) -> np.ndarray:
        """Signed diagonal category self-exclusion penalty (additive across seeds)."""
        nb = len(bundles)
        slot_pen = np.zeros((self.n, nb), dtype=np.float32)
        for j, s in enumerate(bundles):
            for it in s:
                c_id = self.cat_lbl[it]
                if self.dup_rate[c_id] < thresh:
                    slot_pen[self.cat_lbl == c_id, j] -= (1.0 - self.dup_rate[c_id])
        return slot_pen


class BilateralContentBridge:
    """Disentangles category/type centroids from content features and propagates them across the
    bipartite graph operator.
    """
    @staticmethod
    def disentangle(x: np.ndarray, lbl: np.ndarray, n_clusters: int) -> np.ndarray:
        y = x.copy()
        for c in range(n_clusters):
            idx = np.where(lbl == c)[0]
            if len(idx):
                y[idx] = x[idx] - x[idx].mean(0)
        return (y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    @staticmethod
    def build_dense_bridge(style_feat: np.ndarray, graph_op: SimplicialGraphOperator, alpha_bridge: float = 0.5) -> np.ndarray:
        p_feat = graph_op.propagate(style_feat)
        p_feat /= (np.linalg.norm(p_feat, axis=1, keepdims=True) + 1e-8)
        br = style_feat + alpha_bridge * p_feat
        return (br / (np.linalg.norm(br, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    @staticmethod
    def build_cold_warm_knn(content_feat: np.ndarray, freq: np.ndarray,
                            cat_lbl: np.ndarray = None, k_nn: int = 10) -> sp.csr_matrix:
        n = content_feat.shape[0]
        warm_mask = freq > 0
        if not np.any(warm_mask):
            return sp.csr_matrix((n, n), dtype=np.float32)
        kr, kc, kv = [], [], []
        if cat_lbl is not None:
            n_cats = int(cat_lbl.max()) + 1
            groups = [np.where(cat_lbl == c)[0] for c in range(n_cats)]
        else:
            groups = [np.arange(n)]

        all_warm = np.where(warm_mask)[0]
        for idx_all in groups:
            if len(idx_all) == 0:
                continue
            idx_warm = idx_all[warm_mask[idx_all]]
            if len(idx_warm) == 0:
                idx_warm = all_warm
            sim = content_feat[idx_all] @ content_feat[idx_warm].T
            for ii, a_idx in enumerate(idx_all):
                if warm_mask[a_idx]:
                    w_pos = np.searchsorted(idx_warm, a_idx)
                    if w_pos < len(idx_warm) and idx_warm[w_pos] == a_idx:
                        sim[ii, w_pos] = -1e9
            k_use = min(k_nn, len(idx_warm))
            top_k = np.argpartition(-sim, k_use - 1, axis=1)[:, :k_use]
            top_v = np.maximum(0.0, np.take_along_axis(sim, top_k, axis=1)) ** 2
            rsum = top_v.sum(1, keepdims=True) + 1e-9
            top_v /= rsum
            for ii, a_idx in enumerate(idx_all):
                for jj in range(k_use):
                    if top_v[ii, jj] > 0:
                        kr.append(int(a_idx))
                        kc.append(int(idx_warm[top_k[ii, jj]]))
                        kv.append(float(top_v[ii, jj]))
        return sp.csr_matrix((np.asarray(kv, np.float32), (kr, kc)), shape=(n, n))


class EmpiricalBayesRouter:
    """Exposure-Modulated Conjugate Normal-Normal Empirical-Bayes Router.

    Generative & Observation Model for Standardized Latent Compatibility theta_{S,c}:
        theta_{S,c} | mu_0 ~ N(mu_0(c|S), sigma_0^2),
        Y(c|S) | theta_{S,c}, f_c ~ N(theta_{S,c}, sigma_eps^2 / f_c).

    1. Variance-Component Estimation (`fit_variance_components`):
       Under the conditional second-moment equation E[(Y(c|S) - mu_0(c|S))^2 | f_c] = sigma_0^2 + sigma_eps^2 * (1 / f_c)
       for f_c > 0:
       - If Var(1 / f_c) == 0 across warm items (e.g., equal-frequency data) or fewer than 2 warm items exist,
         sigma_0^2 and sigma_eps^2 are NOT separately identifiable; `fit_variance_components` returns
         `identifiable=False`, `raw_sigma0_sq=None`, `raw_sigma_eps_sq=None`, and `kappa_f=0.0`.
       - When Var(1 / f_c) > 0, `raw_sigma_eps_sq` (the OLS slope) and `raw_sigma0_sq` (the OLS intercept)
         are the exact unclipped, unrounded OLS moment estimators (which are unbiased for sigma_eps^2 and
         sigma_0^2 under the linear conditional-moment model), while `sigma0_sq`, `sigma_eps_sq`, and `kappa_f`
         apply positivity truncation and ratio clipping for numerical admissibility in finite samples (which
         trades unbiasedness for non-negativity).

    2. Exact Conjugate Posterior Update (`conjugate_posterior` / `apply_eb_shrinkage`):
           hat{theta}_{S,c} = (1 - u_c) * Y(c|S) + u_c * mu_0(c|S),   where u_c = kappa_f / (f_c + kappa_f).
    """
    @staticmethod
    def fit_variance_components(freq, Y, mu_0) -> dict:
        """Constrained OLS variance-component estimator for
        E[(Y - mu_0)^2 | f_c] = sigma_0^2 + sigma_eps^2 * (1 / f_c) on warm items (f_c > 0).

        Returns unrounded float64 raw OLS estimates (`raw_sigma0_sq`, `raw_sigma_eps_sq`),
        `identifiable` flag (`True` iff Var(1/f_c) > 0 across >= 2 warm items), and admissible
        truncated plug-in values (`sigma0_sq`, `sigma_eps_sq`, `kappa_f`).
        """
        f = freq.detach().cpu().numpy().ravel() if isinstance(freq, torch.Tensor) else np.asarray(freq, dtype=np.float64).ravel()
        Y_np = Y.detach().cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y, dtype=np.float64)
        M_np = mu_0.detach().cpu().numpy() if isinstance(mu_0, torch.Tensor) else np.asarray(mu_0, dtype=np.float64)
        if Y_np.ndim == 1:
            Y_np = Y_np[:, None]
            M_np = M_np[:, None]
        warm = f > 0
        if int(warm.sum()) < 2:
            return {
                "identifiable": False,
                "raw_sigma0_sq": None,
                "raw_sigma_eps_sq": None,
                "sigma0_sq": None,
                "sigma_eps_sq": None,
                "kappa_f": 0.0,
            }
        x = (1.0 / f[warm]).astype(np.float64)
        v = np.mean(np.square(Y_np[warm] - M_np[warm]), axis=1).astype(np.float64)
        x_bar = float(np.mean(x))
        v_bar = float(np.mean(v))
        x_cent = x - x_bar
        denom = float(np.dot(x_cent, x_cent))
        if denom <= 1e-12:
            # Equal-frequency warm items: slope and intercept are not separately identifiable
            return {
                "identifiable": False,
                "raw_sigma0_sq": None,
                "raw_sigma_eps_sq": None,
                "sigma0_sq": None,
                "sigma_eps_sq": None,
                "kappa_f": 0.0,
            }
        raw_slope = float(np.dot(x_cent, v - v_bar) / denom)
        raw_intercept = float(v_bar - raw_slope * x_bar)
        sigma0_sq = float(max(raw_intercept, 1e-4))
        sigma_eps_sq = float(max(raw_slope, 1e-4))
        kappa_f = float(np.clip(sigma_eps_sq / sigma0_sq, 0.05, 50.0))
        return {
            "identifiable": True,
            "raw_sigma0_sq": raw_intercept,
            "raw_sigma_eps_sq": raw_slope,
            "sigma0_sq": sigma0_sq,
            "sigma_eps_sq": sigma_eps_sq,
            "kappa_f": kappa_f,
        }

    @staticmethod
    def shrinkage_candidates(variance_components: dict, candidates):
        """Keep shrinkage disabled when its variance components are not identifiable.

        Evaluator validation searches must respect the estimator's zero fallback,
        even when their usual candidate grid contains positive variance ratios.
        """
        if not variance_components["identifiable"]:
            return (0.0,)
        return tuple(candidates)

    @staticmethod
    def fit_eb_moments(freq: np.ndarray, R_cooc_val=None, Y_val=None, mu_0_val=None) -> dict:
        if Y_val is not None and mu_0_val is not None:
            vc = EmpiricalBayesRouter.fit_variance_components(freq, Y_val, mu_0_val)
            kappa_f = vc["kappa_f"]
            sigma0_sq = vc["sigma0_sq"]
            sigma_eps_sq = vc["sigma_eps_sq"]
            identifiable = vc["identifiable"]
        else:
            f = np.asarray(freq, dtype=np.float64).ravel()
            warm_f = f[f > 0]
            if len(warm_f) > 1 and float(np.var(warm_f)) > 1e-12:
                mu_f = float(np.mean(warm_f))
                var_f = float(np.var(warm_f))
                kappa_f = float(np.clip((mu_f ** 2) / max(var_f - mu_f, 1e-3), 0.5, 25.0))
                identifiable = True
            else:
                kappa_f = 0.0
                identifiable = False
            sigma0_sq = 1.0 if identifiable else None
            sigma_eps_sq = kappa_f if identifiable else None

        kappa_p = 2.0
        if R_cooc_val is not None:
            rc = R_cooc_val.detach().cpu().numpy() if isinstance(R_cooc_val, torch.Tensor) else np.asarray(R_cooc_val)
            pos_rc = rc[rc > 0]
            if len(pos_rc) > 1:
                mu_c = float(np.mean(pos_rc))
                var_c = float(np.var(pos_rc))
                kappa_p = float(np.clip((mu_c ** 2) / max(var_c - mu_c, 1e-3), 0.5, 10.0))

        return {
            "identifiable": identifiable,
            "sigma0_sq": sigma0_sq,
            "sigma_eps_sq": sigma_eps_sq,
            "kappa_f": kappa_f,
            "kappa_p": kappa_p,
        }

    @staticmethod
    def conjugate_posterior(Y, mu_0, freq, kappa_f: float):
        """Computes the exact conjugate normal-normal posterior expectation:
            theta_hat = (1 - u_c) * Y + u_c * mu_0,   where u_c = kappa_f / (f_c + kappa_f).
        Supports both torch.Tensor and np.ndarray.
        """
        if kappa_f <= 0.0:
            return Y.clone() if isinstance(Y, torch.Tensor) else np.asarray(Y, dtype=np.float32).copy()
        if isinstance(Y, torch.Tensor):
            f_t = freq if isinstance(freq, torch.Tensor) else torch.tensor(freq, dtype=Y.dtype, device=Y.device)
            if Y.ndim == 2 and f_t.ndim == 1:
                f_t = f_t[:, None]
            u_c = kappa_f / (f_t + kappa_f)
            return (1.0 - u_c) * Y + u_c * mu_0
        f_np = np.asarray(freq, dtype=np.float32)
        if Y.ndim == 2 and f_np.ndim == 1:
            f_np = f_np[:, None]
        u_c = kappa_f / (f_np + kappa_f)
        return ((1.0 - u_c) * Y + u_c * mu_0).astype(np.float32)

    @staticmethod
    def apply_eb_shrinkage(Y, mu_0_or_Z, freq_t, kappa_f: float = 1.0,
                           w: dict = None, prior_keys: list = None, prior_scale: float = 1.0):
        """Computes the exact conjugate normal-normal posterior:
            theta_hat = (1 - u_c) * Y + u_c * mu_0,   where u_c = kappa_f / (f_c + kappa_f).
        Uses a frozen scalar `prior_scale` fitted on validation so test inference is strictly
        independent of test batching.
        """
        if isinstance(mu_0_or_Z, dict):
            raw_prior = sum(w.get(k, 0.5) * mu_0_or_Z[k] for k in prior_keys)
            mu_0 = raw_prior * float(prior_scale)
        else:
            mu_0 = mu_0_or_Z * float(prior_scale) if prior_scale != 1.0 else mu_0_or_Z
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu_0, freq_t, kappa_f)

    @staticmethod
    def apply_tail_cold_calibration(F_mat, Z_prior, freq, R_cooc=None,
                                    z_thresh: float = 4.5, max_cold_per_query: int = 1,
                                    target_rank: int = 19, k_top: int = 20):
        """Optional post-hoc top-K slot-reservation retrieval policy (distinct from the conjugate posterior)."""
        is_torch = isinstance(F_mat, torch.Tensor)
        F_np = F_mat.detach().cpu().numpy().copy() if is_torch else np.asarray(F_mat, dtype=np.float32).copy()
        Zp_np = Z_prior.detach().cpu().numpy() if isinstance(Z_prior, torch.Tensor) else np.asarray(Z_prior, dtype=np.float32)
        f_np = freq.detach().cpu().numpy().ravel() if isinstance(freq, torch.Tensor) else np.asarray(freq).ravel()

        cold_mask = (f_np == 0)
        if not np.any(cold_mask) or F_np.shape[0] <= k_top:
            return F_mat

        top_k_vals = np.partition(F_np, -k_top, axis=0)[-k_top:]
        top_k_sorted = np.sort(top_k_vals, axis=0)
        slot_idx = max(0, min(k_top - 1, k_top - target_rank))
        base_target_score = top_k_sorted[slot_idx][None, :] + 0.05

        if R_cooc is not None:
            rc_np = R_cooc.detach().cpu().numpy() if isinstance(R_cooc, torch.Tensor) else np.asarray(R_cooc)
            n_cooc_q = (rc_np > 0).sum(0, keepdims=True)
            target_score = np.where(n_cooc_q == 0, top_k_sorted[min(k_top - 2, slot_idx + 10)][None, :] + 0.05, base_target_score)
        else:
            target_score = base_target_score

        Zp_cold = np.where(cold_mask[:, None], Zp_np, -1e9)
        if max_cold_per_query == 1:
            top_c_thresh = np.max(Zp_cold, axis=0, keepdims=True) - 1e-6
        else:
            k_c = min(max_cold_per_query, int(cold_mask.sum()))
            top_c_thresh = np.partition(Zp_cold, -k_c, axis=0)[-k_c][None, :] - 1e-6

        promote_mask = (Zp_cold >= top_c_thresh) & (Zp_np > z_thresh)
        F_out = np.where(promote_mask, np.maximum(F_np, target_score), F_np)
        if is_torch:
            return torch.tensor(F_out, dtype=F_mat.dtype, device=F_mat.device)
        return F_out

    @staticmethod
    def prune_weights_val(w: dict, Zv: dict, eval_ndcg_fn, optional_keys: tuple = ("bridge", "contrast", "softcat", "homo"),
                          val_eps: float = 5e-4) -> dict:
        """Validation-tolerance backward elimination."""
        w_curr = {k: v for k, v in w.items() if v > 0}
        if len(w_curr) <= 1:
            return w_curr
        base_score = eval_ndcg_fn(sum(w_curr[k] * Zv[k] for k in w_curr))
        for k in optional_keys:
            if k in w_curr and len(w_curr) > 1:
                w_try = {k2: v2 for k2, v2 in w_curr.items() if k2 != k}
                try_score = eval_ndcg_fn(sum(w_try[k2] * Zv[k2] for k2 in w_try))
                if try_score >= base_score - val_eps:
                    w_curr = w_try
                    base_score = try_score
        return w_curr

    @staticmethod
    def prune_weights_mdl(w: dict, Zv: dict, eval_ndcg_fn, optional_keys: tuple = ("bridge", "contrast", "softcat", "homo"),
                          mdl_eps: float = 5e-4) -> dict:
        """Alias for `prune_weights_val`."""
        return EmpiricalBayesRouter.prune_weights_val(w, Zv, eval_ndcg_fn, optional_keys=optional_keys, val_eps=mdl_eps)

    @staticmethod
    def precision_backoff_weights(Zv: dict, eval_ndcg_fn, sigs: list, gamma: float = 1.5, tau_rel: float = 0.25) -> dict:
        lifts = {}
        for k in sigs:
            nd = eval_ndcg_fn(Zv[k])
            lifts[k] = max(0.0, float(nd))
        max_l = max(lifts.values()) if lifts else 0.0
        if max_l <= 0.0:
            return {k: 1.0 for k in sigs}
        w_eb = {}
        for k, val in lifts.items():
            if val >= tau_rel * max_l:
                w_eb[k] = round((val / max_l) ** gamma, 4)
        return w_eb
