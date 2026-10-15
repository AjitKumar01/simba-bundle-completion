"""Controlled One-Factor-at-a-Time Ablation Driver (`simba/simba/ablation_study.py`).

Isolates each mechanism while holding base pairwise channel weights strictly fixed, and fits
both `prior_scale` and `kappa_f` (`fit_variance_components`) on the validation split (`vset`)
so no test-batch statistics are used at test time:
- (0) Pairwise-Base: Purely linear pairwise channels (including pairwise category compatibility).
- (1) + Simplicial-Incidence (Delta_inc): Adds ONLY the pure higher-order hyperedge surplus (|b cap S| >= 2).
- (2) + Rao-Concordance (R_{-1}): Adds ONLY dissimilarity-weighted 2nd-order seed concordance
      [A^2 / (A + tau)] * sum_{s != s'} (1 - G_{s,s'}) p_s p_{s'}.
- (3) + Full Higher-Order (Delta_inc + R_{-1}): Combines both mechanisms on fixed base weights.
- (4) + Conjugate EB Posterior: Applies the exact conjugate normal-normal posterior
      theta_hat = (1 - u_c) * Y + u_c * mu_0 using validation-frozen `prior_scale` and `kappa_f`.
"""
import os, sys, csv, json, math, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simba.operators import SimplicialGraphOperator, BilateralContentBridge, EmpiricalBayesRouter, zcols
from simba.paths import dataset_dir

def eval_top20(F_mat, bundles):
    n, nb = F_mat.shape
    rng = np.random.default_rng(0)
    F = F_mat + rng.random((n, nb)).astype(np.float32) * 1e-6
    for j, (s, _) in enumerate(bundles):
        F[s, j] = -1e9
    top = np.argpartition(-F, 20, axis=0)[:20]
    return float(np.mean([len(set(top[:, j]) & set(gt)) / len(gt) for j, (_, gt) in enumerate(bundles)]))


def run_retail_ablation(ds: str, seeds: int = 5, output=None):
    root = dataset_dir(ds, "BR_DIR")
    raw = defaultdict(list)
    with open(os.path.join(root, "bundle_item.csv")) as f:
        r = csv.reader(f); next(r)
        for b, it in r:
            raw[int(b)].append(int(it))
    items = sorted({i for v in raw.values() for i in v})
    imap = {it: k for k, it in enumerate(items)}; n = len(items)
    seen = set(); bkeys_all, bundles_all = [], []
    for b, v in raw.items():
        u = list(dict.fromkeys(imap[i] for i in v))
        if len(u) >= 2:
            k = tuple(sorted(u))
            if k not in seen:
                seen.add(k); bkeys_all.append(b); bundles_all.append(u)

    b2u, b2s = {}, {}
    with open(os.path.join(root, "user_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for u, b, ts in r:
            b2u[int(b)] = int(u)
    with open(os.path.join(root, "session_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for s, b in r:
            b2s[int(b)] = int(s)

    uip_r, uip_c, uip_map = [], [], {}
    with open(os.path.join(root, "user_item_pretrain.csv")) as f:
        r = csv.reader(f); next(r)
        for u, it, _ in r:
            it_i = int(it)
            if it_i in imap:
                uip_r.append(uip_map.setdefault(int(u), len(uip_map)))
                uip_c.append(imap[it_i])
    X_pre = sp.csr_matrix((np.ones(len(uip_r), np.float32), (uip_r, uip_c)), shape=(max(len(uip_map), 1), n))
    udeg = np.asarray(X_pre.sum(1)).ravel()
    C_pre = ((sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre).T @ (sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre)).tocsr()
    C_pre.setdiag(0)

    def parse_path(cat):
        cat = cat.strip()
        if cat.startswith("[[") and cat.endswith("]]"):
            return [x.strip() for x in cat[2:-2].split("],[")[0].split(",") if x.strip()]
        return []

    cat_paths = [[] for _ in range(n)]; catmap = {}; cat_lbl = np.full(n, -1, int)
    with open(os.path.join(root, "item_categories.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0]); p = parse_path(",".join(row[1:]))
            if it in imap and p:
                idx = imap[it]; cat_paths[idx] = p
                cat_lbl[idx] = catmap.setdefault(tuple(p), len(catmap))
    for i in range(n):
        if cat_lbl[i] < 0:
            cat_lbl[i] = len(catmap); catmap[f"_n{i}"] = cat_lbl[i]

    titles = ["" for _ in range(n)]
    with open(os.path.join(root, "item_titles.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0])
            if it in imap:
                titles[imap[it]] = ",".join(row[1:])
    rich_txt = [titles[i] + " " + " ".join(cat_paths[i]) for i in range(n)]
    tf_mat = normalize(TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(rich_txt), norm="l2", axis=1).tocsr()
    CONTENT = TruncatedSVD(128, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)

    steps = ["0_Pairwise", "1_+SimpInc", "2_+RaoConc", "3_+FullHO", "4_+ConjEB"]
    acc = {k: [] for k in steps}
    for sd in range(seeds):
        order = list(range(len(bundles_all))); random.Random(sd).shuffle(order)
        allb = [bundles_all[i] for i in order]; allk = [bkeys_all[i] for i in order]
        ntr = int(0.7 * len(allb)); nva = int(0.1 * len(allb))
        trb, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]; trk = allk[:ntr]

        def mask_split(bs, s0):
            r = random.Random(s0); out = []
            for b in bs:
                bb = b[:]; r.shuffle(bb); k = max(1, int(round(len(bb) * 0.5)))
                s, h = bb[:k], bb[k:]
                if not h:
                    s, h = bb[:-1], [bb[-1]]
                out.append((s, h))
            return out

        vset = mask_split(vab, sd + 1)
        tset = mask_split(teb, sd + 2)

        op = SimplicialGraphOperator(trb, n_items=n, svd_dim=64, alpha_deg=0.5, gamma_vol=0.5, beta_clique=1.5)
        ur, uc, umap_tr, sr, sc_, smap_tr = [], [], {}, [], [], {}
        for bi_key, b in zip(trk, trb):
            u = b2u.get(bi_key, -1)
            if u >= 0:
                ui = umap_tr.setdefault(u, len(umap_tr))
                for it in b:
                    ur.append(ui); uc.append(it)
            s_id = b2s.get(bi_key, -1)
            if s_id >= 0:
                si = smap_tr.setdefault(s_id, len(smap_tr))
                for it in b:
                    sr.append(si); sc_.append(it)
        X_u = sp.csr_matrix((np.ones(len(ur), np.float32), (ur, uc)), shape=(max(len(umap_tr), 1), n))
        C_u = (X_u.T @ X_u).tocsr(); C_u.setdiag(0)
        X_s = sp.csr_matrix((np.ones(len(sr), np.float32), (sr, sc_)), shape=(max(len(smap_tr), 1), n))
        C_s = (X_s.T @ X_s).tocsr(); C_s.setdiag(0)

        W_sym = (sp.diags(1.0 / np.sqrt(op.freq + 1.0)) @ (op.C + 0.5 * C_s + 0.5 * C_u + 0.2 * C_pre) @ sp.diags(1.0 / np.sqrt(op.freq + 1.0))).tocsr()
        TF_bridge = normalize(tf_mat + 1.5 * (W_sym @ tf_mat) + 0.5 * (W_sym @ (W_sym @ tf_mat)), norm="l2", axis=1).tocsr()
        BR_DENSE = (CONTENT + 1.5 * (W_sym @ CONTENT) + 0.5 * (W_sym @ (W_sym @ CONTENT))).astype(np.float32)
        BR_DENSE /= (np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-8)

        def score_steps_0_to_3(bset):
            seeds_only = [s for s, _ in bset]
            Ind = op.build_indicator(seeds_only)
            s_pair_b, _ = op.score_pairwise_and_diff2(seeds_only)
            delta_inc, _ = op.score_simplicial_incidence_only(seeds_only, beta_clique=1.5)
            s_u = np.asarray((C_u @ Ind).todense())
            s_s = np.asarray((C_s @ Ind).todense())
            s_pre = np.asarray((C_pre @ Ind).todense())
            s_br = np.asarray((TF_bridge @ (TF_bridge.T @ Ind)).todense())
            s_tf = np.asarray((tf_mat @ (tf_mat.T @ Ind)).todense())

            pair_collab = s_pair_b + 0.35 * s_u + 0.35 * s_s
            diff2_pair = np.asarray(W_sym @ pair_collab)
            inc_collab = pair_collab + delta_inc
            diff2_inc = np.asarray(W_sym @ inc_collab)
            r_rao = op.score_rao_concordance(seeds_only, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.25)

            F0 = zcols(pair_collab) + 0.8 * zcols(diff2_pair) + 0.8 * zcols(s_br) + 0.5 * zcols(s_tf) + 0.5 * zcols(s_pre)
            F1 = zcols(inc_collab)  + 0.8 * zcols(diff2_inc)  + 0.8 * zcols(s_br) + 0.5 * zcols(s_tf) + 0.5 * zcols(s_pre)
            F2 = F0 + 0.35 * zcols(r_rao)
            F3 = F1 + 0.35 * zcols(r_rao)
            raw_prior = 0.8 * zcols(s_br) + 0.5 * zcols(s_tf) + 0.5 * zcols(s_pre) + 0.35 * zcols(r_rao)
            return F0, F1, F2, F3, raw_prior

        # Fit `frozen_prior_scale` and `vc` strictly on the validation split (`vset`)
        _, _, _, F3_val, raw_prior_val = score_steps_0_to_3(vset)
        frozen_prior_scale = float(np.std(F3_val) / (np.std(raw_prior_val) + 1e-9))
        mu0_val = raw_prior_val * frozen_prior_scale
        vc_val = EmpiricalBayesRouter.fit_variance_components(op.freq, F3_val, mu0_val)

        # Evaluate on test split (`tset`) using validation-frozen `frozen_prior_scale` and `vc_val["kappa_f"]`
        F0, F1, F2, F3, raw_prior_test = score_steps_0_to_3(tset)
        mu0_test = raw_prior_test * frozen_prior_scale
        F4 = EmpiricalBayesRouter.conjugate_posterior(F3, mu0_test, op.freq, vc_val["kappa_f"])

        for k_name, Fm in zip(steps, [F0, F1, F2, F3, F4]):
            acc[k_name].append(eval_top20(Fm, tset))

    print(f"\n=== {ds.upper()} ({seeds}-Seed Controlled One-Factor-at-a-Time Ablation; Validation-Frozen Scale & Variance) ===", flush=True)
    base_m = float(np.mean(acc["0_Pairwise"]))
    for k_name in steps:
        m = float(np.mean(acc[k_name]))
        print(f"  {k_name:14s}: R@20 = {100*m:.2f}% ({100*(m - base_m):+.2f} pp vs Pairwise)", flush=True)
    result = {"dataset": ds, "split_seeds": list(range(seeds)),
              "calibration": "prior scale and variance fitted on validation only",
              "steps": {name: {"per_seed_R@20": values,
                                  "mean_R@20": float(np.mean(values)),
                                  "sample_std_R@20": float(np.std(values, ddof=1)) if seeds > 1 else None}
                        for name, values in acc.items()}}
    if output:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{ds}_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["electronic", "food", "clothing"], default=["electronic", "food", "clothing"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/generated"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for ds_name in args.datasets:
        run_retail_ablation(ds_name, args.seeds, args.output)
