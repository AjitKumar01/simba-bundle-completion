"""SIMBA Evaluator for BundleRec E-Commerce Datasets (electronic / food / clothing) under Duplicate-Free RaMen 7:1:2 Protocol.

Integrates the unified 3-estimator framework with validation-frozen prior scaling:
1. Pure Simplicial Hyperedge Incidence Surplus (`score_simplicial_incidence_only`, Delta_inc)
2. Redundancy-Corrected Quadratic Concordance (`score_rao_concordance`, R_{-1})
3. Exact Exposure-Modulated Conjugate Normal-Normal Posterior (`conjugate_posterior`, (1 - u_c)*Y + u_c*mu_0)
   with validation-frozen `prior_scale` and constrained OLS variance estimation (`fit_variance_components`).
"""
import os, sys, csv, math, random, time
from collections import defaultdict
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simba._device import DEVICE
from simba.paths import dataset_dir
from simba.operators import SimplicialGraphOperator, EmpiricalBayesRouter, zcols

DS = os.environ.get("BRDS", "electronic")
ROOT = dataset_dir(DS, "BR_DIR")
K = 20; MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.5")); SEEDS = int(os.environ.get("SEEDS", "5"))
USE_AUX = os.environ.get("NO_AUX", "0") != "1"

raw = defaultdict(list)
with open(os.path.join(ROOT, "bundle_item.csv")) as f:
    r = csv.reader(f); next(r)
    for b, it in r:
        raw[int(b)].append(int(it))
items = sorted({i for v in raw.values() for i in v})
imap = {it: k for k, it in enumerate(items)}; n = len(items)

seen_bundles = set()
bkeys_all, bundles_all = [], []
for b, v in raw.items():
    u = list(dict.fromkeys(imap[i] for i in v))
    if len(u) >= 2:
        key = tuple(sorted(u))
        if key not in seen_bundles:
            seen_bundles.add(key)
            bkeys_all.append(b)
            bundles_all.append(u)

b2u = {}
with open(os.path.join(ROOT, "user_bundle.csv")) as f:
    r = csv.reader(f); next(r)
    for u, b, ts in r:
        b2u[int(b)] = int(u)
b2s = {}
with open(os.path.join(ROOT, "session_bundle.csv")) as f:
    r = csv.reader(f); next(r)
    for s, b in r:
        b2s[int(b)] = int(s)

uip_r, uip_c = [], []; uip_map = {}
if USE_AUX:
    with open(os.path.join(ROOT, "user_item_pretrain.csv")) as f:
        r = csv.reader(f); next(r)
        for u, it, _ in r:
            it_i = int(it)
            if it_i in imap:
                uip_r.append(uip_map.setdefault(int(u), len(uip_map)))
                uip_c.append(imap[it_i])
X_pre = sp.csr_matrix((np.ones(len(uip_r), np.float32), (uip_r, uip_c)), shape=(max(len(uip_map), 1), n))
udeg = np.asarray(X_pre.sum(1)).ravel()
X_pre_w = sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre
C_pre = (X_pre_w.T @ X_pre_w).tocsr(); C_pre.setdiag(0)


def parse_path(cat):
    cat = cat.strip()
    if cat.startswith("[[") and cat.endswith("]]"):
        return [x.strip() for x in cat[2:-2].split("],[")[0].split(",") if x.strip()]
    return []


cat_paths = [[] for _ in range(n)]
catmap_leaf = {}; cat_lbl = np.full(n, -1, int)
catmap_l2 = {}; cat_l2 = np.full(n, -1, int)
with open(os.path.join(ROOT, "item_categories.csv")) as f:
    r = csv.reader(f); next(r)
    for row in r:
        it = int(row[0]); p = parse_path(",".join(row[1:]))
        if it in imap and p:
            idx = imap[it]; cat_paths[idx] = p
            cat_lbl[idx] = catmap_leaf.setdefault(tuple(p), len(catmap_leaf))
            cat_l2[idx] = catmap_l2.setdefault(tuple(p[:min(2, len(p))]), len(catmap_l2))
for i in range(n):
    if cat_lbl[i] < 0:
        cat_lbl[i] = len(catmap_leaf); catmap_leaf[f"_n{i}"] = cat_lbl[i]
    if cat_l2[i] < 0:
        cat_l2[i] = len(catmap_l2); catmap_l2[f"_n{i}"] = cat_l2[i]
NCc = int(cat_lbl.max()) + 1
NCl2 = int(cat_l2.max()) + 1

titles = ["" for _ in range(n)]
with open(os.path.join(ROOT, "item_titles.csv")) as f:
    r = csv.reader(f); next(r)
    for row in r:
        it = int(row[0])
        if it in imap:
            titles[imap[it]] = ",".join(row[1:])
rich_txt = [titles[i] + " " + " ".join(cat_paths[i]) for i in range(n)]
tf_mat = TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(rich_txt)
tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
cdim = min(128, tf_mat.shape[1] - 1)
CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)
CONTENT_T = torch.tensor(CONTENT, device=DEVICE)


class Contrast(nn.Module):
    def __init__(self, d, h=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h))
        self.k = nn.Linear(h, h); self.q = nn.Parameter(torch.randn(h)); self.h = h

    def emb(self, f):
        return self.enc(f)

    def set_rep(self, sf, sm):
        e = self.emb(sf)
        a = (self.k(e) @ self.q / math.sqrt(self.h)).masked_fill(~sm.bool(), -1e9).softmax(-1)
        return (a[..., None] * e).sum(1)


def train_contrast(train):
    MK = 8; torch.manual_seed(0)
    m = Contrast(CONTENT.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    bids = [b for b in train if len(b) >= 2]; rng = random.Random(0); npr = np.random.default_rng(0)

    def pad(sl):
        sf = torch.zeros(len(sl), MK, CONTENT.shape[1], device=DEVICE)
        sm = torch.zeros(len(sl), MK, device=DEVICE)
        for i, s in enumerate(sl):
            s = s[:MK]; sf[i, :len(s)] = CONTENT_T[s]; sm[i, :len(s)] = 1
        return sf, sm

    for _ in range(400):
        sl, pi = [], []
        for _ in range(128):
            b = rng.choice(bids); it = b[:]; rng.shuffle(it); k = max(1, len(it) // 2)
            sl.append(it[:k]); pi.append(it[k])
        sr = m.set_rep(*pad(sl))
        cand = torch.cat([torch.tensor(pi, device=DEVICE), torch.tensor(npr.integers(n, size=256), device=DEVICE)])
        loss = Fn.cross_entropy(sr @ m.emb(CONTENT_T[cand]).t() / 0.1, torch.arange(len(sl), device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        E = m.emb(CONTENT_T)
    return m, E, pad


def build(tr_bundles, tr_bkeys):
    graph_op = SimplicialGraphOperator(tr_bundles, n_items=n, svd_dim=128, alpha_deg=0.5, gamma_vol=0.5, beta_clique=1.5)
    freq = graph_op.freq

    ur, uc = [], []; umap_tr = {}
    sr, sc_ = [], []; smap_tr = {}
    for bi_key, b in zip(tr_bkeys, tr_bundles):
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

    P = (sp.diags(1.0 / np.sqrt(freq + 1.0)) @ (graph_op.C + 0.5 * C_s + 0.5 * C_u + (0.2 * C_pre if USE_AUX else 0.0)) @ sp.diags(1.0 / np.sqrt(freq + 1.0))).tocsr()
    TF_bridge = normalize(tf_mat + 1.5 * (P @ tf_mat) + 0.5 * (P @ (P @ tf_mat)), norm="l2", axis=1).tocsr()
    BR_DENSE = (CONTENT + 1.5 * (P @ CONTENT) + 0.5 * (P @ (P @ CONTENT))).astype(np.float32)
    BR_DENSE /= (np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-8)

    def build_pcat(lbl_arr, n_types, keep_diag=False):
        tr_, tc_ = [], []
        for bi, b in enumerate(tr_bundles):
            for it in b:
                tr_.append(bi); tc_.append(lbl_arr[it])
        T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(tr_bundles), n_types))
        Ct = (T.T @ T).tocsr()
        if not keep_diag:
            Ct.setdiag(0); Ct.eliminate_zeros()
        fc = np.asarray(T.sum(0)).ravel() + 1e-9; totc = float(fc.sum()); coc = Ct.tocoo()
        pc_ = np.maximum(0., np.log((coc.data * totc) / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
        return sp.csr_matrix((pc_, (coc.row, coc.col)), shape=(n_types, n_types))

    Pcat_leaf = build_pcat(cat_lbl, NCc, keep_diag=False)
    Pcat_homo = build_pcat(cat_lbl, NCc, keep_diag=True)
    Pcat_l2 = build_pcat(cat_l2, NCl2, keep_diag=True)
    model, E, pad = train_contrast(tr_bundles)
    return graph_op, C_u, C_s, P, TF_bridge, BR_DENSE, Pcat_leaf, Pcat_homo, Pcat_l2, model, E, pad


SIG = ["simp", "diff2", "cooc", "ppmi", "csvd", "cat", "homo", "tfidf", "bridge", "pretrain", "contrast"]
if not USE_AUX:
    SIG = [s for s in SIG if s != "pretrain"]


def score_split(bset, built):
    graph_op, C_u, C_s, P, TF_bridge, BR_DENSE, Pcat_leaf, Pcat_homo, Pcat_l2, model, E, pad = built
    seeds_list = [s for s, _ in bset]
    nb = len(seeds_list)
    Ind = graph_op.build_indicator(seeds_list)

    s_pair_b, _ = graph_op.score_pairwise_and_diff2(seeds_list)
    delta_inc, _ = graph_op.score_simplicial_incidence_only(seeds_list, beta_clique=1.5)
    s_u = np.asarray((C_u @ Ind).todense())
    s_s = np.asarray((C_s @ Ind).todense())

    s_pair = (s_pair_b + 0.35 * s_u + 0.35 * s_s).astype(np.float32)
    s_diff2_pair = np.asarray(P @ s_pair).astype(np.float32)
    s_simp = (s_pair + delta_inc).astype(np.float32)
    s_diff2 = np.asarray(P @ s_simp).astype(np.float32)

    s_rao = graph_op.score_rao_concordance(
        seeds_list, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.25, tau=0.05
    )

    s_cooc = np.asarray((graph_op.C @ Ind).todense()).astype(np.float32)
    s_ppmi = np.asarray((graph_op.PPMI @ Ind).todense()).astype(np.float32)
    Us = np.stack([graph_op.Ucsvd[s].mean(0) for s in seeds_list], 1)
    s_csvd = (graph_op.Ucsvd @ Us).astype(np.float32)

    s_cat = np.zeros((n, nb), np.float32)
    s_homo = np.zeros((n, nb), np.float32)
    for j, s in enumerate(seeds_list):
        sa = [cat_lbl[i] for i in s]
        sa2 = [cat_l2[i] for i in s]
        s_cat[:, j] = np.asarray(Pcat_leaf[sa].sum(0)).ravel()[cat_lbl]
        s_homo[:, j] = np.asarray(Pcat_homo[sa].sum(0)).ravel()[cat_lbl] + 0.5 * np.asarray(Pcat_l2[sa2].sum(0)).ravel()[cat_l2]

    s_tfidf = np.asarray((tf_mat @ (tf_mat.T @ Ind)).todense()).astype(np.float32)
    s_bridge = np.asarray((TF_bridge @ (TF_bridge.T @ Ind)).todense()).astype(np.float32)
    s_pre = np.asarray((C_pre @ Ind).todense()).astype(np.float32)
    with torch.no_grad():
        sr = model.set_rep(*pad([s[:8] for s in seeds_list]))
        s_con = (E @ sr.t()).cpu().numpy().astype(np.float32)

    R = {
        "simp": s_simp, "diff2": s_diff2, "pair": s_pair, "diff2_pair": s_diff2_pair,
        "rao": s_rao, "cooc": s_cooc, "ppmi": s_ppmi, "csvd": s_csvd,
        "cat": s_cat, "homo": s_homo, "tfidf": s_tfidf, "bridge": s_bridge,
        "pretrain": s_pre, "contrast": s_con,
    }
    Z = {k: zcols(R[k]) for k in SIG}
    Z["pair"] = zcols(R["pair"])
    Z["diff2_pair"] = zcols(R["diff2_pair"])
    Z["rao"] = zcols(R["rao"])
    return R, Z


def eval_mat(F_mat, bundles, Ks=(10, 20)):
    nb = len(bundles); rng = np.random.default_rng(0)
    F = F_mat + rng.random((n, nb)).astype(np.float32) * 1e-6
    for j, (s, _) in enumerate(bundles):
        F[s, j] = -1e9
    top = np.argpartition(-F, max(Ks), axis=0)[:max(Ks)]
    acc = {K2: ([], []) for K2 in Ks}
    for j, (_, gt) in enumerate(bundles):
        col = top[:, j]; order = col[np.argsort(-F[col, j])]; g = set(gt)
        for K2 in Ks:
            hits = [1 if p in g else 0 for p in order[:K2]]
            dcg = sum(x / math.log2(i + 2) for i, x in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), K2)))
            acc[K2][0].append(sum(hits) / len(g)); acc[K2][1].append(dcg / idcg if idcg else 0.0)
    return {K2: (float(np.mean(acc[K2][0])), float(np.mean(acc[K2][1]))) for K2 in Ks}


def mask(bundles, seed):
    r = random.Random(seed); out = []
    for b in bundles:
        bb = b[:]; r.shuffle(bb); k = max(1, int(round(len(bb) * (1 - MASK_FRAC))))
        s, h = bb[:k], bb[k:]
        if not h:
            s, h = bb[:-1], [bb[-1]]
        out.append((s, h))
    return out


def train_item2vec_br(train, dim=64, epochs=8, neg=5, bs=4096, maxpairs=3_000_000):
    pr, pc = [], []; order = list(range(len(train))); random.Random(0).shuffle(order)
    for idx in order:
        b = train[idx]
        for a in range(len(b)):
            for c in range(len(b)):
                if a != c:
                    pr.append(b[a]); pc.append(b[c])
        if len(pr) >= maxpairs:
            break
    pr = np.asarray(pr, np.int64); pc = np.asarray(pc, np.int64)
    freq_i = np.zeros(n)
    for b in train:
        for it in b:
            freq_i[it] += 1
    pn = (freq_i + 1e-9) ** 0.75; pn /= pn.sum()
    U = nn.Embedding(n, dim).to(DEVICE); V = nn.Embedding(n, dim).to(DEVICE)
    nn.init.normal_(U.weight, std=0.1); nn.init.normal_(V.weight, std=0.1)
    opt = torch.optim.Adam(list(U.parameters()) + list(V.parameters()), lr=0.01); npr = np.random.default_rng(0)
    for _ in range(epochs):
        perm = npr.permutation(len(pr))
        for s in range(0, len(pr), bs):
            idx = perm[s:s + bs]; c = torch.tensor(pr[idx], device=DEVICE); o = torch.tensor(pc[idx], device=DEVICE)
            ng = torch.tensor(npr.choice(n, size=(len(idx), neg), p=pn), device=DEVICE)
            uc = U(c); pos = Fn.logsigmoid((uc * V(o)).sum(-1)); negl = Fn.logsigmoid(-(uc.unsqueeze(1) * V(ng)).sum(-1)).sum(-1)
            loss = -(pos + negl).mean(); opt.zero_grad(); loss.backward(); opt.step()
    W = U.weight.detach().cpu().numpy().astype(np.float32)
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)


t0 = time.time()
METH = ["popularity", "co-occurrence", "content-kNN", "item2vec", "Pairwise-Repl", "SIMBA-NoRes", "Ours"]
agg = {m: {10: ([], []), 20: ([], [])} for m in METH}
for sd in range(SEEDS):
    order = list(range(len(bundles_all))); random.Random(sd).shuffle(order)
    allb = [bundles_all[i] for i in order]; allk = [bkeys_all[i] for i in order]
    ntr = int(0.7 * len(allb)); nva = int(0.1 * len(allb))
    trb, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]
    trk = allk[:ntr]
    built = build(trb, trk); graph_op = built[0]; freq = graph_op.freq; W = train_item2vec_br(trb)
    vset, tset = mask(vab, sd + 1), mask(teb, sd + 2)
    Rv, Zv = score_split(vset, built)
    Rt, Zt = score_split(tset, built)

    Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
    w = {k: 0.0 for k in SIG}; best = -1.0
    for _ in range(4):
        imp = False
        for k in SIG:
            for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                w2 = dict(w); w2[k] = val
                if sum(w2.values()) == 0:
                    continue
                F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                nd = eval_mat(F_v, vset, Ks=(20,))[20][1]
                if nd > best + 2e-4:
                    best = nd; w = w2; imp = True
        if not imp:
            break
    bw = EmpiricalBayesRouter.prune_weights_val(
        w, Zv_pair, lambda fmat: eval_mat(fmat, vset, Ks=(20,))[20][1],
        optional_keys=("contrast", "homo"), val_eps=3e-4
    )

    def build_Y(Z_dict, ho_params):
        alpha_inc, g_rao = ho_params
        Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
        base_p = sum(bw[k] * Zp[k] for k in bw)
        base_i = sum(bw[k] * Z_dict[k] for k in bw)
        return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

    best_ho = (0.5, 0.25)
    best_ho_nd = -1.0
    for alpha_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.2, 0.35]:
            nd = eval_mat(build_Y(Zv, (alpha_inc, g_rao)), vset, Ks=(20,))[20][1]
            if nd > best_ho_nd + 1e-5:
                best_ho_nd = nd; best_ho = (alpha_inc, g_rao)
    best = max(best, best_ho_nd)

    prior_ks = [k for k in ("cat", "homo", "tfidf", "bridge", "pretrain", "contrast") if k in SIG and bw.get(k, 0) > 0] or ["bridge", "tfidf", "cat"]

    # Freeze prior_scale on the validation set (`Zv`) so test inference is strictly independent of test batching
    Y_v = build_Y(Zv, best_ho)
    raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + best_ho[1] * Zv["rao"]
    frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
    mu0_v = raw_pr_v * frozen_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(freq, Y_v, mu0_v)

    def apply_eb(Z_dict, ho_params, kf, p_scale):
        Y = build_Y(Z_dict, ho_params)
        if kf <= 0.0:
            return Y
        raw_prior = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
        mu_0 = raw_prior * float(p_scale)
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu_0, freq, kf)

    best_kf = vc["kappa_f"]
    for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.1, 0.5, vc["kappa_f"], 2.0]):
        nd = eval_mat(apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale), vset, Ks=(20,))[20][1]
        if nd > best + 1e-4:
            best = nd; best_kf = kf_cand

    pop_mat = zcols(np.tile(freq[:, None], (1, len(tset))))
    cnt_mat = zcols(np.stack([CONTENT @ CONTENT[s].mean(0) for s, _ in tset], 1))
    i2v_mat = zcols(np.stack([W @ W[s].mean(0) for s, _ in tset], 1))
    res = {
        "popularity": eval_mat(pop_mat, tset),
        "co-occurrence": eval_mat(Zt["cooc"], tset),
        "content-kNN": eval_mat(cnt_mat, tset),
        "item2vec": eval_mat(i2v_mat, tset),
        "Pairwise-Repl": eval_mat(apply_eb(Zt, (0.0, 0.0), best_kf, frozen_prior_scale), tset),
        "SIMBA-NoRes": eval_mat(build_Y(Zt, best_ho), tset),
        "Ours": eval_mat(apply_eb(Zt, best_ho, best_kf, frozen_prior_scale), tset),
    }
    for m in METH:
        for K2 in (10, 20):
            agg[m][K2][0].append(res[m][K2][0]); agg[m][K2][1].append(res[m][K2][1])

print(f"\n=== {DS} MATRIX (full-ranking R/N@10,20; {SEEDS} duplicate-free seeds; aux={USE_AUX}; {time.time()-t0:.0f}s) ===", flush=True)
print(f"{'method':16s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}", flush=True)
for m in METH:
    a = agg[m]
    print(f"{m:16s} {np.mean(a[10][0]):8.4f} {np.mean(a[10][1]):8.4f} {np.mean(a[20][0]):8.4f} {np.mean(a[20][1]):8.4f}", flush=True)
