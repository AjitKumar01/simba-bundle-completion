"""SIMBA Evaluator for POG, POG_dense, Spotify, and Spotify_sparse.

Integrates the unified 3-estimator framework:
1. Pure Simplicial Hyperedge Incidence Surplus (`score_simplicial_incidence_only`, Delta_inc)
2. Redundancy-Corrected Quadratic Concordance (`score_rao_concordance`, R_gamma)
3. Exact Exposure-Modulated Conjugate Normal-Normal Posterior (`conjugate_posterior`, (1 - u_c)*Y + u_c*mu_0)
   with validation-frozen `prior_scale` and constrained OLS variance estimation (`fit_variance_components`),
   guaranteeing that test-query predictions are strictly independent of test batching.
"""
import json, math, os, random, sys, time
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simba._device import DEVICE
from simba.paths import dataset_dir
from simba.operators import (
    SimplicialGraphOperator,
    DualSpectrumOntology,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)

DS = os.environ.get("BUNDLE_DS", "pog")
POG = dataset_dir(DS, "DATA_DIR")
MUSIC = not os.path.exists(os.path.join(POG, "title_emb.npy"))
MAXK = 40 if MUSIC else 4
K = 20
VAL_N = 400
TEST_N = 3000


def rb(f):
    o = {}
    for l in open(os.path.join(POG, f)):
        p = l.strip().split(",")
        if len(p) >= 2:
            o[int(p[0])] = [int(x) for x in p[1:]]
    return o


def _loadpt(f):
    x = torch.load(os.path.join(POG, f), map_location="cpu", weights_only=False).detach().float().numpy()
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


n = json.load(open(os.path.join(POG, "count.json")))["#I"]
RAMEN = os.environ.get("RAMEN_SEED", "0")
rseed = int(RAMEN)
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.5"))
if os.path.exists(os.path.join(POG, "bi_full.txt")):
    fullb = list(rb("bi_full.txt").values())
else:
    vi, vgt = rb("bi_valid_input.txt"), rb("bi_valid_gt.txt")
    ti, tgt = rb("bi_test_input.txt"), rb("bi_test_gt.txt")
    fullb = list(rb("bi_train.txt").values())
    fullb += [vi[k] + vgt[k] for k in vi if k in vgt] + [ti[k] + tgt[k] for k in ti if k in tgt]

_ddbck = os.environ.get("DDBC_K")
if _ddbck:
    _k = int(_ddbck); _rt = random.Random(rseed + 99); _trunc = []
    for v in fullb:
        if len(v) >= _k:
            _st = _rt.randint(0, len(v) - _k); _trunc.append(v[_st:_st + _k])
    fullb = _trunc

seen_bundles = set()
allb = []
for v in fullb:
    u = list(dict.fromkeys(v))
    if len(u) >= 2:
        key = tuple(sorted(u))
        if key not in seen_bundles:
            seen_bundles.add(key)
            allb.append(u)

random.Random(rseed).shuffle(allb)
ntr = int(0.7 * len(allb)); nva = int(0.1 * len(allb))
trb, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]
train = {i: b for i, b in enumerate(trb)}


def _mask(bundles, s0):
    r = random.Random(s0); inp, gt = {}, {}
    for i, b in enumerate(bundles):
        bb = b[:]; r.shuffle(bb)
        k = max(1, int(round(len(bb) * (1 - MASK_FRAC))))
        seeds, held = bb[:k], bb[k:]
        if not held:
            seeds, held = bb[:-1], [bb[-1]]
        inp[i], gt[i] = seeds, held
    return inp, gt


RAMEN_SPLITS = {"valid": _mask(vab, rseed + 1), "test": _mask(teb, rseed + 2)}


def load_split(split, cap):
    inp, gt = RAMEN_SPLITS[split]
    bs = [b for b in inp if b in gt and inp[b] and gt[b]]
    random.Random(0).shuffle(bs); bs = bs[:cap]
    return [inp[b] for b in bs], [gt[b] for b in bs]


print(f"{DS}: {n} items, {len(train)} duplicate-free train bundles, MUSIC={MUSIC}", flush=True)

graph_op = SimplicialGraphOperator(train, n_items=n, svd_dim=128)
freq = graph_op.freq

feats = {}
if MUSIC:
    if os.path.exists(os.path.join(POG, "content_feature.pt")):
        feats["audio"] = _loadpt("content_feature.pt")
    clusterfeat = feats["audio"]
else:
    title = np.load(os.path.join(POG, "title_emb.npy")).astype(np.float32)
    title /= (np.linalg.norm(title, axis=1, keepdims=True) + 1e-8)
    feats["content"] = title
    if os.path.exists(os.path.join(POG, "content_feature.pt")):
        feats["image"] = _loadpt("content_feature.pt")
    if os.path.exists(os.path.join(POG, "description_feature.pt")):
        feats["desc"] = _loadpt("description_feature.pt")
    c_parts = [title] + ([feats["image"]] if "image" in feats else []) + ([feats["desc"]] if "desc" in feats else [])
    mm_feat = np.concatenate(c_parts, axis=1).astype(np.float32)
    mm_feat /= (np.linalg.norm(mm_feat, axis=1, keepdims=True) + 1e-8)
    clusterfeat = mm_feat

info = json.load(open(os.path.join(POG, "item_info.json")))
HAS_EXPLICIT_CAT = (not MUSIC) and (info["0"].get("cate") is not None or info["0"].get("cate_id") is not None)
alb_lbl = None
if HAS_EXPLICIT_CAT:
    cmap = {}; lbl = np.zeros(n, int)
    for k, v in info.items():
        cval = v.get("cate") if v.get("cate") is not None else v.get("cate_id")
        cmap.setdefault(cval, len(cmap)); lbl[int(k)] = cmap[cval]
    NC = int(lbl.max()) + 1
    cat_lbl = lbl.copy()
elif MUSIC and info["0"].get("artist_uri") is not None:
    km_tmp = MiniBatchKMeans(n_clusters=400, random_state=0, n_init=3, batch_size=4096).fit(clusterfeat)
    lbl = km_tmp.labels_; NC = int(lbl.max()) + 1
    am = {}; cat_lbl = np.zeros(n, int)
    alm = {}; alb_lbl = np.zeros(n, int)
    for k, v in info.items():
        cat_lbl[int(k)] = am.setdefault(v.get("artist_uri", "?"), len(am))
        alb_lbl[int(k)] = alm.setdefault(v.get("album_uri", "?"), len(alm))
else:
    km_tmp = MiniBatchKMeans(n_clusters=400, random_state=0, n_init=3, batch_size=4096).fit(clusterfeat)
    lbl = km_tmp.labels_; NC = int(lbl.max()) + 1
    cat_lbl = lbl.copy()

onto_op = DualSpectrumOntology(
    train, cat_lbl=cat_lbl, n_items=n, alb_lbl=alb_lbl,
    cluster_feat=(clusterfeat if not MUSIC else None),
    n_soft_clusters=400, soft_tau=0.05, device=DEVICE,
)

if MUSIC:
    feats["audiostyle"] = BilateralContentBridge.disentangle(feats["audio"], lbl, NC)
    feats["bridge"] = BilateralContentBridge.build_dense_bridge(feats["audiostyle"], graph_op, alpha_bridge=0.5)
    rao_dense_feat = feats["audiostyle"]
else:
    feats["style"] = BilateralContentBridge.disentangle(title, lbl, NC)
    if "image" in feats:
        feats["imgstyle"] = BilateralContentBridge.disentangle(feats["image"], lbl, NC)
    if "desc" in feats:
        feats["descstyle"] = BilateralContentBridge.disentangle(feats["desc"], lbl, NC)
    st_parts = [feats["style"]] + ([feats["imgstyle"]] if "imgstyle" in feats else []) + ([feats["descstyle"]] if "descstyle" in feats else [])
    st_all = np.concatenate(st_parts, axis=1).astype(np.float32)
    st_all /= (np.linalg.norm(st_all, axis=1, keepdims=True) + 1e-8)
    feats["bridge"] = BilateralContentBridge.build_dense_bridge(st_all, graph_op, alpha_bridge=0.5)
    rao_dense_feat = feats["style"]

base = feats["content"] if "content" in feats else feats["audio"]
parts = [base] + ([feats["image"]] if "image" in feats else []) + ([feats["desc"]] if "desc" in feats else [])
Fnp = np.concatenate(parts, 1).astype(np.float32)
FEAT = torch.tensor(Fnp, device=DEVICE)


class Contrast(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
        self.k = nn.Linear(d, d); self.pool_q = nn.Parameter(torch.randn(d)); self.d = d

    def emb(self, f):
        return self.enc(f)

    def set_rep(self, sf, sm):
        e = self.emb(sf)
        a = (self.k(e) @ self.pool_q / math.sqrt(self.d)).masked_fill(~sm.bool(), -1e9).softmax(-1)
        return (a[..., None] * e).sum(1)


torch.manual_seed(0)
model = Contrast(Fnp.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
bids = [b for b in train if len(train[b]) >= 2]
rng = random.Random(0); npr = np.random.default_rng(0)


def pad(sl):
    Bn = len(sl)
    sf = torch.zeros(Bn, MAXK, Fnp.shape[1], device=DEVICE)
    sm = torch.zeros(Bn, MAXK, device=DEVICE)
    for i, s in enumerate(sl):
        s = s[:MAXK]; sf[i, :len(s)] = FEAT[s]; sm[i, :len(s)] = 1
    return sf, sm


for step in range(900 if not MUSIC else 700):
    model.train(); sl, pi = [], []
    for _ in range(128):
        b = rng.choice(bids); it = train[b][:]; rng.shuffle(it); k = max(1, len(it) // 2)
        sl.append(it[:k]); pi.append(it[k])
    sr = model.set_rep(*pad(sl))
    cand = torch.cat([torch.tensor(pi, device=DEVICE), torch.tensor(npr.integers(n, size=256), device=DEVICE)])
    loss = Fn.cross_entropy(sr @ model.emb(FEAT[cand]).t() / 0.07, torch.arange(len(sl), device=DEVICE))
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    Emb = model.emb(FEAT)

feats["csvd"] = graph_op.Ucsvd
FEAT_T = {k: torch.tensor(v, device=DEVICE) for k, v in feats.items()}

SIG = (["simp", "diff2", "pop", "cooc", "ppmi", "csvd", "cat", "homo", "contrast", "audiostyle", "bridge"]) if MUSIC else \
      (["cat", "softcat", "style", "imgstyle", "bridge", "simp", "diff2", "pop", "cooc", "ppmi", "csvd", "contrast"])
SIG = [s for s in SIG if (s not in ("imgstyle",) or "imgstyle" in FEAT_T)]


def score_all_signals(bundles):
    """Computes raw score matrices R[k] and column-wise z-normalized Z[k] per query."""
    nb = len(bundles)
    Ind = graph_op.build_indicator(bundles)
    sc_pair, sc_diff2_pair = graph_op.score_pairwise_and_diff2(bundles)
    delta_inc, diff2_delta = graph_op.score_simplicial_incidence_only(bundles, beta_clique=1.5)
    sc_rao = graph_op.score_rao_concordance(
        bundles, dense_feat=rao_dense_feat, cat_lbl=cat_lbl,
        gamma=(1.0 if MUSIC else -1.0), alpha_dense=0.15, tau=0.05
    )
    sc_simp = sc_pair + delta_inc
    sc_diff2 = sc_diff2_pair + diff2_delta
    R = {
        "simp": torch.tensor(sc_simp, device=DEVICE),
        "diff2": torch.tensor(sc_diff2, device=DEVICE),
        "pair": torch.tensor(sc_pair, device=DEVICE),
        "diff2_pair": torch.tensor(sc_diff2_pair, device=DEVICE),
        "delta_inc": torch.tensor(delta_inc, device=DEVICE),
        "rao": torch.tensor(sc_rao, device=DEVICE),
        "pop": torch.tensor(np.tile(np.log1p(freq)[:, None], (1, nb)), dtype=torch.float32, device=DEVICE),
        "cooc": torch.tensor(np.asarray((graph_op.C @ Ind).todense()), dtype=torch.float32, device=DEVICE),
        "ppmi": torch.tensor(np.asarray((graph_op.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE),
    }
    out_cat = np.zeros((n, nb), np.float32)
    for j, s in enumerate(bundles):
        sa = [cat_lbl[i] for i in s]
        out_cat[:, j] = np.asarray(onto_op.Pcat[sa].sum(0)).ravel()[cat_lbl]
    if not MUSIC:
        out_cat = out_cat + 0.5 * onto_op.score_slot_saturation(bundles)
    R["cat"] = torch.tensor(out_cat, device=DEVICE)
    if MUSIC:
        art_sc = np.asarray((onto_op.I_art_w @ (onto_op.I_art_w.T @ Ind)).todense())
        alb_sc = np.asarray((onto_op.I_alb_w @ (onto_op.I_alb_w.T @ Ind)).todense())
        R["homo"] = torch.tensor((art_sc + 2.0 * alb_sc).astype(np.float32), device=DEVICE)
    else:
        Ms = torch.stack([onto_op.Msoft_t[s].sum(0) for s in bundles], 1)
        R["softcat"] = onto_op.MP_t @ Ms
    with torch.no_grad():
        sr = model.set_rep(*pad([s[:MAXK] for s in bundles]))
        R["contrast"] = Emb @ sr.t()
    for k in ("csvd", "style", "imgstyle", "audiostyle", "bridge"):
        if k in FEAT_T:
            f = FEAT_T[k]
            M = torch.stack([f[s].mean(0) for s in bundles], 1)
            R[k] = f @ M
    Z = {k: zcols(R[k]) for k in SIG}
    Z["pair"] = zcols(R["pair"])
    Z["diff2_pair"] = zcols(R["diff2_pair"])
    Z["rao"] = zcols(R["rao"])
    return R, Z


def eval_fused(fused, bundles, gts, Ks=(10, 20)):
    Kmax = max(Ks)
    f = fused + torch.rand_like(fused) * 1e-6
    for j, s in enumerate(bundles):
        f[s, j] = -1e9
    top = f.topk(Kmax, dim=0).indices.cpu().numpy()
    acc = {K2: ([], []) for K2 in Ks}
    for j, gt in enumerate(gts):
        gtset = set(gt); pred = top[:, j]
        for K2 in Ks:
            hits = [1 if p in gtset else 0 for p in pred[:K2]]
            dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gtset), K2)))
            acc[K2][0].append(sum(hits) / len(gtset))
            acc[K2][1].append(dcg / idcg if idcg else 0.0)
    return {K2: (float(np.mean(acc[K2][0])), float(np.mean(acc[K2][1]))) for K2 in Ks}


def fit_simba(vb, vg, Rv, Zv):
    """Fits pairwise base weights w, nested higher-order weights (alpha_inc, gamma_rao),
    and the conjugate posterior parameters (kf, prior_scale) strictly on the validation set.
    `prior_scale` is frozen from validation so test-time scoring is 100% invariant to test batching.
    """
    Z_pair_v = dict(Zv); Z_pair_v["simp"] = Zv["pair"]; Z_pair_v["diff2"] = Zv["diff2_pair"]
    GPS = {k: [0.0, 0.25, 0.5, 1.0, 1.5] for k in SIG}
    w = {k: 0.0 for k in SIG}; best_nd = -1.0
    for _ in range(4):
        improved = False
        for k in SIG:
            for v in GPS[k]:
                w2 = dict(w); w2[k] = v
                if sum(w2.values()) == 0:
                    continue
                fused = sum(w2[k2] * Z_pair_v[k2] for k2 in SIG if w2[k2])
                nd = eval_fused(fused, vb, vg, Ks=(20,))[20][1]
                if nd > best_nd + 2e-4:
                    best_nd = nd; w = w2; improved = True
        if not improved:
            break
    w = EmpiricalBayesRouter.prune_weights_val(
        w, Z_pair_v, lambda fmat: eval_fused(fmat, vb, vg, Ks=(20,))[20][1],
        optional_keys=("bridge", "contrast", "softcat", "homo"), val_eps=3e-4
    )

    freq_t = torch.tensor(freq, device=DEVICE)[:, None]
    prior_cand = ("cat", "homo", "audiostyle", "bridge", "contrast") if MUSIC else ("cat", "softcat", "style", "imgstyle", "bridge", "contrast")
    prior_keys = [k for k in prior_cand if k in SIG and w.get(k, 0) > 0]
    if not prior_keys:
        prior_keys = [k for k in prior_cand if k in SIG]

    base_pair_v = sum(w[k] * Z_pair_v[k] for k in w)
    base_inc_v = sum(w[k] * Zv[k] for k in w)
    best_ho = (0.0, 0.0)
    best_nd = eval_fused(base_pair_v, vb, vg, Ks=(20,))[20][1]
    for alpha_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.2, 0.35]:
            cand_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]
            nd = eval_fused(cand_v, vb, vg, Ks=(20,))[20][1]
            if nd > best_nd + 5e-5:
                best_nd = nd; best_ho = (alpha_inc, g_rao)

    alpha_inc, g_rao = best_ho
    Y_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]

    # Compute validation-frozen prior_scale so test predictions never depend on test batch composition
    raw_prior_v = sum(w.get(k, 0.5) * Zv[k] for k in prior_keys) + g_rao * Zv["rao"]
    if w.get("pop", 0) > 0:
        raw_prior_v = raw_prior_v + w["pop"] * Zv["pop"]
    val_prior_scale = float(((Y_v.std() + 1e-9) / (raw_prior_v.std() + 1e-9)).item())
    mu0_scaled_v = raw_prior_v * val_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(freq, Y_v, mu0_scaled_v)
    kf_est = vc["kappa_f"]

    best_eb = (0.0, val_prior_scale)
    for p_scale in (val_prior_scale, 1.0):
        mu0_cand = raw_prior_v * p_scale
        for kf in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.5, 1.0, kf_est, 5.0]):
            f_eb = EmpiricalBayesRouter.conjugate_posterior(Y_v, mu0_cand, freq_t, kf)
            nd = eval_fused(f_eb, vb, vg, Ks=(20,))[20][1]
            if nd > best_nd + 5e-5:
                best_nd = nd; best_eb = (kf, p_scale)
    return w, prior_keys, best_eb, freq_t, best_ho


def apply_simba_full(Z, w, prior_keys, best_eb, freq_t, best_ho):
    """Applies SIMBA scoring using the validation-frozen scalar `prior_scale` from `best_eb`,
    guaranteeing that each query column in `Z` is scored independently of other queries in the batch.
    """
    alpha_inc, g_rao = best_ho
    Z_pair = dict(Z); Z_pair["simp"] = Z["pair"]; Z_pair["diff2"] = Z["diff2_pair"]
    base_pair = sum(w[k] * Z_pair[k] for k in w)
    base_inc = sum(w[k] * Z[k] for k in w)
    Y = (1.0 - alpha_inc) * base_pair + alpha_inc * base_inc + g_rao * Z["rao"]
    kf, frozen_prior_scale = best_eb
    raw_prior = sum(w.get(k, 0.5) * Z[k] for k in prior_keys) + g_rao * Z["rao"]
    if w.get("pop", 0) > 0:
        raw_prior = raw_prior + w["pop"] * Z["pop"]
    mu_0 = raw_prior * float(frozen_prior_scale)
    return EmpiricalBayesRouter.conjugate_posterior(Y, mu_0, freq_t, kf)


def train_item2vec_W(dim=64, epochs=5, neg=5, bs=4096, maxpairs=3_000_000):
    bundles = [b for b in train.values() if len(b) >= 2]
    pr, pc = [], []; order = list(range(len(bundles))); random.Random(0).shuffle(order)
    for idx in order:
        b = bundles[idx]
        for a in range(len(b)):
            for c in range(len(b)):
                if a != c:
                    pr.append(b[a]); pc.append(b[c])
        if len(pr) >= maxpairs:
            break
    pr = np.asarray(pr, np.int64); pc = np.asarray(pc, np.int64)
    pn = (freq + 1e-9) ** 0.75; pn = pn / pn.sum()
    U = nn.Embedding(n, dim).to(DEVICE); V = nn.Embedding(n, dim).to(DEVICE)
    nn.init.normal_(U.weight, std=0.1); nn.init.normal_(V.weight, std=0.1)
    opt_i2v = torch.optim.Adam(list(U.parameters()) + list(V.parameters()), lr=0.01)
    npr_i2v = np.random.default_rng(0)
    for _ in range(epochs):
        perm = npr_i2v.permutation(len(pr))
        for s in range(0, len(pr), bs):
            idx = perm[s:s + bs]; c = torch.tensor(pr[idx], device=DEVICE); o = torch.tensor(pc[idx], device=DEVICE)
            ng = torch.tensor(npr_i2v.choice(n, size=(len(idx), neg), p=pn), device=DEVICE)
            uc = U(c); pos = Fn.logsigmoid((uc * V(o)).sum(-1))
            negl = Fn.logsigmoid(-(uc.unsqueeze(1) * V(ng)).sum(-1)).sum(-1)
            loss = -(pos + negl).mean(); opt_i2v.zero_grad(); loss.backward(); opt_i2v.step()
    W = U.weight.detach()
    return W / (W.norm(dim=1, keepdim=True) + 1e-8)


if os.environ.get("TIMING"):
    nparam = sum(p.numel() for p in model.parameters())
    vb, vg = load_split("valid", VAL_N)
    Rv, Zv = score_all_signals(vb)
    w_ad, prior_keys, best_eb, freq_t, best_ho = fit_simba(vb, vg, Rv, Zv)
    tb, tg = load_split("test", min(TEST_N, 1000))
    t0 = time.time()
    Rt, Zt = score_all_signals(tb)
    fused = apply_simba_full(Zt, w_ad, prior_keys, best_eb, freq_t, best_ho)
    for j, s in enumerate(tb):
        fused[s, j] = -1e9
    _ = fused.topk(20, dim=0).indices.cpu().numpy()
    dt = time.time() - t0
    print(f"{DS} TIMING | learned params: {nparam:,} | catalogue n={n:,} | "
          f"full-ranking inference {dt:.2f}s / {len(tb)} bundles = {1000*dt/len(tb):.2f} ms/bundle | device={DEVICE}", flush=True)
    sys.exit()


if os.environ.get("BACKOFF"):
    t0 = time.time()
    vb, vg = load_split("valid", VAL_N)
    Rv, Zv = score_all_signals(vb)
    w_ad, prior_keys, best_eb, freq_t, best_ho = fit_simba(vb, vg, Rv, Zv)
    w_prec = EmpiricalBayesRouter.precision_backoff_weights(
        Zv, lambda Zk: eval_fused(Zk, vb, vg, Ks=(20,))[20][1], SIG, gamma=1.5, tau_rel=0.20
    )
    tb, tg = load_split("test", TEST_N); CHUNK = 200
    agg = {"SIMBA-Grid+EB": {10: [[], []], 20: [[], []]}, "SIMBA-PrecBackoff": {10: [[], []], 20: [[], []]}}
    for off in range(0, len(tb), CHUNK):
        cb, cg = tb[off:off + CHUNK], tg[off:off + CHUNK]
        Rc, Zc = score_all_signals(cb)
        f1 = apply_simba_full(Zc, w_ad, prior_keys, best_eb, freq_t, best_ho)
        f2 = apply_simba_full(Zc, w_prec, prior_keys, best_eb, freq_t, best_ho)
        for name, fmat in [("SIMBA-Grid+EB", f1), ("SIMBA-PrecBackoff", f2)]:
            m = eval_fused(fmat, cb, cg, Ks=(10, 20))
            for K2 in (10, 20):
                agg[name][K2][0].extend([m[K2][0]] * len(cb))
                agg[name][K2][1].extend([m[K2][1]] * len(cb))
    print(f"\n=== {DS} BACKOFF COMPARISON ({time.time()-t0:.0f}s) ===", flush=True)
    for name in agg:
        a = agg[name]
        print(f"{name:18s} R@10 {np.mean(a[10][0]):.4f} N@10 {np.mean(a[10][1]):.4f} | R@20 {np.mean(a[20][0]):.4f} N@20 {np.mean(a[20][1]):.4f}", flush=True)
    sys.exit()


t = time.time()
content_sig = "audiostyle" if MUSIC else "style"
vb, vg = load_split("valid", VAL_N)
Rv, Zv = score_all_signals(vb)
w_ad, prior_keys, best_eb, freq_t, best_ho = fit_simba(vb, vg, Rv, Zv)
W = train_item2vec_W()
tb, tg = load_split("test", TEST_N); CHUNK = 200; Ks = (10, 20)
methods = ["popularity", "co-occurrence", "content-kNN", "item2vec", "Pairwise-Repl", "SIMBA-NoRes", "Ours"]
agg = {m: {K2: ([], []) for K2 in Ks} for m in methods}
cold_hits, cold_tot = 0, 0
for off in range(0, len(tb), CHUNK):
    cb, cg = tb[off:off + CHUNK], tg[off:off + CHUNK]
    Rc, Zc = score_all_signals(cb)
    popm = zcols(torch.tensor(np.tile(freq[:, None], (1, len(cb))), device=DEVICE))
    Ms = torch.stack([W[s].mean(0) for s in cb], 0); i2v = W @ Ms.T
    pair_repl = apply_simba_full(Zc, w_ad, prior_keys, best_eb, freq_t, (0.0, 0.0))
    no_res = apply_simba_full(Zc, w_ad, prior_keys, (0.0, 1.0), freq_t, best_ho)
    ours_fused = apply_simba_full(Zc, w_ad, prior_keys, best_eb, freq_t, best_ho)

    f_tmp = ours_fused.clone()
    for j, s in enumerate(cb):
        f_tmp[s, j] = -1e9
    top20_c = f_tmp.topk(20, dim=0).indices.cpu().numpy()
    for j, gt in enumerate(cg):
        pset = set(top20_c[:, j])
        for g_item in gt:
            if freq[g_item] == 0:
                cold_tot += 1
                if g_item in pset:
                    cold_hits += 1

    mats = {
        "popularity": popm,
        "co-occurrence": Zc["cooc"],
        "content-kNN": Zc[content_sig],
        "item2vec": i2v,
        "Pairwise-Repl": pair_repl,
        "SIMBA-NoRes": no_res,
        "Ours": ours_fused,
    }
    for m in methods:
        res = eval_fused(mats[m], cb, cg, Ks=Ks)
        for K2 in Ks:
            agg[m][K2][0].extend([res[K2][0]] * len(cb))
            agg[m][K2][1].extend([res[K2][1]] * len(cb))

print(f"\n=== {DS} MATRIX (full-ranking R/N@10,20; test N={len(tb)}; cold R@20={100*cold_hits/max(cold_tot,1):.2f}% [{cold_hits}/{cold_tot}]; {time.time()-t:.0f}s) ===", flush=True)
print(f"{'method':16s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}", flush=True)
for m in methods:
    a = agg[m]
    print(f"{m:16s} {np.mean(a[10][0]):8.4f} {np.mean(a[10][1]):8.4f} {np.mean(a[20][0]):8.4f} {np.mean(a[20][1]):8.4f}", flush=True)
