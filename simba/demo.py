"""Run the three core estimators on included synthetic data without downloads."""
import argparse
import json
from importlib.resources import files
from pathlib import Path
import numpy as np
from .operators import SimplicialGraphOperator, EmpiricalBayesRouter, zcols


def run(data, k=5):
    feat = np.asarray(data["features"], dtype=np.float32)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    cat = np.asarray(data["categories"], dtype=int)
    op = SimplicialGraphOperator(data["train"], len(feat), svd_dim=min(8, len(feat)-2))

    def score(queries):
        seeds = [s for s, _ in queries]
        pair, diffusion = op.score_pairwise_and_diff2(seeds)
        incidence, _ = op.score_simplicial_incidence_only(seeds)
        rao = op.score_rao_concordance(seeds, feat, cat, gamma=-1, alpha_dense=.2)
        prior = zcols(np.stack([feat @ feat[s].mean(0) for s in seeds], axis=1))
        Y = zcols(pair) + .25*zcols(diffusion) + .25*zcols(incidence) + .35*zcols(rao) + .5*prior
        return Y, prior

    Yv, prior_v = score(data["validation"])
    scale = float(np.std(Yv)/(np.std(prior_v)+1e-9))
    vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Yv, scale*prior_v)
    Y, prior = score(data["test"])
    posterior = EmpiricalBayesRouter.conjugate_posterior(Y, prior*scale, op.freq, vc["kappa_f"])
    recommendations = []
    for j, (seeds, targets) in enumerate(data["test"]):
        scores = posterior[:, j].copy()
        scores[seeds] = -np.inf
        ranked = np.lexsort((np.arange(len(feat)), -scores))[:min(k, len(feat)-len(seeds))]
        recommendations.append({"seeds": seeds, "targets": targets, "recommendations": ranked.tolist(),
                                "recall": len(set(ranked)&set(targets))/len(targets)})
    return {"dataset": "included synthetic example", "k": k,
            "validation_calibration": {"prior_scale": scale, **vc},
            "mean_recall": float(np.mean([r["recall"] for r in recommendations])),
            "queries": recommendations}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)
    if args.k < 1:
        p.error("--k must be positive")
    data = json.loads(args.data.read_text() if args.data else files("simba").joinpath("example_data.json").read_text())
    result = run(data, args.k)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
