"""Run measured SIMBA evaluators; historical baseline numbers are never injected."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from .paths import DATASETS, PROJECT_ROOT

METHODS = ("popularity", "co-occurrence", "content-kNN", "item2vec", "Pairwise-Repl", "SIMBA-NoRes", "Ours")
ROW = re.compile(r"^(" + "|".join(map(re.escape, METHODS)) + r")\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def parse_rows(stdout):
    rows = {}
    for line in stdout.splitlines():
        match = ROW.match(line.strip())
        if match:
            values = [float(v) for v in match.groups()[1:]]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f"Invalid measured metrics: {line}")
            rows[match.group(1)] = values
    missing = set(METHODS) - rows.keys()
    if missing:
        raise ValueError(f"Missing measured methods: {sorted(missing)}")
    return rows


def run_worker(dataset, env, log):
    if dataset in ("electronic", "food", "clothing"):
        module = "simba.bundlerec_eval"
        env["BRDS"] = dataset
    elif dataset == "dunnhumby":
        module = "simba.dunnhumby_eval"
    else:
        module = "simba.fullrank_eval"
        env["BUNDLE_DS"] = dataset
    process = subprocess.run([sys.executable, "-u", "-m", module], cwd=PROJECT_ROOT,
                             env=env, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + "\nSTDERR\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(f"{dataset} exited {process.returncode}; see {log}")
    return parse_rows(process.stdout)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datasets", nargs="*", choices=DATASETS)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/generated")
    args = p.parse_args(argv)
    if args.seeds < 1:
        p.error("--seeds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    table = {}
    for ds in args.datasets or DATASETS:
        env = dict(os.environ, SEEDS=str(args.seeds))
        if ds in ("pog", "pog_dense", "spotify", "spotify_sparse"):
            runs = []
            for seed in range(args.seeds):
                print(f"RUN {ds} split {seed}", flush=True)
                runs.append(run_worker(ds, dict(env, RAMEN_SEED=str(seed)), args.output / f"{ds}_seed{seed}.log"))
            table[ds] = {method: [sum(run[method][k] for run in runs)/len(runs) for k in range(4)] for method in METHODS}
        else:
            print(f"RUN {ds} {args.seeds} splits", flush=True)
            table[ds] = run_worker(ds, env, args.output / f"{ds}.log")
    with (args.output / "measured_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "R@10", "N@10", "R@20", "N@20"])
        for ds, methods in table.items():
            for method, values in methods.items():
                writer.writerow([ds, method, *values])
    (args.output / "run.json").write_text(json.dumps({"requested_split_runs": args.seeds,
        "datasets": list(table), "metric_source": "current evaluator stdout only",
        "limitation": "Worker summaries are rounded to four decimals; ablation JSON contains per-seed Recall@20."}, indent=2) + "\n")
    print(args.output / "measured_matrix.csv")


if __name__ == "__main__":
    main()
