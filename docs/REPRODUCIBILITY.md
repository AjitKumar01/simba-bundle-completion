# Reproducibility and evaluation protocol

## Quick verification

```bash
python tests/test_audit.py
python -m unittest discover -s tests -p 'test_standalone.py'
python -m simba.demo --output artifacts/demo.json
```

The core tests check stochastic active rows, incidence separation, concordance
identities, redundant-seed behavior, posterior equality, variance estimation,
frozen batch calibration and every evaluator's non-identification policy.
The packaging tests check the offline example, paths and fail-closed orchestration.

## Controlled retail ablation

```bash
python -m simba.ablation_study --datasets electronic food clothing --seeds 5 \
  --output results/generated
```

The driver deduplicates item sets before the 70/10/20 split. Split seeds are
0..4, masking seeds are split_seed+1 for validation and +2 for test. Half of
each bundle is observed. Every retained retail test query is ranked over the
full item catalogue with input seeds excluded. Base coefficients stay fixed
across steps. Prior scale and variance components fit validation scores only.

JSON outputs contain each split's Recall@20, its mean and sample standard
deviation (`ddof=1`). A one-split run reports standard deviation as null.
Five overlapping split runs are not five independent populations; descriptive
variation must not be represented as a population-level significance guarantee.

## Standard evaluator matrix

```bash
python -m simba.run_matrix food clothing --seeds 5
python -m simba.run_matrix --seeds 5
```

The second command runs all eight datasets and can take substantial time.
The driver stores worker logs, run metadata and `measured_matrix.csv` under
`results/generated`. It fails on worker failure, missing methods or invalid
metrics. It does not add CLHE/RaMen published numbers to measured output.

Standard evaluators have dataset-specific feature banks and tuning grids.
The POG/music evaluator caps validation at 400 queries and test at 3,000;
Dunnhumby caps validation at 800 and test at 2,000. Full catalogue ranking does
not mean all available queries are evaluated. Retail uses all retained queries.
Standard neural training initialization is fixed; split variation differs from
initialization variation. Some standard evaluators use small random score
perturbations for ties; exact tied rankings can vary across executions/devices.

Dunnhumby retains the original 30,000-basket sample, basket-size limits and a
global minimum-frequency filter of two. That filter uses held-out activity to
select retained items, which limits strict inductive interpretation. Chronological
runs share one temporal boundary and vary masking, not independent time windows.
These limitations are documented rather than silently changing the existing
benchmark protocol during packaging.

## Resource expectations

The demo and operator tests need seconds on CPU. Benchmark runs train encoders,
build graphs and allocate catalogue-by-query score matrices; music has hundreds
of thousands of items and can require several GiB of RAM. There is no claim
that the small trainable encoder implies equally small total memory.

For comparable CPU timing, set `FORCE_DEVICE=cpu`, `OMP_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, and `VECLIB_MAXIMUM_THREADS=2`, and avoid concurrent jobs.
`requirements-repro.txt` records versions in the local verification environment.
It is not a cross-platform binary lockfile.

Historical scores in `results/historical_matrix.csv` predate the current
controlled protocol and are never automatically treated as new measurements.
