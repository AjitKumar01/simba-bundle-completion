# SIMBA — Bundle Completion

SIMBA completes a partial bundle using three distinct sources of evidence:
**joint incidence inside historical bundles**, **redundancy-aware agreement
across query seeds**, and **exposure-modulated Gaussian posterior calibration**.

This repository contains the operators, eight-dataset evaluator family, a
controlled retail ablation, theory documentation, installation metadata and
an offline example with included synthetic data. It runs independently of the
original parent research project. Real benchmarks are downloaded separately
from their original providers using the links and instructions below.

## Install and run an offline example

Python 3.10+ is required. CPU is supported; CUDA and Apple MPS are optional.

```bash
git clone https://github.com/AjitKumar01/simba-bundle-completion.git
cd simba-bundle-completion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m simba.demo --output artifacts/demo.json
```

The demo fits calibration on its synthetic validation queries and ranks its
synthetic test queries. It needs no benchmark download or external model.
`simba-demo` and `simba-data` are also installed as command-line entry points.
The example is a functionality check, not a benchmark-performance claim.

## Theory and implementation

* [Detailed LaTeX theory: derivations, proofs, assumptions and limits](docs/THEORY.tex)
  ([compiled PDF](docs/THEORY.pdf); [contents and build instructions](docs/THEORY.md))
* [Architecture and code map](docs/ARCHITECTURE.md)
* [Paper-style methodology](paper/method.tex), with [standalone LaTeX wrapper](paper/main.tex)
* [Evaluation protocol and reproduction commands](docs/REPRODUCIBILITY.md)
* [Results and provenance](docs/RESULTS.md)
* [Source history](docs/PROVENANCE.md) and [citations](docs/CITATIONS.md)

The core statistical update is the exact normal–normal posterior mean under
the stated observation model:

\[
\widehat\theta_{S,c}=(1-u_c)Y(c\mid S)+u_c\mu_0(c\mid S),
\qquad u_c=\frac{\kappa_f}{f_c+\kappa_f}.
\]

Raw OLS variance components and constrained plug-in estimates are reported
separately. If the components are non-identifiable, every evaluator restricts
its posterior search to `kappa_f=0`, disabling shrinkage. Prior scales and
selected parameters are frozen on validation before test scoring.

## Download and use the data

| Benchmarks | Original download source | Local location |
|---|---|---|
| Electronic, Food, Clothing | [BundleRec](https://github.com/BundleRec/bundle_recommendation/tree/087da2e2b4d8738ef23f3f6f2005505e8685edde/dataset) | `data/electronic`, `data/food`, `data/clothing` |
| POG, POG_dense, Spotify, Spotify_sparse | [Hugging Face: xhLiu/BundleConstruction](https://huggingface.co/datasets/xhLiu/BundleConstruction/tree/main) | `data/<dataset>` |
| Dunnhumby | [The Complete Journey — Source Files](https://www.dunnhumby.com/source-files/) | `data/dunnhumby` |

For retail, download only the six files needed by the implementation:

```bash
python -m simba.data_tools fetch --datasets electronic food clothing
python -m simba.data_tools check --datasets electronic food clothing
python -m simba.ablation_study --datasets food clothing --seeds 5
```

For POG/music, download upstream files, then prepare the additional fashion
title embeddings:

```bash
python -m simba.data_tools fetch --datasets pog pog_dense spotify spotify_sparse
python -m pip install -e '.[data]'
python -m simba.data_tools prepare-titles --datasets pog pog_dense
BUNDLE_DS=pog python -m simba.fullrank_eval
BUNDLE_DS=spotify python -m simba.fullrank_eval
```

The first command reports missing fashion title embeddings until the preparation
step completes. Download Dunnhumby through its provider's site and copy
`transaction_data.csv` and `product.csv` into `data/dunnhumby`, then run:

```bash
python -m simba.data_tools check --datasets dunnhumby
SPLIT_MODE=random python -m simba.dunnhumby_eval
```

[Detailed data instructions](docs/DATA.md) explain required files, local imports,
title preprocessing, checksums, auxiliary logs and path overrides. Set
`SIMBA_DATA_DIR=/path/to/data` to use a different root. Real data are not rehosted;
their upstream terms apply independently of this repository's code license.

## Verify and reproduce

```bash
python tests/test_audit.py
python -m unittest discover -s tests -p 'test_standalone.py'
python -m simba.ablation_study --datasets electronic food clothing --seeds 5 \
  --output results/generated
python -m simba.run_matrix food clothing --seeds 5
```

The ablation writes per-seed Recall@20, means and sample standard deviations
to JSON. The matrix driver writes current measured results and worker logs;
it fails on worker errors or missing/invalid metrics and never inserts
published baseline values. Full benchmark runs can take substantial time and
memory. The default POG/music and Dunnhumby evaluators cap test-query counts;
see the [protocol](docs/REPRODUCIBILITY.md).

## Controlled retail results

Mean Recall@20 across five split seeds, in percent. Base coefficients stay
fixed; posterior scale and variance fit validation only.

| Dataset | Pairwise | + Incidence | + Rao | + Both | + Posterior | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Electronic | 49.73 | 50.03 | 50.05 | 50.25 | **52.35** | Author-reported; independent five-seed attempt incomplete |
| Food | 56.19 | 56.27 | 56.33 | 56.43 | **59.34** | Independently reproduced |
| Clothing | 55.30 | 55.30 | 55.27 | 55.27 | **59.02** | Independently reproduced |

Most gain in these rows comes from posterior calibration. Higher-order effects
are modest and dataset-dependent. These results establish neither universal
higher-order superiority nor a comparison against original multimodal CLHE/RaMen.
Older scores are kept in `results/historical_matrix.csv` as an explicit archive.

## License and attribution

SIMBA code and the original synthetic example use the [MIT license](LICENSE).
Third-party datasets and embedding models retain their respective terms.
Please cite their original publications using [CITATIONS.md](docs/CITATIONS.md).
