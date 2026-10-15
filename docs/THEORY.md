# SIMBA theory

The detailed mathematical specification is maintained in **[THEORY.tex](THEORY.tex)**.
It is a standalone LaTeX document with definitions, derivations, propositions,
proofs, counterexamples, implementation correspondence, and references.
Read the [compiled PDF](THEORY.pdf) directly without a LaTeX installation.

It covers:

- Pairwise induction, symmetric smoothing, and active-row stochastic diffusion.
- Pure incidence surplus, its exact triple-incidence representation, and a
  degree-matched separation from identical pairwise projections.
- Incidence activation and the limits of any ranking-improvement claim.
- Elementary symmetric polynomials, Rao diversity, score bounds, redundancy,
  permutation invariance, duplication caveats, and exact scaling laws.
- Gaussian posterior derivation, cold-item limits, conditional Bayes risk,
  variance-component OLS, identifiability, truncation, and model misspecification.
- Validation-frozen batch invariance, controlled ablation semantics,
  computational costs, optional weighted-input caveats, and unresolved questions.

From the repository root, compile with a standard LaTeX installation:

```bash
mkdir -p artifacts/theory
for pass in 1 2 3; do
  pdflatex -halt-on-error -interaction=nonstopmode \
    -output-directory=artifacts/theory docs/THEORY.tex || exit 1
done
```

The output is `artifacts/theory/THEORY.pdf`. Repeated passes resolve the table
of contents and references. To refresh the checked-in PDF after editing the
source, copy that output to `docs/THEORY.pdf`. No external figures, BibTeX database, benchmark
downloads, or Python dependencies are needed to compile the theory document.

The shorter [paper methodology](../paper/method.tex) remains separate.
See [RESULTS.md](RESULTS.md) for measured evidence and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for experiment protocols.
