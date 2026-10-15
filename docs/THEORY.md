# SIMBA: mathematical specification and scope

This document describes the implemented operators. The paper-style version is
[paper/method.tex](../paper/method.tex). SIMBA combines established mathematical
tools in a bundle-completion procedure; Rao diversity and Gaussian conjugacy
are prior results, not newly invented primitives.

## 1. Task and notation

Let B be the binary training-bundle by item incidence matrix. Let S be the
observed seed set and c a candidate outside S. Item exposure is
f_i = sum_b B_bi, bundle size is d_b, and query indicators form I_S.
The implementation uses a_s = (f_s + 5)^(-alpha), v_b = (d_b + 1)^(-gamma).
The default exponents are alpha = gamma = 1/2.

Training bundles provide collaborative evidence. Item text, categories and
frozen content vectors provide catalogue information. Validation queries fit
fusion coefficients and posterior calibration; test targets fit nothing.

## 2. Pairwise backbone

For a fixed item c, additive seed support is

    A_c(S) = sum_s W_cs,
    W = B^T diag(v) B diag(a), with diagonal removed.

For candidates outside S, `score_pairwise_and_diff2` agrees with this expression.
Its intermediate scores may include seed self-contributions; seeds are excluded
before final ranking. Extra user/session and content channels are explicitly
dataset-specific and are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

The symmetric smoothing operator is

    P_sym X = D_a [B^T D_(1/(d+1)) B - diag] D_a X.

The code also exposes P_rw, obtained by zeroing the diagonal of
D_(1/f) B^T D_(1/d) B and normalizing nonzero rows. Such rows sum to one;
isolated/cold rows sum to zero. P_sym is not itself a Markov transition.
Current evaluator smoothing uses P_sym or a dataset-specific symmetric graph.

## 3. Pure joint-incidence surplus

Let o_b = |b intersect S| and u_b = sum_(s in b intersect S) a_s. Define

    Delta_inc(c | S) = sum_(b containing c) v_b u_b beta max(0, o_b - 1).

`score_simplicial_incidence_only` returns this surplus and its smoothed version,
without the additive backbone. For nonnegative beta:

* Delta_inc is nonnegative.
* It is zero for |S| <= 1.
* For c outside S it can be positive only if a training bundle contains c
  together with at least two seeds.

The fraction of queries with any bundle overlapping at least two seeds bounds
the opportunity for direct activation. This is not a ranking-improvement theorem.
Smoothing may propagate an activated signal to candidates outside that bundle.

### Information separation

Consider these two four-edge hypergraphs on items 0..5:

    H1: 012, 034, 135, 245
    H2: 013, 024, 125, 345

They have exactly the same full B^T B, including the diagonal. Bundle sizes and
item frequencies also match. Their pairwise transition operators are identical.
For S={0,1}, candidate 2 has joint three-item incidence in H1 but not H2, so
Delta_inc distinguishes them. This is an existence result about extra information
in B; it does not imply every pair of different hypergraphs is distinguishable.

## 4. Pairwise seed concordance

Let nonnegative w_cs be candidate support from seed s. In the Rao implementation,
this includes the pairwise transition plus an optional positive content-similarity
term. Write A_c = sum_s w_cs and p_cs = w_cs/A_c when A_c>0.

The unweighted second-order score is

    R_0 = [(sum_s w_cs)^2 - sum_s w_cs^2] / (A_c + tau)
        = A_c^2/(A_c+tau) [1 - sum_s p_cs^2].

The identity follows by expanding the square. If A_c=0, define R_0=0.
It is zero when at most one seed supplies support. At fixed A_c, spreading
support across more seeds increases the Gini–Simpson factor.

It is a second-order interaction, not strict conjunction across all seeds:
(1.5,1.5,0) has larger e2 than (2.1,0.45,0.45), although the latter has a
larger full product. No Product-of-Experts ranking equivalence is claimed.

### Rao weighting for redundancy

For normalized content vectors, the seed similarity is

    G_ss' = 0.5 * 1[category_s = category_s']
          + 0.5 * max(0, cosine(x_s,x_s')),

with diagonal one. With only one information source available, that source
defines G. In complementary mode K_ss'=1-G_ss' off the diagonal, and

    R_Rao(c | S) = w_c^T K w_c / (A_c + tau).

This is Rao's quadratic diversity of the support shares, multiplied by
A_c^2/(A_c+tau). Completely redundant seeds (G=1) receive zero agreement bonus.
Dissimilarity is a proxy for distinct constraints, not proof of functional
complementarity or statistical independence. Partially duplicated seed groups
need not be duplication-invariant. The music evaluator uses K=1+G off-diagonal
to favor thematic coherence instead of complementary slots.

If 0<=K<=1, then 0<=R_Rao<=A_c^2/(A_c+tau)<=A_c. Joint scaling obeys

    R(lambda*w; lambda*tau) = lambda*R(w; tau).

With fixed tau, exact degree-one homogeneity does not hold; approximate linear
scaling requires both relevant support scales to be large compared with tau.
Rao aggregation uses pairwise/feature support and cannot recover joint incidence
lost by that support representation. It is conceptually distinct from Delta_inc;
no statistical orthogonality or independence between channels is assumed.

## 5. Exposure-modulated Gaussian shrinkage

Define theta as standardized latent compatibility, not a calibrated membership
probability or a log-lift estimator. Conditional on the fitted prior mean mu0:

    theta | mu0 ~ Normal(mu0, sigma0^2)
    Y | theta,f ~ Normal(theta, sigma_eps^2/f), for f>0.

Under this model, completing the square gives posterior precision
1/sigma0^2 + f/sigma_eps^2 and posterior mean

    theta_hat = (f*Y + kappa*mu0)/(f+kappa)
              = (1-u)*Y + u*mu0,
    kappa = sigma_eps^2/sigma0^2, u=kappa/(f+kappa).

For positive kappa and f=0, the implementation uses the zero-likelihood-precision
limit theta_hat=mu0. The update is a convex combination. For kappa=0, the explicit
software policy is to disable shrinkage and return Y, including cold rows.
The algebra is exact conditional on the model; Gaussian errors, exposure scaling
and prior calibration are assumptions requiring empirical assessment.

### Raw OLS versus constrained plug-in estimates

The model implies E[(Y-mu0)^2 | f] = sigma0^2 + sigma_eps^2/f.
For validation queries, regress v_c=mean_S[(Y_cS-mu0_cS)^2] on x_c=1/f_c
over warm items. With nonconstant x, the raw slope and intercept are the
unrounded OLS estimates. Unbiasedness refers to those raw components under
the conditional-moment model, not to their ratio or the constrained values.

The implementation floors both components at 1e-4 and clips their ratio to
[0.05,50]. These are admissibility choices, not unbiased transformations.
It returns both versions. With fewer than two warm items or numerically zero
inverse-frequency variation, it reports non-identification and kappa=0.
Every evaluator excludes positive tuning candidates in that case.

## 6. Validation-frozen inference

Fit the prior scale from validation scores only:

    prior_scale = std(Y_validation)/std(raw_prior_validation).

Freeze it, the selected kappa and fusion parameters before test evaluation.
Catalogue z-normalization is per query column; adding unrelated query columns
must not change an existing query's score. The suite checks this property and
the production scoring function was independently checked on the earlier
batch-dependence counterexample. Floating-point hardware differences and the
evaluators' stochastic tie perturbations can still affect exact tied rankings.

## 7. Scope of evidence

The small encoder's parameter count excludes graphs, catalogue vectors and
score matrices. Full-catalogue computation may require several GiB of RAM.
The supplied toy example demonstrates execution, not benchmark performance.
Reported gains, confidence and protocol limitations are in
[RESULTS.md](RESULTS.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## References

* Zhou, Huang and Schölkopf (2006), [Learning with Hypergraphs](https://papers.nips.cc/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf).
* Rao (1982), [Diversity and dissimilarity coefficients: a unified approach](https://doi.org/10.1016/0040-5809(82)90004-1).
* Rendle (2010), [Factorization Machines](https://doi.org/10.1109/ICDM.2010.127).
* Ma et al. (2024), [CLHE](https://arxiv.org/abs/2310.18770).
* Nguyen et al. (2025), [RaMen](https://arxiv.org/abs/2507.14361).
