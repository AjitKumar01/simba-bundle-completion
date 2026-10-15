# Evidence and result provenance

The current controlled retail ablation uses fixed base coefficients and
validation-frozen prior scale and variance calibration. Scores below are mean
full-catalogue Recall@20 across five split seeds, expressed as percentages.

| Dataset | Pairwise | + Incidence | + Rao | + Both | + Posterior | Provenance |
|---|---:|---:|---:|---:|---:|---|
| Electronic | 49.73 | 50.03 | 50.05 | 50.25 | 52.35 | Author-reported; independent 90-second attempt incomplete |
| Food | 56.19 | 56.27 | 56.33 | 56.43 | 59.34 | Independently rerun from fd79d50, five splits |
| Clothing | 55.30 | 55.30 | 55.27 | 55.27 | 59.02 | Independently rerun from fd79d50, five splits |

The posterior gains reported from unrounded output are +3.16 percentage points
on Food and +3.72 on Clothing. Displayed rounded means can differ by 0.01 pp
from differences computed before rounding. No standard deviation is reconstructed
from these rounded means. New ablation runs save per-seed values and sample
standard deviations as JSON.

The packaging changes preserve those score computations. The non-identification
guard preserves the existing grids for all 15 retail training splits, whose
inverse frequencies have nonzero variation. No new five-seed benchmark was
performed merely to package this repository.

Higher-order effects are modest and dataset-dependent. Most measured gain in
these rows comes from posterior calibration. The table does not establish a
universal higher-order advantage or superiority over original multimodal CLHE
and RaMen. Prior metadata-adapted baseline experiments are not implementations
shipped in this repository and are not advertised as reproducible here.

POG/music/Dunnhumby code is included, but the current controlled ablation driver
implements retail only. Earlier additional-domain claims and historical baseline
scores are not newly verified by this release. The full historical CSV is
retained solely as an explicitly labelled archive.
