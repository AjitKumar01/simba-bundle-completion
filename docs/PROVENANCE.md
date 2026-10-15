# Source and packaging provenance

This standalone repository starts from the author's SIMBA commit
`fd79d501306e0615e53f9902dc78fa20f3ecd329` plus the reviewed non-identification
guard and reporting corrections. Earlier source commits discussed during review
were `4d93eff`, `83b4ae8` and `9dd1fd2`.

Packaging adds installation metadata, an offline synthetic demo, shared local
data paths, original-source download instructions, test coverage and CI. It
also separates historical scores from new measured output and saves per-seed
retail ablation records. The core scoring equations and default ablation
coefficients are retained.

The original source checkout and the larger parent research project are not
required after installation and data download. No credentials, machine-specific
virtual environments or raw third-party datasets are part of the Git history.
