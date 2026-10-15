# Download and use the benchmark data

The repository includes original synthetic example data in
`simba/example_data.json`. The real datasets retain their providers' terms and
are downloaded from the original sources. No third-party benchmark files or
private account information are committed.

## Retail: Electronic, Food and Clothing

Source: [BundleRec dataset](https://github.com/BundleRec/bundle_recommendation/tree/087da2e2b4d8738ef23f3f6f2005505e8685edde/dataset)
and [dataset description](https://github.com/BundleRec/bundle_recommendation/blob/087da2e2b4d8738ef23f3f6f2005505e8685edde/dataset/README.md).
The downloader pins upstream commit `087da2e2b4d8738ef23f3f6f2005505e8685edde`.

```bash
python -m simba.data_tools fetch --datasets electronic food clothing
python -m simba.data_tools check --datasets electronic food clothing
python -m simba.ablation_study --datasets food --seeds 5
BRDS=food python -m simba.bundlerec_eval
NO_AUX=1 BRDS=food python -m simba.bundlerec_eval
```

Each `data/<domain>/` needs `bundle_item.csv`, `user_bundle.csv`,
`session_bundle.csv`, `item_titles.csv`, `item_categories.csv`, and
`user_item_pretrain.csv`. The last file is a separate auxiliary log whose time
range is not guaranteed to precede all bundles. `NO_AUX=1` disables it in the
standard retail evaluator; the fixed retail ablation includes it. The unsplit
`user_item.csv` is not used to build collaborative graphs.

## POG, POG_dense, Spotify and Spotify_sparse

Source: [xhLiu/BundleConstruction](https://huggingface.co/datasets/xhLiu/BundleConstruction/tree/main),
also linked by [Bundle-MLLM](https://github.com/Xiaohao-Liu/Bundle-MLLM).
The downloader pins revision `81b05d686972686107f348e202b71936716d2891`.

```bash
python -m simba.data_tools fetch --datasets pog pog_dense spotify spotify_sparse
```

Fashion data additionally require locally generated title embeddings. After
the raw files download, generate them with the same encoder family used in
the original local preparation:

```bash
python -m pip install -e '.[data]'
python -m simba.data_tools prepare-titles --datasets pog pog_dense
python -m simba.data_tools check --datasets pog pog_dense spotify spotify_sparse
BUNDLE_DS=pog python -m simba.fullrank_eval
BUNDLE_DS=spotify python -m simba.fullrank_eval
```

`fetch` reports the missing title embeddings and exits nonzero until they are
prepared. The encoder is `BAAI/bge-small-zh-v1.5`, with normalized float32 output
in `title_emb.npy`. The original preparation did not record a model revision;
regeneration is not a claim of bitwise reproduction of historical embeddings.
Record your model revision and file hashes for a new study.

Each directory needs `count.json`, `item_info.json`, `content_feature.pt`, and
the five `bi_train`, `bi_valid_input`, `bi_valid_gt`, `bi_test_input`, and
`bi_test_gt` text files. Fashion also uses `description_feature.pt` and
`title_emb.npy`. If an existing `bi_full.txt` is imported it is used; otherwise
full bundles are reconstructed from the supplied train/input/target files.
The script never downloads upstream collaborative-factor files for scoring.

## Dunnhumby: The Complete Journey

Download **The Complete Journey** from [dunnhumby Source Files](https://www.dunnhumby.com/source-files/).
Follow the provider's download flow and terms, extract the archive, and place
`transaction_data.csv` and `product.csv` in `data/dunnhumby/`.

```bash
python -m simba.data_tools check --datasets dunnhumby
SPLIT_MODE=random python -m simba.dunnhumby_eval
SPLIT_MODE=chrono python -m simba.dunnhumby_eval
```

The other marketing/promotion CSVs are not inputs to SIMBA. The source data
are not rehosted here; the provider's [terms](https://www.dunnhumby.com/terms-and-conditions/)
apply to them.

## Existing local copies and inventories

For an existing project containing `BundleRec/dataset`, `bundlemllm/datasets`
and `dunnhumby`, copy the required files into this repository with:

```bash
python -m simba.data_tools import-local --source-parent /path/to/old/project \
  --datasets electronic food clothing pog pog_dense spotify spotify_sparse dunnhumby
python -m simba.data_tools check --datasets electronic food clothing \
  --hashes --manifest artifacts/data_inventory.json
```

`check` verifies required-file presence. `--hashes` records SHA-256 hashes of
the files actually present; it does not certify equality with a historical
benchmark. Set `SIMBA_DATA_DIR=/path/to/data` to use another root without
copying files. Per-dataset overrides `DATA_DIR`, `BR_DIR`, and `DUNN_DIR` are
also supported by their evaluators.

Please cite the original dataset publications; see [CITATIONS.md](CITATIONS.md).
