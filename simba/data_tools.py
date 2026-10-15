"""Fetch benchmark files from original providers, verify local inputs, or import existing data."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen
from .paths import DATASETS, data_root

RETAIL = ("electronic", "food", "clothing")
FASHION = ("pog", "pog_dense")
MUSIC = ("spotify", "spotify_sparse")
SPLITS = ("bi_train.txt", "bi_valid_input.txt", "bi_valid_gt.txt", "bi_test_input.txt", "bi_test_gt.txt")
RETAIL_FILES = ("bundle_item.csv", "user_bundle.csv", "session_bundle.csv", "item_titles.csv", "item_categories.csv", "user_item_pretrain.csv")
BUNDLE_REV = "087da2e2b4d8738ef23f3f6f2005505e8685edde"
HF_REV = "81b05d686972686107f348e202b71936716d2891"


def required_files(dataset):
    if dataset in RETAIL:
        return RETAIL_FILES
    if dataset == "dunnhumby":
        return ("transaction_data.csv", "product.csv")
    base = (*SPLITS, "count.json", "item_info.json", "content_feature.pt")
    return (*base, "description_feature.pt", "title_emb.npy") if dataset in FASHION else base


def source_url(dataset, filename):
    if dataset in RETAIL:
        return f"https://raw.githubusercontent.com/BundleRec/bundle_recommendation/{BUNDLE_REV}/dataset/{dataset}/{filename}"
    if dataset in (*FASHION, *MUSIC) and filename != "title_emb.npy":
        return f"https://huggingface.co/datasets/xhLiu/BundleConstruction/resolve/{HF_REV}/{dataset}/{filename}?download=true"
    return None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dataset, root):
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for name in required_files(dataset):
        path = folder / name
        if path.is_file():
            continue
        url = source_url(dataset, name)
        if not url:
            print(f"MANUAL {dataset}/{name}: see docs/DATA.md", flush=True)
            continue
        print(f"DOWNLOAD {dataset}/{name}", flush=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(url, timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def import_local(dataset, source, root):
    if dataset in RETAIL:
        origin = source / "BundleRec" / "dataset" / dataset
    elif dataset == "dunnhumby":
        origin = source / "dunnhumby"
    else:
        origin = source / "bundlemllm" / "datasets" / dataset
    dest = root / dataset
    dest.mkdir(parents=True, exist_ok=True)
    for name in (*required_files(dataset), "bi_full.txt"):
        src = origin / name
        if src.is_file() and src.resolve() != (dest / name).resolve():
            shutil.copy2(src, dest / name)


def prepare_titles(dataset, root):
    if dataset not in FASHION:
        raise ValueError("Title embedding preparation applies to pog and pog_dense only")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    info = json.loads((root / dataset / "item_info.json").read_text())
    titles = [info[str(i)].get("title") or "" for i in range(len(info))]
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vectors = model.encode(titles, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    np.save(root / dataset / "title_emb.npy", vectors.astype(np.float32))


def check(datasets, root, hashes=False):
    complete = True
    records = []
    for ds in datasets:
        missing = []
        for name in required_files(ds):
            path = root / ds / name
            if not path.is_file():
                missing.append(name)
            else:
                records.append({"dataset": ds, "file": name, "bytes": path.stat().st_size,
                                **({"sha256": sha256(path)} if hashes else {})})
        print(f"{ds}: " + ("READY" if not missing else "MISSING " + ", ".join(missing)), flush=True)
        complete &= not missing
    return complete, records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["fetch", "check", "import-local", "prepare-titles"])
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(RETAIL))
    p.add_argument("--root", type=Path, default=data_root())
    p.add_argument("--source-parent", type=Path)
    p.add_argument("--hashes", action="store_true")
    p.add_argument("--manifest", type=Path, help="Write an inventory of the files actually present")
    args = p.parse_args(argv)
    if args.action == "import-local" and not args.source_parent:
        p.error("import-local requires --source-parent")
    for ds in args.datasets:
        if args.action == "fetch":
            fetch(ds, args.root)
        elif args.action == "import-local":
            import_local(ds, args.source_parent, args.root)
        elif args.action == "prepare-titles":
            prepare_titles(ds, args.root)
    complete, records = check(args.datasets, args.root, args.hashes)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"files": records}, indent=2) + "\n")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
