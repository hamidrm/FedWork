import argparse, os, random, shutil
from pathlib import Path
from typing import List

def list_images(d: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in exts])

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)   # hardlink (no extra disk)
    except OSError:
        shutil.copy2(src, dst)  # fallback

def split_and_link(src_class_dir: Path, out_root: Path, splits, seed: int):
    rng = random.Random(seed)
    imgs = list_images(src_class_dir)
    rng.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * splits["train"])
    n_val   = int(n * splits["val"])
    n_test  = n - n_train - n_val

    parts = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train+n_val],
        "test":  imgs[n_train+n_val:] if splits["test"] > 0 else [],
    }

    cls_name = src_class_dir.name  # "person" or "non_person"
    for split, files in parts.items():
        if split == "test" and splits["test"] == 0:
            continue
        for i, src in enumerate(files):
            dst = out_root / split / cls_name / f"{src.stem}_{i:06d}{src.suffix.lower()}"
            link_or_copy(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Path containing two dirs: person/ and non_person/")
    ap.add_argument("--out", default="./data/vww",
                    help="Output root to create train/val(/test) tree")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio",   type=float, default=0.2)
    ap.add_argument("--test-ratio",  type=float, default=0.0,
                    help="Set >0 to also make a test split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    assert (src / "person").is_dir() and (src / "non_person").is_dir(), \
        "Expected src/person and src/non_person directories."

    tot = args.train_ratio + args.val_ratio + args.test_ratio
    assert abs(tot - 1.0) < 1e-6, "train + val + test ratios must sum to 1.0"

    splits = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}

    # Clean output if exists? (comment out if you prefer to append)
    if out.exists():
        print(f"[INFO] Removing existing {out} …")
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    if args.test_ratio > 0:
        (out / "test").mkdir(parents=True, exist_ok=True)

    # Process both classes
    for cls in ["person", "non_person"]:
        split_and_link(src / cls, out, splits, seed=args.seed)

    print("[DONE] Created split at:", out.resolve())
    for split in ["train", "val"] + (["test"] if args.test_ratio > 0 else []):
        for cls in ["person", "non_person"]:
            p = out / split / cls
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f" {split}/{cls}: {n} images")

if __name__ == "__main__":
    main()
