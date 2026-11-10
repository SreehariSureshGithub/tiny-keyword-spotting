"""Build the binary train/val/test dataset from Google Speech Commands.

Expected input: an extracted Speech Commands folder, i.e. one sub-folder per word
(`stop/`, `go/`, `yes/`, ...) plus `testing_list.txt` and `validation_list.txt`.

Output layout (what train.py / evaluate.py read):

    Dataset/
      train/0/*.wav   train/1/*.wav
      val/0/*.wav     val/1/*.wav
      test/0/*.wav    test/1/*.wav

Label 1 = the keyword, label 0 = every other selected word.
The official validation/testing lists are used, so the split is speaker-independent.

Examples:
    python prepare_dataset.py --source data/speech_commands_v0.01
    python prepare_dataset.py --source data/speech_commands_v0.01 --negatives go no up down
"""

import argparse
import shutil
from pathlib import Path

from kws.config import DATASET_DIR


def read_list(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="extracted Speech Commands folder")
    parser.add_argument("--out", default=DATASET_DIR, type=Path, help="output folder (default: ./Dataset)")
    parser.add_argument("--keyword", default="stop", help="word to detect (default: stop)")
    parser.add_argument("--negatives", nargs="*", default=None,
                        help="words to use as the negative class (default: all other words)")
    args = parser.parse_args()

    src: Path = args.source
    test_set = read_list(src / "testing_list.txt")
    val_set = read_list(src / "validation_list.txt")

    words = sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if args.keyword not in words:
        raise SystemExit(f"Keyword folder '{args.keyword}' not found in {src}")
    negatives = args.negatives if args.negatives else [w for w in words if w != args.keyword]

    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            (args.out / split / label).mkdir(parents=True, exist_ok=True)

    counts = {(s, l): 0 for s in ("train", "val", "test") for l in ("0", "1")}
    for word in [args.keyword, *negatives]:
        label = "1" if word == args.keyword else "0"
        for wav in (src / word).glob("*.wav"):
            rel = f"{word}/{wav.name}"
            split = "test" if rel in test_set else "val" if rel in val_set else "train"
            # Speech Commands reuses <speaker>_nohash_<n>.wav across words, so prefix
            # the word to stop clips from different words overwriting each other.
            shutil.copy2(wav, args.out / split / label / f"{word}_{wav.name}")
            counts[(split, label)] += 1

    print(f"Keyword: {args.keyword} | negatives: {len(negatives)} words")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}  keyword={counts[(split, '1')]:6d}  other={counts[(split, '0')]:6d}")
    print(f"Done. Dataset written to {args.out}")


if __name__ == "__main__":
    main()
