#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
SOURCE_NAME_RULES = ("keep", "binary")


@dataclass(frozen=True)
class FileEntry:
    dataset: str
    source: str
    video: str
    rel_path: str
    existing_split: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan datasets and generate config txt file lists for train/val/test. "
            "Supports both dataset/source/video/frame and dataset/split/source/video/frame layouts."
        )
    )
    parser.add_argument("--datasets-root", default="datasets", help="Root path of raw datasets")
    parser.add_argument("--output-root", default="config/datasets", help="Root path to write txt file lists")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to process. Empty means all dataset folders under datasets-root",
    )
    parser.add_argument(
        "--ratios",
        default="0.8,0.1,0.1",
        help="Split ratios for train,val,test when no split exists in dataset path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for video-level split")
    parser.add_argument(
        "--respect-existing-splits",
        action="store_true",
        help="Use split folder from path when present (train/val/test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing txt files if they exist",
    )
    parser.add_argument(
        "--source-name-rule",
        default="keep",
        choices=SOURCE_NAME_RULES,
        help="How to normalize source names in output txt files: keep or binary(real/fake)",
    )
    return parser.parse_args()


def parse_ratios(ratio_text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in ratio_text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--ratios must contain three comma-separated numbers, for train,val,test")

    ratios = tuple(float(x) for x in parts)
    if any(x < 0 for x in ratios):
        raise ValueError("Ratios must be non-negative")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios sum must be > 0")

    return tuple(x / total for x in ratios)


def normalize_source_name(source: str, rule: str) -> str:
    if rule == "keep":
        return source
    if rule == "binary":
        return "real" if "real" in source.lower() else "fake"
    raise ValueError(f"Unknown source name rule: {rule}")


def infer_entry(
    datasets_root: Path,
    file_path: Path,
    respect_existing_splits: bool,
    source_name_rule: str,
) -> FileEntry | None:
    rel_parts = file_path.relative_to(datasets_root).parts

    # Need at least dataset/source/frame
    if len(rel_parts) < 3:
        return None

    dataset = rel_parts[0]
    split = None

    # Layout A: dataset/split/source/video/frame
    if respect_existing_splits and len(rel_parts) >= 4 and rel_parts[1] in SPLITS:
        split = rel_parts[1]
        source = rel_parts[2]
        # If no dedicated video folder exists, group by source.
        video = rel_parts[3] if len(rel_parts) >= 5 else f"{source}__single"
    else:
        # Layout B: dataset/source/video/frame
        source = rel_parts[1]
        video = rel_parts[2] if len(rel_parts) >= 4 else f"{source}__single"

    source = normalize_source_name(source, source_name_rule)

    rel_path = file_path.as_posix()
    return FileEntry(dataset=dataset, source=source, video=video, rel_path=rel_path, existing_split=split)


def scan_entries(
    datasets_root: Path,
    dataset_filter: set[str],
    respect_existing_splits: bool,
    source_name_rule: str,
) -> list[FileEntry]:
    entries: list[FileEntry] = []

    for image_path in datasets_root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            dataset_name = image_path.relative_to(datasets_root).parts[0]
        except IndexError:
            continue

        if dataset_filter and dataset_name not in dataset_filter:
            continue

        entry = infer_entry(datasets_root, image_path, respect_existing_splits, source_name_rule)
        if entry is not None:
            entries.append(entry)

    return sorted(entries, key=lambda x: x.rel_path)


def split_videos(video_names: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, set[str]]:
    unique_videos = sorted(set(video_names))
    rng = random.Random(seed)
    rng.shuffle(unique_videos)

    n = len(unique_videos)

    # Keep practical defaults for very small datasets.
    if n == 1:
        return {"train": {unique_videos[0]}, "val": set(), "test": set()}
    if n == 2:
        return {"train": {unique_videos[0]}, "val": set(), "test": {unique_videos[1]}}

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    # Keep at least one test video when possible.
    if n >= 3 and n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    train_videos = set(unique_videos[:n_train])
    val_videos = set(unique_videos[n_train : n_train + n_val])
    test_videos = set(unique_videos[n_train + n_val :])

    return {"train": train_videos, "val": val_videos, "test": test_videos}


def build_output_index(entries: list[FileEntry], ratios: tuple[float, float, float], seed: int) -> dict[str, dict[str, dict[str, list[str]]]]:
    # dataset -> split -> source -> [paths]
    index: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    grouped: dict[str, dict[str, list[FileEntry]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        grouped[entry.dataset][entry.source].append(entry)

    for dataset, source_entries in grouped.items():
        for source, files in source_entries.items():
            has_existing_split = any(f.existing_split is not None for f in files)

            if has_existing_split:
                for file in files:
                    split = file.existing_split or "test"
                    index[dataset][split][source].append(file.rel_path)
                continue

            video_to_files: dict[str, list[FileEntry]] = defaultdict(list)
            for file in files:
                video_to_files[file.video].append(file)

            split_to_videos = split_videos(list(video_to_files.keys()), ratios=ratios, seed=seed)

            for split, videos in split_to_videos.items():
                for video in videos:
                    for file in video_to_files[video]:
                        index[dataset][split][source].append(file.rel_path)

    # Keep deterministic order.
    for dataset in index:
        for split in index[dataset]:
            for source in index[dataset][split]:
                index[dataset][split][source] = sorted(set(index[dataset][split][source]))

    return index


def write_filelists(index: dict[str, dict[str, dict[str, list[str]]]], output_root: Path, overwrite: bool) -> list[Path]:
    written_files: list[Path] = []

    for dataset, split_map in sorted(index.items()):
        for split, source_map in sorted(split_map.items()):
            for source, paths in sorted(source_map.items()):
                if not paths:
                    continue

                output_file = output_root / dataset / split / f"{source}.txt"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                if output_file.exists() and not overwrite:
                    raise FileExistsError(
                        f"Target file already exists: {output_file}. Use --overwrite to replace it."
                    )

                output_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
                written_files.append(output_file)

    return written_files


def build_summary(index: dict[str, dict[str, dict[str, list[str]]]]) -> list[str]:
    lines = []
    for dataset, split_map in sorted(index.items()):
        lines.append(f"[{dataset}]")
        for split in SPLITS:
            if split not in split_map:
                continue
            source_map = split_map[split]
            total_files = sum(len(paths) for paths in source_map.values())
            lines.append(f"  {split}: files={total_files}, sources={len(source_map)}")
        lines.append("")
    return lines


def main() -> int:
    args = parse_args()

    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)

    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets-root does not exist: {datasets_root}")

    dataset_filter = {x.strip() for x in args.datasets.split(",") if x.strip()}
    ratios = parse_ratios(args.ratios)

    entries = scan_entries(
        datasets_root=datasets_root,
        dataset_filter=dataset_filter,
        respect_existing_splits=args.respect_existing_splits,
        source_name_rule=args.source_name_rule,
    )

    if not entries:
        print("No image files found. Nothing to do.")
        return 0

    index = build_output_index(entries=entries, ratios=ratios, seed=args.seed)
    written_files = write_filelists(index=index, output_root=output_root, overwrite=args.overwrite)

    print(f"Scanned files: {len(entries)}")
    print(f"Generated txt files: {len(written_files)}")
    for line in build_summary(index):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
