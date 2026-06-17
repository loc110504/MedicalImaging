import argparse
import os
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def get_acdc_fold_ids(fold):
    all_cases = ["patient{:0>3}".format(i) for i in range(1, 101)]
    predefined = {
        "fold1": ["patient{:0>3}".format(i) for i in range(1, 21)],
        "fold2": ["patient{:0>3}".format(i) for i in range(21, 41)],
        "fold3": ["patient{:0>3}".format(i) for i in range(41, 61)],
        "fold4": ["patient{:0>3}".format(i) for i in range(61, 81)],
        "fold5": ["patient{:0>3}".format(i) for i in range(81, 101)],
        "MAAGfold": ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]],
        "MAAGfold70": ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]],
    }

    if fold not in predefined:
        raise ValueError("Unsupported ACDC fold: {}".format(fold))

    test_ids = predefined[fold]
    train_ids = [case_id for case_id in all_cases if case_id not in test_ids]
    return train_ids, test_ids


def get_acdc_split_files(base_dir, split, fold):
    train_ids, val_ids = get_acdc_fold_ids(fold)
    folder = Path(base_dir) / ("ACDC_training_slices" if split == "train" else "ACDC_training_volumes")
    selected_ids = train_ids if split == "train" else val_ids
    files = []
    for name in sorted(os.listdir(folder)):
        if any(re.match(r"{}.*".format(case_id), name) for case_id in selected_ids):
            files.append(folder / name)
    return files


def get_mscmr_split_files(base_dir, split):
    folder_name = "MSCMR_training_slices" if split == "train" else "MSCMR_validation_volumes"
    folder = Path(base_dir) / folder_name
    return [folder / name for name in sorted(os.listdir(folder))]


def iter_slices(array):
    if array.ndim == 2:
        yield array
    elif array.ndim == 3:
        for idx in range(array.shape[0]):
            yield array[idx]
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))


def detect_class_ids_and_ignore(labels, scribbles):
    label_values = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    scribble_values = set(int(v) for v in np.unique(scribbles)) if scribbles is not None else set()
    ignore_values = sorted(v for v in scribble_values if v not in label_values)
    return label_values, ignore_values


def init_stats():
    return {
        "num_files": 0,
        "num_slices": 0,
        "gt_pixel_counts": Counter(),
        "gt_slice_presence": Counter(),
        "empty_foreground_slices": 0,
        "empty_foreground_files": 0,
        "scribble_available_files": 0,
        "scribble_pixel_counts": Counter(),
        "scribble_slice_presence": Counter(),
        "annotated_scribble_pixels": 0,
        "annotated_scribble_slices": 0,
        "empty_scribble_slices": 0,
        "class_ids": set(),
        "scribble_ignore_values": Counter(),
        "files_without_scribble_key": 0,
    }


def analyze_split(files):
    stats = init_stats()

    for file_path in files:
        stats["num_files"] += 1
        with h5py.File(file_path, "r") as handle:
            label = handle["label"][:]
            scribble = handle["scribble"][:] if "scribble" in handle else None

        class_ids, ignore_values = detect_class_ids_and_ignore(label, scribble)
        stats["class_ids"].update(class_ids)
        for value in ignore_values:
            stats["scribble_ignore_values"][value] += 1

        for class_id in class_ids:
            stats["gt_pixel_counts"][class_id] += int(np.sum(label == class_id))

        file_has_foreground = False
        label_slices = list(iter_slices(label))
        scribble_slices = list(iter_slices(scribble)) if scribble is not None else [None] * len(label_slices)

        if scribble is None:
            stats["files_without_scribble_key"] += 1
        else:
            stats["scribble_available_files"] += 1

        for label_slice, scribble_slice in zip(label_slices, scribble_slices):
            stats["num_slices"] += 1
            foreground_present = False
            for class_id in class_ids:
                if class_id == 0:
                    continue
                present = bool(np.any(label_slice == class_id))
                if present:
                    stats["gt_slice_presence"][class_id] += 1
                    foreground_present = True
                    file_has_foreground = True

            if not foreground_present:
                stats["empty_foreground_slices"] += 1

            if scribble_slice is None:
                continue

            valid_scribble_mask = np.isin(scribble_slice, class_ids)
            annotated_pixels = int(valid_scribble_mask.sum())
            stats["annotated_scribble_pixels"] += annotated_pixels

            if annotated_pixels > 0:
                stats["annotated_scribble_slices"] += 1
            else:
                stats["empty_scribble_slices"] += 1

            for class_id in class_ids:
                count = int(np.sum(scribble_slice == class_id))
                stats["scribble_pixel_counts"][class_id] += count
                if count > 0:
                    stats["scribble_slice_presence"][class_id] += 1

        if not file_has_foreground:
            stats["empty_foreground_files"] += 1

    stats["class_ids"] = sorted(stats["class_ids"])
    return stats


def format_ratio(numerator, denominator):
    if denominator == 0:
        return "0.00%"
    return "{:.2f}%".format(100.0 * float(numerator) / float(denominator))


def build_distribution_lines(counter, class_ids, total, title):
    lines = [title]
    nonzero_counts = []
    for class_id in class_ids:
        count = int(counter[class_id])
        if count > 0:
            nonzero_counts.append(count)
        lines.append(
            "  class {}: {:>10} pixels ({})".format(
                class_id, count, format_ratio(count, total)
            )
        )

    if nonzero_counts:
        imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)
        lines.append("  imbalance ratio (max/min non-zero): {:.2f}".format(imbalance_ratio))
    return lines


def build_presence_lines(counter, class_ids, total_slices, title):
    lines = [title]
    for class_id in class_ids:
        if class_id == 0:
            continue
        present_slices = int(counter[class_id])
        lines.append(
            "  class {}: present in {:>6} slices ({})".format(
                class_id, present_slices, format_ratio(present_slices, total_slices)
            )
        )
    return lines


def print_split_report(dataset_name, split_name, stats):
    class_ids = stats["class_ids"]
    total_gt_pixels = sum(int(stats["gt_pixel_counts"][class_id]) for class_id in class_ids)
    total_fg_pixels = sum(int(stats["gt_pixel_counts"][class_id]) for class_id in class_ids if class_id != 0)

    print("=" * 80)
    print("{} - {}".format(dataset_name, split_name))
    print("files: {}".format(stats["num_files"]))
    print("2D slices/images analyzed: {}".format(stats["num_slices"]))
    print("class ids in GT label: {}".format(class_ids))
    print("empty foreground files: {} / {}".format(stats["empty_foreground_files"], stats["num_files"]))
    print(
        "empty foreground slices: {} / {} ({})".format(
            stats["empty_foreground_slices"],
            stats["num_slices"],
            format_ratio(stats["empty_foreground_slices"], stats["num_slices"]),
        )
    )
    print()

    for line in build_distribution_lines(stats["gt_pixel_counts"], class_ids, total_gt_pixels, "GT pixel distribution"):
        print(line)
    print()

    if total_fg_pixels > 0:
        fg_counter = Counter({class_id: stats["gt_pixel_counts"][class_id] for class_id in class_ids if class_id != 0})
        for line in build_distribution_lines(fg_counter, [class_id for class_id in class_ids if class_id != 0], total_fg_pixels, "Foreground-only GT distribution"):
            print(line)
        print()

    for line in build_presence_lines(stats["gt_slice_presence"], class_ids, stats["num_slices"], "Foreground class presence by slice"):
        print(line)
    print()

    if stats["scribble_available_files"] > 0:
        total_scribble_pixels = sum(int(stats["scribble_pixel_counts"][class_id]) for class_id in class_ids)
        print("scribble available in files: {} / {}".format(stats["scribble_available_files"], stats["num_files"]))
        print("files without scribble key: {}".format(stats["files_without_scribble_key"]))
        print("scribble ignore values seen: {}".format(dict(stats["scribble_ignore_values"])))
        print(
            "annotated scribble slices: {} / {} ({})".format(
                stats["annotated_scribble_slices"],
                stats["num_slices"],
                format_ratio(stats["annotated_scribble_slices"], stats["num_slices"]),
            )
        )
        print(
            "empty scribble slices: {} / {} ({})".format(
                stats["empty_scribble_slices"],
                stats["num_slices"],
                format_ratio(stats["empty_scribble_slices"], stats["num_slices"]),
            )
        )
        print(
            "annotated scribble pixels: {} / {} ({})".format(
                stats["annotated_scribble_pixels"],
                total_gt_pixels,
                format_ratio(stats["annotated_scribble_pixels"], total_gt_pixels),
            )
        )
        print()

        for line in build_distribution_lines(stats["scribble_pixel_counts"], class_ids, total_scribble_pixels, "Scribble pixel distribution"):
            print(line)
        print()

        for line in build_presence_lines(stats["scribble_slice_presence"], class_ids, stats["num_slices"], "Scribble class presence by slice"):
            print(line)
    else:
        print("No scribble key found in this split.")

    print()


def analyze_dataset(dataset_name, root_path, fold):
    if dataset_name == "ACDC":
        split_files = {
            "train": get_acdc_split_files(root_path, "train", fold),
            "val": get_acdc_split_files(root_path, "val", fold),
        }
    elif dataset_name == "MSCMR":
        split_files = {
            "train": get_mscmr_split_files(root_path, "train"),
            "val": get_mscmr_split_files(root_path, "val"),
        }
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    for split_name, files in split_files.items():
        stats = analyze_split(files)
        print_split_report(dataset_name, split_name, stats)


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Analyze class balance and scribble coverage for ACDC/MSCMR.")
    parser.add_argument(
        "--dataset",
        choices=["ACDC", "MSCMR", "both"],
        default="both",
        help="dataset to analyze",
    )
    parser.add_argument(
        "--acdc_root",
        type=str,
        default=str(repo_root / "data" / "ACDC"),
        help="path to ACDC root",
    )
    parser.add_argument(
        "--mscmr_root",
        type=str,
        default=str(repo_root / "data" / "MSCMR"),
        help="path to MSCMR root",
    )
    parser.add_argument(
        "--acdc_fold",
        type=str,
        default="MAAGfold70",
        help="ACDC fold used to define train/val split",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected = ["ACDC", "MSCMR"] if args.dataset == "both" else [args.dataset]
    roots = {
        "ACDC": args.acdc_root,
        "MSCMR": args.mscmr_root,
    }

    for dataset_name in selected:
        analyze_dataset(dataset_name, roots[dataset_name], args.acdc_fold)


if __name__ == "__main__":
    main()
