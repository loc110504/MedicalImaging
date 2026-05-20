import argparse
import csv
import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F

from networks.net_factory import net_factory

np.bool = np.bool_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../../data/ACDC")
    parser.add_argument("--model", type=str, default="unet_hl")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../checkpoints/ACDC_EvidentialFullMask/unet_hl_best_model.pth",
    )
    parser.add_argument("--num_slices", type=int, default=5, help="random slices to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="../../code/test/uncertainty_vis_evidential")
    parser.add_argument("--num_clicks", type=int, default=5)
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        default="evidential",
        choices=["evidential", "entropy", "maxprob"],
    )
    parser.add_argument(
        "--click_selection",
        type=str,
        default="topk_nms",
        choices=["topk", "topk_nms", "error_aware_oracle"],
    )
    parser.add_argument("--min_click_distance", type=int, default=8)
    parser.add_argument(
        "--click_polarity",
        type=str,
        default="gt",
        choices=["none", "gt", "pred", "error_type"],
    )
    parser.add_argument("--foreground_classes", type=int, nargs="+", default=None)
    parser.add_argument("--save_click_json", action="store_true")
    parser.add_argument("--save_click_csv", action="store_true")
    parser.add_argument("--show_click_index", action="store_true")
    return parser.parse_args()


def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize01(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=np.float32)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + eps)


def get_selected_uncertainty_map(uncertainty_type, maxprob_unc, entropy_unc, evidential_unc):
    mapping = {
        "maxprob": maxprob_unc,
        "entropy": entropy_unc,
        "evidential": evidential_unc,
    }
    if uncertainty_type not in mapping:
        raise ValueError("Unknown uncertainty_type: {}".format(uncertainty_type))
    selected = np.asarray(mapping[uncertainty_type], dtype=np.float32)
    if selected.ndim != 2:
        raise ValueError("Uncertainty map must be 2D [H, W], got shape {}".format(selected.shape))
    return normalize01(selected).astype(np.float32)


def make_foreground_mask(mask, foreground_classes):
    mask = np.asarray(mask)
    if foreground_classes is None or len(foreground_classes) == 0:
        return mask > 0
    return np.isin(mask, np.asarray(foreground_classes))


def _iter_sorted_candidates(uncertainty, exclude_mask=None):
    h, w = uncertainty.shape
    flat = uncertainty.reshape(-1)
    order = np.argsort(flat)[::-1]
    for idx in order:
        y, x = divmod(int(idx), w)
        if exclude_mask is not None and exclude_mask[y, x]:
            continue
        yield x, y, float(uncertainty[y, x])


def select_topk_uncertain_points(uncertainty, num_clicks, exclude_mask=None):
    clicks = []
    seen = set()
    for x, y, value in _iter_sorted_candidates(uncertainty, exclude_mask=exclude_mask):
        if (x, y) in seen:
            continue
        clicks.append({"x": int(x), "y": int(y), "uncertainty": float(value)})
        seen.add((x, y))
        if len(clicks) >= num_clicks:
            break
    if len(clicks) < num_clicks:
        print("warning: requested {} clicks but found {}".format(num_clicks, len(clicks)))
    return clicks


def _is_far_enough(x, y, clicks, min_distance):
    for c in clicks:
        dist = np.sqrt((x - c["x"]) ** 2 + (y - c["y"]) ** 2)
        if dist < min_distance:
            return False
    return True


def select_topk_uncertain_points_nms(uncertainty, num_clicks, min_distance=8, exclude_mask=None):
    clicks = []
    for x, y, value in _iter_sorted_candidates(uncertainty, exclude_mask=exclude_mask):
        if _is_far_enough(x, y, clicks, min_distance):
            clicks.append({"x": int(x), "y": int(y), "uncertainty": float(value)})
        if len(clicks) >= num_clicks:
            break
    if len(clicks) < num_clicks:
        print("warning: requested {} clicks but found {} with min_distance={}".format(
            num_clicks, len(clicks), min_distance
        ))
    return clicks


def assign_click_polarity(clicks, gt_mask, pred_mask, foreground_classes, mode="gt"):
    gt_fg = make_foreground_mask(gt_mask, foreground_classes)
    pred_fg = make_foreground_mask(pred_mask, foreground_classes)
    fn = gt_fg & (~pred_fg)
    fp = (~gt_fg) & pred_fg

    labeled = []
    for c in clicks:
        x, y = c["x"], c["y"]
        item = dict(c)
        if mode == "none":
            label = None
            label_name = "unknown"
        elif mode == "gt":
            label = 1 if gt_fg[y, x] else 0
            label_name = "positive" if label == 1 else "negative"
        elif mode == "pred":
            label = 1 if pred_fg[y, x] else 0
            label_name = "positive" if label == 1 else "negative"
        elif mode == "error_type":
            if fn[y, x]:
                label = 1
            elif fp[y, x]:
                label = 0
            else:
                label = 1 if gt_fg[y, x] else 0
            label_name = "positive" if label == 1 else "negative"
        else:
            raise ValueError("Unknown click_polarity mode: {}".format(mode))
        item["label"] = label
        item["label_name"] = label_name
        labeled.append(item)
    return labeled


def select_error_aware_oracle_points(
    uncertainty, gt_mask, pred_mask, num_clicks, foreground_classes, min_distance=8
):
    gt_fg = make_foreground_mask(gt_mask, foreground_classes)
    pred_fg = make_foreground_mask(pred_mask, foreground_classes)
    fn = gt_fg & (~pred_fg)
    fp = (~gt_fg) & pred_fg
    score_pos = uncertainty * fn.astype(np.float32)
    score_neg = uncertainty * fp.astype(np.float32)
    clicks = []

    for _ in range(num_clicks):
        best = None
        for src_name, score_map, label in (("fn", score_pos, 1), ("fp", score_neg, 0)):
            for x, y, value in _iter_sorted_candidates(score_map):
                if value <= 0:
                    break
                if _is_far_enough(x, y, clicks, min_distance):
                    best = (x, y, float(uncertainty[y, x]), label)
                    break
            if best is not None:
                break
        if best is None:
            break
        x, y, unc_value, label = best
        clicks.append(
            {
                "x": int(x),
                "y": int(y),
                "uncertainty": float(unc_value),
                "label": label,
                "label_name": "positive" if label == 1 else "negative",
            }
        )
        score_pos[y, x] = 0.0
        score_neg[y, x] = 0.0

    if len(clicks) < num_clicks:
        fallback = select_topk_uncertain_points_nms(
            uncertainty, num_clicks * 2, min_distance=min_distance
        )
        used = {(c["x"], c["y"]) for c in clicks}
        extras = [c for c in fallback if (c["x"], c["y"]) not in used]
        extras = assign_click_polarity(
            extras, gt_mask=gt_mask, pred_mask=pred_mask, foreground_classes=foreground_classes, mode="gt"
        )
        need = num_clicks - len(clicks)
        clicks.extend(extras[:need])

    if len(clicks) < num_clicks:
        print("warning: requested {} clicks but found {} in error_aware_oracle".format(
            num_clicks, len(clicks)
        ))
    return clicks


def list_cases(root_path):
    test_txt = os.path.join(root_path, "test.txt")
    with open(test_txt, "r") as f:
        cases = [line.strip() for line in f if line.strip()]
    return [os.path.splitext(case)[0] for case in cases]


def load_case(root_path, case_name):
    h5_path = os.path.join(root_path, "ACDC_training_volumes", case_name + ".h5")
    with h5py.File(h5_path, "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]
    return image, label


def sample_random_slices(root_path, num_slices, rng):
    sampled = []
    cases = list_cases(root_path)
    case_order = cases[:]
    rng.shuffle(case_order)

    for case_name in case_order:
        image_3d, label_3d = load_case(root_path, case_name)
        fg_indices = np.where(label_3d.reshape(label_3d.shape[0], -1).sum(axis=1) > 0)[0]
        candidates = fg_indices.tolist() if len(fg_indices) > 0 else list(range(label_3d.shape[0]))
        if not candidates:
            continue
        slice_idx = rng.choice(candidates)
        sampled.append((case_name, int(slice_idx), image_3d, label_3d))
        if len(sampled) >= num_slices:
            break

    return sampled


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "net", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format: {}".format(checkpoint_path))


def load_model_weights(net, state_dict):
    try:
        net.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    net.load_state_dict(cleaned_state_dict)


def forward_logits(net, x):
    out = net(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def preprocess_slice(image_3d, label_3d, slice_idx, device):
    image_2d = image_3d[slice_idx]
    label_2d = label_3d[slice_idx]
    h, w = image_2d.shape
    image_256 = zoom(image_2d, (256 / h, 256 / w), order=0)
    label_256 = zoom(label_2d, (256 / h, 256 / w), order=0)
    input_tensor = torch.from_numpy(image_256).unsqueeze(0).unsqueeze(0).float().to(device)
    return image_256, label_256, input_tensor


def predict_and_uncertainty(net, x, num_classes):
    logits = forward_logits(net, x)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1)

    maxprob_unc = 1.0 - prob.max(dim=1).values
    entropy_unc = -(prob * torch.log(prob + 1e-8)).sum(dim=1) / np.log(num_classes)

    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    evidential_unc = num_classes / (alpha.sum(dim=1) + 1e-8)

    return (
        pred.squeeze(0).cpu().numpy().astype(np.int32),
        normalize01(maxprob_unc.squeeze(0).cpu().numpy()),
        normalize01(entropy_unc.squeeze(0).cpu().numpy()),
        normalize01(evidential_unc.squeeze(0).cpu().numpy()),
    )


def save_figure(case_name, slice_idx, image_256, label_256, pred_256, maxprob_unc, entropy_unc, evidential_unc, output_dir):
    seg_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0),
            (0.88, 0.24, 0.22),
            (0.23, 0.69, 0.29),
            (0.18, 0.45, 0.86),
        ]
    )

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    panels = [
        ("Image", normalize01(image_256), "gray", None),
        ("GT", label_256, seg_cmap, (0, 3)),
        ("Pred", pred_256, seg_cmap, (0, 3)),
        ("MaxProb Unc", maxprob_unc, "jet", (0, 1)),
        ("Entropy Unc", entropy_unc, "jet", (0, 1)),
        ("Evidential Unc", evidential_unc, "jet", (0, 1)),
    ]

    for ax, (title, data, cmap, limits) in zip(axes, panels):
        if limits is None:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("{} slice {}".format(case_name, slice_idx), fontsize=12)
    save_path = os.path.join(output_dir, "{}_slice{}_pred_uncertainty.png".format(case_name, slice_idx))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _draw_clicks(ax, clicks, show_click_index=True):
    for idx, click in enumerate(clicks, start=1):
        x, y = click["x"], click["y"]
        label = click.get("label", None)
        if label == 1:
            ax.scatter(x, y, c="lime", marker="o", s=70, edgecolors="black", linewidths=1.0)
        elif label == 0:
            ax.scatter(x, y, c="red", marker="x", s=80, linewidths=2.0)
        else:
            ax.scatter(x, y, c="yellow", marker="*", s=90, edgecolors="black", linewidths=1.0)
        if show_click_index:
            ax.text(
                x + 3,
                y + 3,
                str(idx),
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )


def save_click_figure(
    case_name,
    slice_idx,
    image_256,
    label_256,
    pred_256,
    uncertainty,
    clicks,
    uncertainty_type,
    click_polarity,
    output_dir,
    num_classes,
    num_clicks,
    show_click_index=True,
):
    seg_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0),
            (0.88, 0.24, 0.22),
            (0.23, 0.69, 0.29),
            (0.18, 0.45, 0.86),
        ]
    )

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    pred_fg = pred_256 > 0
    panels = [
        ("Image + Clicks", normalize01(image_256), "gray", None),
        ("GT + Clicks", label_256, seg_cmap, (0, num_classes - 1)),
        ("Pred + Clicks", pred_256, seg_cmap, (0, num_classes - 1)),
        ("{} Unc + Clicks".format(uncertainty_type.capitalize()), uncertainty, "jet", (0, 1)),
    ]

    for ax, (title, data, cmap, limits) in zip(axes[:4], panels):
        if limits is None:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
        _draw_clicks(ax, clicks, show_click_index=show_click_index)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    axes[4].imshow(normalize01(image_256), cmap="gray")
    axes[4].imshow(pred_fg.astype(np.float32), cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    _draw_clicks(axes[4], clicks, show_click_index=show_click_index)
    axes[4].set_title("Image + Pred Overlay + Clicks", fontsize=10)
    axes[4].axis("off")

    fig.suptitle(
        "{} slice {} | {} EUGIS-style clicks | uncertainty={} | polarity={}".format(
            case_name, slice_idx, num_clicks, uncertainty_type, click_polarity
        ),
        fontsize=12,
    )
    save_path = os.path.join(
        output_dir,
        "{}_slice{}_eugis_clicks_{}_{}.png".format(case_name, slice_idx, uncertainty_type, num_clicks),
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_click_metadata(
    case_name, slice_idx, clicks, uncertainty_type, click_selection, output_dir, num_clicks, save_json=False, save_csv=False
):
    records = []
    for idx, click in enumerate(clicks, start=1):
        label = click.get("label", None)
        label_name = click.get("label_name", "unknown")
        records.append(
            {
                "case_name": case_name,
                "slice_idx": int(slice_idx),
                "click_id": idx,
                "x": int(click["x"]),
                "y": int(click["y"]),
                "uncertainty": float(click["uncertainty"]),
                "label": None if label is None else int(label),
                "label_name": label_name,
                "source": uncertainty_type,
                "selection_method": click_selection,
            }
        )

    base = os.path.join(
        output_dir,
        "{}_slice{}_eugis_clicks_{}_{}".format(case_name, slice_idx, uncertainty_type, num_clicks),
    )
    json_path, csv_path = None, None
    if save_json:
        json_path = base + ".json"
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
    if save_csv:
        csv_path = base + ".csv"
        fields = [
            "case_name",
            "slice_idx",
            "click_id",
            "x",
            "y",
            "uncertainty",
            "label",
            "label_name",
            "source",
            "selection_method",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return json_path, csv_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.foreground_classes is None:
        args.foreground_classes = list(range(1, args.num_classes))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = get_device(args.device)

    net = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes).to(device)
    state_dict = load_checkpoint_state(args.checkpoint)
    load_model_weights(net, state_dict)
    net.eval()

    rng = random.Random(args.seed)
    sampled = sample_random_slices(args.root_path, args.num_slices, rng)

    print("Using device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Sampled {} slices".format(len(sampled)))

    with torch.no_grad():
        for case_name, slice_idx, image_3d, label_3d in sampled:
            image_256, label_256, x = preprocess_slice(image_3d, label_3d, slice_idx, device)
            pred_256, maxprob_unc, entropy_unc, evidential_unc = predict_and_uncertainty(
                net, x, args.num_classes
            )
            selected_unc = get_selected_uncertainty_map(
                args.uncertainty_type, maxprob_unc, entropy_unc, evidential_unc
            )
            out_path = save_figure(
                case_name,
                slice_idx,
                image_256,
                label_256,
                pred_256,
                maxprob_unc,
                entropy_unc,
                evidential_unc,
                args.output_dir,
            )
            print("saved:", out_path)

            if args.click_selection == "topk":
                clicks = select_topk_uncertain_points(selected_unc, args.num_clicks)
            elif args.click_selection == "topk_nms":
                clicks = select_topk_uncertain_points_nms(
                    selected_unc, args.num_clicks, min_distance=args.min_click_distance
                )
            elif args.click_selection == "error_aware_oracle":
                clicks = select_error_aware_oracle_points(
                    selected_unc,
                    label_256,
                    pred_256,
                    args.num_clicks,
                    foreground_classes=args.foreground_classes,
                    min_distance=args.min_click_distance,
                )
            else:
                raise ValueError("Unknown click_selection: {}".format(args.click_selection))

            if args.click_selection != "error_aware_oracle":
                clicks = assign_click_polarity(
                    clicks,
                    gt_mask=label_256,
                    pred_mask=pred_256,
                    foreground_classes=args.foreground_classes,
                    mode=args.click_polarity,
                )

            click_fig_path = save_click_figure(
                case_name,
                slice_idx,
                image_256,
                label_256,
                pred_256,
                selected_unc,
                clicks,
                args.uncertainty_type,
                args.click_polarity,
                args.output_dir,
                num_classes=args.num_classes,
                num_clicks=args.num_clicks,
                show_click_index=args.show_click_index,
            )
            print("saved clicks:", click_fig_path)

            json_path, csv_path = save_click_metadata(
                case_name,
                slice_idx,
                clicks,
                args.uncertainty_type,
                args.click_selection,
                args.output_dir,
                args.num_clicks,
                save_json=args.save_click_json,
                save_csv=args.save_click_csv,
            )
            if json_path:
                print("saved click json:", json_path)
            if csv_path:
                print("saved click csv:", csv_path)


if __name__ == "__main__":
    main()
