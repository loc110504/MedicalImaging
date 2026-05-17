import argparse
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
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label as cc_label, zoom
import torch
import torch.nn.functional as F

from networks.net_factory import net_factory

np.bool = np.bool_


try:
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    HAS_RANDOM_WALKER = True
except Exception:
    HAS_RANDOM_WALKER = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../../data/ACDC")
    parser.add_argument("--model", type=str, default="unet_hl")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--slices_per_case", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./compact_scribble_vis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_classes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--unc_percentile", type=float, default=90)
    parser.add_argument("--band_radius", type=int, default=5)
    parser.add_argument("--safety_margin", type=int, default=4)
    parser.add_argument("--max_components", type=int, default=2)
    parser.add_argument("--max_scribble_len", type=int, default=25)
    parser.add_argument("--scribble_width", type=int, default=1)
    parser.add_argument("--min_component_size", type=int, default=6)
    parser.add_argument("--run_random_walker", action="store_true")
    parser.add_argument("--rw_beta", type=float, default=100)
    parser.add_argument("--rw_mode", type=str, default="bf")
    return parser.parse_args()


def normalize01(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    return (arr - mn) / (mx - mn + eps)


def load_checkpoint_flexible(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "net", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unsupported checkpoint format: {}".format(ckpt_path))


def load_state_flexible(net, state_dict):
    try:
        net.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    net.load_state_dict(cleaned)


def forward_logits(net, x, model_name):
    out = net(x)
    if model_name == "unet_hl":
        return out[0] if isinstance(out, (tuple, list)) else out
    return out[0] if isinstance(out, (tuple, list)) else out


def compute_entropy_uncertainty(logits, num_classes):
    prob = torch.softmax(logits, dim=1)
    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
    entropy = entropy / np.log(num_classes)
    return entropy.squeeze(0).detach().cpu().numpy().astype(np.float32)


def compute_evidential_style_uncertainty(logits, num_classes):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    s = alpha.sum(dim=1)
    unc = num_classes / (s + 1e-8)
    return unc.squeeze(0).detach().cpu().numpy().astype(np.float32)


def make_boundary_band(mask, band_radius=5):
    mask = mask.astype(bool)
    dil = binary_dilation(mask, iterations=max(1, band_radius))
    ero = binary_erosion(mask, iterations=max(1, band_radius))
    return np.logical_xor(dil, ero)


def extract_compact_scribble(candidate, unc, max_components=2, max_scribble_len=25, scribble_width=1, min_component_size=6):
    candidate = candidate.astype(bool)
    scribble = np.zeros_like(candidate, dtype=bool)
    labeled, n_comp = cc_label(candidate)

    comp_infos = []
    for cid in range(1, n_comp + 1):
        comp = labeled == cid
        size = int(comp.sum())
        if size < min_component_size:
            continue
        unc_vals = unc[comp]
        score = 0.7 * float(unc_vals.max()) + 0.3 * float(unc_vals.mean())
        comp_infos.append((cid, score, size))

    comp_infos.sort(key=lambda x: x[1], reverse=True)
    selected_infos = comp_infos[: max(0, max_components)]
    selected_ids = [c[0] for c in selected_infos]

    for cid in selected_ids:
        comp = labeled == cid
        coords = np.argwhere(comp)
        if coords.shape[0] == 0:
            continue
        unc_vals = unc[comp]
        peak_idx = int(np.argmax(unc_vals))
        peak = coords[peak_idx]
        dist = np.sqrt(((coords - peak[None, :]) ** 2).sum(axis=1))
        keep = coords[np.argsort(dist)[: max(1, max_scribble_len)]]
        scribble[keep[:, 0], keep[:, 1]] = True

    if scribble_width > 1 and scribble.any():
        scribble = binary_dilation(scribble, iterations=int(scribble_width - 1))

    debug = {
        "candidate": candidate,
        "selected_components": selected_ids,
        "component_scores": [(int(cid), float(score), int(size)) for cid, score, size in selected_infos],
    }
    return scribble, debug


def generate_compact_signed_scribbles(pred, unc, class_id, band_radius=5, safety_margin=4, unc_percentile=90, max_components=2, max_scribble_len=25, scribble_width=1, min_component_size=6):
    mask = pred == class_id
    band = make_boundary_band(mask, band_radius=band_radius)
    dist_in = distance_transform_edt(mask)
    dist_out = distance_transform_edt(~mask)
    thr = np.percentile(unc, unc_percentile)
    high_unc = unc >= thr

    pos_candidate = high_unc & band & mask & (dist_in >= safety_margin)
    neg_candidate = high_unc & band & (~mask) & (dist_out >= safety_margin)

    pos_scribble, pos_debug = extract_compact_scribble(
        pos_candidate,
        unc,
        max_components=max_components,
        max_scribble_len=max_scribble_len,
        scribble_width=scribble_width,
        min_component_size=min_component_size,
    )
    neg_scribble, neg_debug = extract_compact_scribble(
        neg_candidate,
        unc,
        max_components=max_components,
        max_scribble_len=max_scribble_len,
        scribble_width=scribble_width,
        min_component_size=min_component_size,
    )

    debug = {
        "band": band,
        "high_unc": high_unc,
        "threshold": float(thr),
        "pos": pos_debug,
        "neg": neg_debug,
    }
    return pos_scribble, neg_scribble, debug


def build_random_walker_markers(pos_scribble, neg_scribble):
    markers = np.zeros(pos_scribble.shape, dtype=np.int32)
    markers[neg_scribble] = 1
    markers[pos_scribble] = 2
    return markers


def random_walker_preview(image_2d, pos_scribble, neg_scribble, beta=100, mode="bf"):
    markers = build_random_walker_markers(pos_scribble, neg_scribble)
    u = np.unique(markers)
    if 1 not in u or 2 not in u:
        return np.zeros_like(markers, dtype=bool), markers, False
    if not HAS_RANDOM_WALKER:
        print("warning: skimage random_walker unavailable; skip preview")
        return np.zeros_like(markers, dtype=bool), markers, False
    try:
        img = normalize01(image_2d)
        data_rw = rescale_intensity(img, in_range=(0, 1), out_range=(-1, 1))
        seg = random_walker(data_rw, markers, beta=beta, mode=mode)
        return seg == 2, markers, True
    except Exception as e:
        print("warning: random walker failed: {}".format(e))
        return np.zeros_like(markers, dtype=bool), markers, False


def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_test_cases(root_path):
    with open(os.path.join(root_path, "test.txt"), "r") as f:
        cases = [line.strip() for line in f if line.strip()]
    return [os.path.splitext(c)[0] for c in cases]


def load_case(root_path, case_name):
    h5_path = os.path.join(root_path, "ACDC_training_volumes", case_name + ".h5")
    with h5py.File(h5_path, "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]
    return image, label


def select_cases(case_list, num_cases, rng):
    num_cases = min(num_cases, len(case_list))
    return rng.sample(case_list, num_cases)


def select_slice_indices(label_3d, slices_per_case, rng):
    fg = np.where(label_3d.reshape(label_3d.shape[0], -1).sum(axis=1) > 0)[0]
    cands = fg.tolist() if len(fg) > 0 else list(range(label_3d.shape[0]))
    if slices_per_case >= len(cands):
        return cands
    return sorted(rng.sample(cands, slices_per_case))


def preprocess_slice(image_3d, label_3d, slice_idx, device):
    image_2d = image_3d[slice_idx]
    gt_2d = label_3d[slice_idx]
    h, w = image_2d.shape
    image_256 = zoom(image_2d, (256 / h, 256 / w), order=0)
    gt_256 = zoom(gt_2d, (256 / h, 256 / w), order=0)
    x = torch.from_numpy(image_256).unsqueeze(0).unsqueeze(0).float().to(device)
    return image_256, gt_256, x


def get_seg_cmap():
    return ListedColormap([
        (0.0, 0.0, 0.0),
        (0.88, 0.24, 0.22),
        (0.23, 0.69, 0.29),
        (0.18, 0.45, 0.86),
    ])


def _plot_scribble_overlay(ax, image_norm, gt, pred, class_id, pos, neg, title):
    ax.imshow(image_norm, cmap="gray")
    ax.contour(pred == class_id, colors="yellow", linewidths=0.8)
    ax.contour(gt == class_id, colors="cyan", linewidths=0.8)
    py, px = np.where(pos)
    ny, nx = np.where(neg)
    if len(px) > 0:
        ax.scatter(px, py, c="lime", s=4)
    if len(nx) > 0:
        ax.scatter(nx, ny, c="red", s=4)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def visualize_compact_overview(case_name, slice_idx, image_2d, gt, pred, entropy_unc, evidential_unc, consensus_unc, scribbles, output_dir):
    image_norm = normalize01(image_2d)
    seg_cmap = get_seg_cmap()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    panels = [
        ("Image", image_norm, "gray", None),
        ("Ground Truth", gt, seg_cmap, (0, 3)),
        ("Prediction", pred, seg_cmap, (0, 3)),
        ("Entropy Uncertainty", entropy_unc, "jet", (0, 1)),
        ("Evidential-style Approx.", evidential_unc, "jet", (0, 1)),
        ("Consensus Uncertainty", consensus_unc, "jet", (0, 1)),
    ]

    for i, (title, data, cmap, lim) in enumerate(panels):
        if lim is None:
            axes[i].imshow(data, cmap=cmap)
        else:
            axes[i].imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1])
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    for i, (key, label) in enumerate([
        ("v1_entropy", "V1 Entropy Compact Scribble"),
        ("v2_evi", "V2 Evidential Compact Scribble"),
        ("v3_consensus", "V3 Consensus Compact Scribble"),
    ]):
        pos = np.zeros_like(pred, dtype=bool)
        neg = np.zeros_like(pred, dtype=bool)
        for c_data in scribbles[key].values():
            pos |= c_data["pos"]
            neg |= c_data["neg"]
        _plot_scribble_overlay(
            axes[6 + i], image_norm, gt, pred, class_id=1, pos=pos, neg=neg,
            title="{}\npos={} neg={}".format(label, int(pos.sum()), int(neg.sum())),
        )

    fig.suptitle("{} slice {}".format(case_name, slice_idx), fontsize=12)
    save_dir = os.path.join(output_dir, "overview")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}_slice{}_compact_scribble_overview.png".format(case_name, slice_idx))
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def visualize_compact_class_detail(case_name, slice_idx, class_id, image_2d, gt, pred, class_scribbles, rw_data, output_dir):
    image_norm = normalize01(image_2d)
    gt_c = gt == class_id
    pred_c = pred == class_id
    fp = pred_c & (~gt_c)
    fn = gt_c & (~pred_c)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].imshow(image_norm, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_c, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT class {}".format(class_id))
    axes[1].axis("off")

    axes[2].imshow(pred_c, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred class {}".format(class_id))
    axes[2].axis("off")

    err = np.zeros(gt.shape + (3,), dtype=np.float32)
    err[..., 0] = fp.astype(np.float32)
    err[..., 2] = fn.astype(np.float32)
    axes[3].imshow(image_norm, cmap="gray")
    axes[3].imshow(err, alpha=0.45)
    axes[3].set_title("Error map")
    axes[3].axis("off")

    _plot_scribble_overlay(
        axes[4], image_norm, gt, pred, class_id,
        class_scribbles["v1_entropy"]["pos"], class_scribbles["v1_entropy"]["neg"],
        "Entropy compact\npos={} neg={}".format(
            int(class_scribbles["v1_entropy"]["pos"].sum()), int(class_scribbles["v1_entropy"]["neg"].sum())
        ),
    )
    _plot_scribble_overlay(
        axes[5], image_norm, gt, pred, class_id,
        class_scribbles["v2_evi"]["pos"], class_scribbles["v2_evi"]["neg"],
        "EVI-style compact\npos={} neg={}".format(
            int(class_scribbles["v2_evi"]["pos"].sum()), int(class_scribbles["v2_evi"]["neg"].sum())
        ),
    )
    _plot_scribble_overlay(
        axes[6], image_norm, gt, pred, class_id,
        class_scribbles["v3_consensus"]["pos"], class_scribbles["v3_consensus"]["neg"],
        "Consensus compact\npos={} neg={}".format(
            int(class_scribbles["v3_consensus"]["pos"].sum()), int(class_scribbles["v3_consensus"]["neg"].sum())
        ),
    )

    pseudo_fg, markers, rw_ok = rw_data
    axes[7].imshow(image_norm, cmap="gray")
    if rw_ok:
        axes[7].imshow(pseudo_fg.astype(np.float32), cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axes[7].set_title("Random walker preview")
    else:
        axes[7].imshow(markers, cmap="viridis", alpha=0.5)
        axes[7].set_title("RW unavailable/invalid markers")
    axes[7].axis("off")

    axes[8].imshow(image_norm, cmap="gray")
    axes[8].imshow(err, alpha=0.35)
    py, px = np.where(class_scribbles["v3_consensus"]["pos"])
    ny, nx = np.where(class_scribbles["v3_consensus"]["neg"])
    if len(px) > 0:
        axes[8].scatter(px, py, c="lime", s=4)
    if len(nx) > 0:
        axes[8].scatter(nx, ny, c="red", s=4)
    axes[8].set_title("Consensus scribble + error")
    axes[8].axis("off")

    fig.suptitle("{} slice {} class {}".format(case_name, slice_idx, class_id), fontsize=12)
    save_dir = os.path.join(output_dir, "details")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}_slice{}_class{}_compact_detail.png".format(case_name, slice_idx, class_id))
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compute_scribbles_for_variant(pred, unc, target_classes, args):
    out = {}
    for c in target_classes:
        pos, neg, dbg = generate_compact_signed_scribbles(
            pred,
            unc,
            class_id=c,
            band_radius=args.band_radius,
            safety_margin=args.safety_margin,
            unc_percentile=args.unc_percentile,
            max_components=args.max_components,
            max_scribble_len=args.max_scribble_len,
            scribble_width=args.scribble_width,
            min_component_size=args.min_component_size,
        )
        out[c] = {"pos": pos, "neg": neg, "debug": dbg}
    return out


@torch.no_grad()
def run_slice(net, args, case_name, image_3d, label_3d, slice_idx, device):
    image_2d, gt_2d, x = preprocess_slice(image_3d, label_3d, slice_idx, device)
    logits = forward_logits(net, x, args.model)

    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)

    entropy_unc = normalize01(compute_entropy_uncertainty(logits, args.num_classes))
    evidential_unc = normalize01(compute_evidential_style_uncertainty(logits, args.num_classes))
    consensus_unc = normalize01(entropy_unc * evidential_unc)

    scribbles = {
        "v1_entropy": compute_scribbles_for_variant(pred, entropy_unc, args.target_classes, args),
        "v2_evi": compute_scribbles_for_variant(pred, evidential_unc, args.target_classes, args),
        "v3_consensus": compute_scribbles_for_variant(pred, consensus_unc, args.target_classes, args),
    }

    saved = visualize_compact_overview(
        case_name, slice_idx, image_2d, gt_2d, pred,
        entropy_unc, evidential_unc, consensus_unc, scribbles, args.output_dir
    )
    print("saved overview:", saved)

    for class_id in args.target_classes:
        cls = {
            "v1_entropy": scribbles["v1_entropy"][class_id],
            "v2_evi": scribbles["v2_evi"][class_id],
            "v3_consensus": scribbles["v3_consensus"][class_id],
        }

        if args.run_random_walker:
            rw = random_walker_preview(
                image_2d,
                cls["v3_consensus"]["pos"],
                cls["v3_consensus"]["neg"],
                beta=args.rw_beta,
                mode=args.rw_mode,
            )
        else:
            markers = build_random_walker_markers(cls["v3_consensus"]["pos"], cls["v3_consensus"]["neg"])
            rw = (np.zeros_like(markers, dtype=bool), markers, False)

        detail = visualize_compact_class_detail(
            case_name, slice_idx, class_id, image_2d, gt_2d, pred, cls, rw, args.output_dir
        )
        print("saved detail:", detail)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = get_device(args.device)
    print("using device:", device)
    print("uncertainty methods: Entropy + Evidential-style Approx. (+ Consensus)")

    net = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    net = net.to(device)
    state_dict = load_checkpoint_flexible(args.checkpoint)
    load_state_flexible(net, state_dict)
    net.eval()

    rng = random.Random(args.seed)
    case_list = list_test_cases(args.root_path)
    selected_cases = select_cases(case_list, args.num_cases, rng)
    print("selected cases:", selected_cases)

    for case in selected_cases:
        image_3d, label_3d = load_case(args.root_path, case)
        slices = select_slice_indices(label_3d, args.slices_per_case, rng)
        print("case {} selected slices {}".format(case, slices))
        for sidx in slices:
            run_slice(net, args, case, image_3d, label_3d, sidx, device)


if __name__ == "__main__":
    main()
