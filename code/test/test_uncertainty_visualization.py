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
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F

from networks.unet import UNet, UNet_HL

np.bool = np.bool_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../../data/ACDC")
    parser.add_argument("--model", type=str, default="unet_hl")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--slices_per_case", type=int, default=1)
    parser.add_argument("--tta_runs", type=int, default=8)
    parser.add_argument("--mc_runs", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./uncertainty_vis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_overlay", action="store_true")
    return parser.parse_args()


def normalize01(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=np.float32)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + eps)


def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name, in_chns, num_classes, device):
    if model_name == "unet_hl":
        net = UNet_HL(in_chns=in_chns, class_num=num_classes)
    elif model_name == "unet":
        net = UNet(in_chns=in_chns, class_num=num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(model_name))
    return net.to(device)


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


def forward_logits(net, x, model_name):
    out = net(x)
    if model_name == "unet_hl":
        return out[0] if isinstance(out, (tuple, list)) else out
    return out[0] if isinstance(out, (tuple, list)) else out


def list_test_cases(root_path):
    test_txt = os.path.join(root_path, "test.txt")
    with open(test_txt, "r") as f:
        cases = [line.strip() for line in f if line.strip()]
    return [os.path.splitext(case)[0] for case in cases]


def select_cases(case_list, num_cases, rng):
    if not case_list:
        raise ValueError("No cases found in test.txt")
    num_cases = min(num_cases, len(case_list))
    return rng.sample(case_list, num_cases)


def load_case(root_path, case_name):
    h5_path = os.path.join(root_path, "ACDC_training_volumes", case_name + ".h5")
    with h5py.File(h5_path, "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]
    return image, label


def select_slice_indices(label, slices_per_case, rng):
    fg_indices = np.where(label.reshape(label.shape[0], -1).sum(axis=1) > 0)[0]
    candidates = fg_indices.tolist() if len(fg_indices) > 0 else list(range(label.shape[0]))
    if not candidates:
        raise ValueError("Case has no slices")
    if slices_per_case >= len(candidates):
        return candidates
    return sorted(rng.sample(candidates, slices_per_case))


def preprocess_slice(image_3d, label_3d, slice_idx, device):
    image_2d = image_3d[slice_idx]
    gt_2d = label_3d[slice_idx]
    x, y = image_2d.shape
    image_256 = zoom(image_2d, (256 / x, 256 / y), order=0)
    gt_256 = zoom(gt_2d, (256 / x, 256 / y), order=0)
    input_tensor = torch.from_numpy(image_256).unsqueeze(0).unsqueeze(0).float().to(device)
    return image_256, gt_256, input_tensor


def maxprob_uncertainty(logits):
    prob = torch.softmax(logits, dim=1)
    unc = 1.0 - prob.max(dim=1).values
    pred = prob.argmax(dim=1)
    return prob, pred, unc


def entropy_uncertainty(prob, num_classes):
    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
    return entropy / np.log(num_classes)


def tta_transforms():
    return [
        ("identity", lambda x: x, lambda x: x),
        ("hflip", lambda x: torch.flip(x, dims=(-1,)), lambda x: torch.flip(x, dims=(-1,))),
        ("vflip", lambda x: torch.flip(x, dims=(-2,)), lambda x: torch.flip(x, dims=(-2,))),
        ("hvflip", lambda x: torch.flip(x, dims=(-2, -1)), lambda x: torch.flip(x, dims=(-2, -1))),
        ("rot90", lambda x: torch.rot90(x, 1, dims=(-2, -1)), lambda x: torch.rot90(x, -1, dims=(-2, -1))),
        ("rot180", lambda x: torch.rot90(x, 2, dims=(-2, -1)), lambda x: torch.rot90(x, 2, dims=(-2, -1))),
        ("rot270", lambda x: torch.rot90(x, 3, dims=(-2, -1)), lambda x: torch.rot90(x, -3, dims=(-2, -1))),
        ("transpose", lambda x: torch.transpose(x, -2, -1), lambda x: torch.transpose(x, -2, -1)),
    ]


@torch.no_grad()
def tta_uncertainty(net, x, model_name, num_classes, num_runs):
    transforms = tta_transforms()[: max(1, min(num_runs, len(tta_transforms())))]
    probs = []
    for _, aug, inv_aug in transforms:
        logits_aug = forward_logits(net, aug(x), model_name)
        prob_aug = torch.softmax(logits_aug, dim=1)
        probs.append(inv_aug(prob_aug))
    probs = torch.stack(probs, dim=0)
    mean_prob = probs.mean(dim=0)
    var_map = probs.var(dim=0, unbiased=False).mean(dim=1)
    entropy_mean = entropy_uncertainty(mean_prob, num_classes)
    pred = mean_prob.argmax(dim=1)
    return mean_prob, pred, var_map, entropy_mean


def get_dropout_modules(net):
    return [module for module in net.modules() if module.__class__.__name__.startswith("Dropout")]


def enable_dropout_only(net):
    net.eval()
    dropout_modules = get_dropout_modules(net)
    for module in dropout_modules:
        module.train()
    return len(dropout_modules)


@torch.no_grad()
def mc_dropout_uncertainty(net, x, model_name, num_runs):
    num_dropout = enable_dropout_only(net)
    probs = []
    for _ in range(max(1, num_runs)):
        logits = forward_logits(net, x, model_name)
        probs.append(torch.softmax(logits, dim=1))
    probs = torch.stack(probs, dim=0)
    mean_prob = probs.mean(dim=0)
    var_map = probs.var(dim=0, unbiased=False).mean(dim=1)
    pred = mean_prob.argmax(dim=1)
    net.eval()
    return mean_prob, pred, var_map, num_dropout


def evidential_style_uncertainty(logits, num_classes):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    s = alpha.sum(dim=1)
    return num_classes / (s + 1e-8)


def to_numpy_map(tensor):
    return tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


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


def get_segmentation_cmap():
    return ListedColormap(
        [
            (0.0, 0.0, 0.0),
            (0.88, 0.24, 0.22),
            (0.23, 0.69, 0.29),
            (0.18, 0.45, 0.86),
        ]
    )


def save_panel_figure(case_name, slice_idx, image_256, gt_256, pred_256, uncertainty_maps, output_dir):
    image_norm = normalize01(image_256)
    seg_cmap = get_segmentation_cmap()
    fig, axes = plt.subplots(1, 8, figsize=(24, 4))

    panels = [
        ("Image", image_norm, "gray", None),
        ("Ground Truth", gt_256, seg_cmap, (0, 3)),
        ("Prediction", pred_256, seg_cmap, (0, 3)),
        ("MaxProb Unc", uncertainty_maps["maxprob"], "jet", (0, 1)),
        ("Entropy Unc", uncertainty_maps["entropy"], "jet", (0, 1)),
        ("TTA Var Unc", uncertainty_maps["tta"], "jet", (0, 1)),
        (uncertainty_maps["mc_title"], uncertainty_maps["mc"], "jet", (0, 1)),
        ("Evidential-style Unc", uncertainty_maps["evidential"], "jet", (0, 1)),
    ]

    for ax, (title, data, cmap, limits) in zip(axes, panels):
        if limits is None:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("{} slice {}".format(case_name, slice_idx), fontsize=12)
    save_path = os.path.join(output_dir, "{}_slice{}_uncertainty.png".format(case_name, slice_idx))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_overlay_figure(case_name, slice_idx, image_256, overlay_map, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(normalize01(image_256), cmap="gray")
    ax.imshow(overlay_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title("{} slice {} overlay".format(case_name, slice_idx))
    ax.axis("off")
    save_path = os.path.join(output_dir, "{}_slice{}_overlay.png".format(case_name, slice_idx))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


@torch.no_grad()
def run_case(net, args, case_name, image_3d, label_3d, slice_idx, device):
    image_256, gt_256, input_tensor = preprocess_slice(image_3d, label_3d, slice_idx, device)

    logits = forward_logits(net, input_tensor, args.model)
    prob, pred, maxprob_unc = maxprob_uncertainty(logits)
    entropy_unc = entropy_uncertainty(prob, args.num_classes)
    _, _, tta_var_unc, _ = tta_uncertainty(net, input_tensor, args.model, args.num_classes, args.tta_runs)
    _, _, mc_var_unc, num_dropout = mc_dropout_uncertainty(net, input_tensor, args.model, args.mc_runs)
    evidential_unc = evidential_style_uncertainty(logits, args.num_classes)

    pred_256 = pred.squeeze(0).detach().cpu().numpy().astype(np.int32)
    uncertainty_maps = {
        "maxprob": normalize01(to_numpy_map(maxprob_unc)),
        "entropy": normalize01(to_numpy_map(entropy_unc)),
        "tta": normalize01(to_numpy_map(tta_var_unc)),
        "mc": normalize01(to_numpy_map(mc_var_unc)),
        "mc_title": "MC Dropout Unc" if num_dropout > 0 else "MC Dropout Unc\npossibly inactive",
        "evidential": normalize01(to_numpy_map(evidential_unc)),
    }

    saved_panel = save_panel_figure(
        case_name=case_name,
        slice_idx=slice_idx,
        image_256=image_256,
        gt_256=gt_256,
        pred_256=pred_256,
        uncertainty_maps=uncertainty_maps,
        output_dir=args.output_dir,
    )
    print("saved panel:", saved_panel)

    if args.save_overlay:
        saved_overlay = save_overlay_figure(
            case_name=case_name,
            slice_idx=slice_idx,
            image_256=image_256,
            overlay_map=uncertainty_maps["tta"],
            output_dir=args.output_dir,
        )
        print("saved overlay:", saved_overlay)


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
    print("evidential warning: output is an evidential-style uncertainty approximation unless the checkpoint was trained with evidential loss.")

    net = build_model(args.model, in_chns=1, num_classes=args.num_classes, device=device)
    state_dict = load_checkpoint_state(args.checkpoint)
    load_model_weights(net, state_dict)
    net.eval()

    rng = random.Random(args.seed)
    cases = select_cases(list_test_cases(args.root_path), args.num_cases, rng)
    print("selected cases:", cases)

    for case_name in cases:
        image_3d, label_3d = load_case(args.root_path, case_name)
        selected_slices = select_slice_indices(label_3d, args.slices_per_case, rng)
        print("case {} selected slices {}".format(case_name, selected_slices))
        for slice_idx in selected_slices:
            print("visualizing case {} slice {}".format(case_name, slice_idx))
            run_case(net, args, case_name, image_3d, label_3d, slice_idx, device)


if __name__ == "__main__":
    main()
