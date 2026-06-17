import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
UTILS_DIR = os.path.join(BASE_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory
from xnetv2_wavelet import build_wavelet_batch_from_tensor

np.bool = np.bool_

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--data", type=str, default="ACDC", choices=["ACDC", "MSCMR"], help="dataset name")
parser.add_argument(
    "--model_path",
    type=str,
    default="../../checkpoints/ACDC_Wavelet_UNet_Scribble/unet_hl_best_model.pth",
    help="checkpoint path",
)
parser.add_argument(
    "--model",
    type=str,
    default="unet_hl",
    choices=["unet", "unet_hl"],
    help="shared-weight backbone",
)
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--in_chns", type=int, default=1, help="input channels")
parser.add_argument("--mode", type=str, default="main", choices=["main", "low", "high", "mean"], help="inference mode")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256], help="patch size")
parser.add_argument("--wavelet_type", type=str, default="haar", help="wavelet basis")
parser.add_argument("--val_alpha", nargs=2, type=float, default=[0.2, 0.2], help="validation alpha range")
parser.add_argument("--val_beta", nargs=2, type=float, default=[0.2, 0.2], help="validation beta range")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def minmax_normalize_batch(batch, eps=1e-8):
    flat = batch.view(batch.size(0), -1)
    min_value = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_value = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (batch - min_value) / (max_value - min_value + eps)


def extract_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        raise ValueError("Unsupported dict output keys: {}".format(list(output.keys())))
    return output


def forward_three_views(model, x_main, x_low, x_high):
    x_main = minmax_normalize_batch(x_main)
    x_low = minmax_normalize_batch(x_low)
    x_high = minmax_normalize_batch(x_high)
    logits_main = extract_logits(model(x_main))
    logits_low = extract_logits(model(x_low))
    logits_high = extract_logits(model(x_high))
    return logits_main, logits_low, logits_high


class WaveletInferenceWrapper(torch.nn.Module):
    def __init__(self, model, mode, wavelet_type, alpha, beta):
        super().__init__()
        self.model = model
        self.mode = mode
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        x = minmax_normalize_batch(x)
        x_low, x_high = build_wavelet_batch_from_tensor(
            x,
            wavelet_type=self.wavelet_type,
            alpha=self.alpha,
            beta=self.beta,
        )
        logits_main, logits_low, logits_high = forward_three_views(self.model, x, x_low, x_high)
        if self.mode == "main":
            return logits_main
        if self.mode == "low":
            return logits_low
        if self.mode == "high":
            return logits_high
        if self.mode == "mean":
            prob = (
                torch.softmax(logits_main, dim=1)
                + torch.softmax(logits_low, dim=1)
                + torch.softmax(logits_high, dim=1)
            ) / 3.0
            return torch.log(prob + 1e-8)
        raise ValueError("Unsupported mode: {}".format(self.mode))


def load_model_state(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def get_case_list(cfg):
    if cfg.data == "ACDC":
        with open(os.path.join(cfg.root_path, "test.txt"), "r") as f:
            image_list = f.readlines()
        return sorted([item.replace("\n", "").split(".")[0] for item in image_list])

    test_dir = os.path.join(cfg.root_path, "MSCMR_testing_volumes")
    image_list = [os.path.splitext(filename)[0] for filename in os.listdir(test_dir) if filename.endswith(".h5")]
    image_list.sort()
    return image_list


def open_case_h5(case, cfg):
    if cfg.data == "ACDC":
        return h5py.File(os.path.join(cfg.root_path, "ACDC_training_volumes", "{}.h5".format(case)), "r")
    return h5py.File(os.path.join(cfg.root_path, "MSCMR_testing_volumes", "{}.h5".format(case)), "r")


def test_single_volume(case, net, cfg):
    h5f = open_case_h5(case, cfg)
    image = h5f["image"][:]
    label = h5f["label"][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        cur_slice = image[ind, :, :]
        x, y = cur_slice.shape
        cur_slice = zoom(cur_slice, (cfg.patch_size[0] / x, cfg.patch_size[1] / y), order=0)
        input_tensor = torch.from_numpy(cur_slice).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            out = net(input_tensor)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / cfg.patch_size[0], y / cfg.patch_size[1]), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    return first_metric, second_metric, third_metric


def inference(cfg):
    image_list = get_case_list(cfg)

    model = net_factory(
        net_type=cfg.model,
        in_chns=cfg.in_chns,
        class_num=cfg.num_classes,
    )
    load_model_state(model, cfg.model_path)
    print("init weight from {}".format(cfg.model_path))

    net = WaveletInferenceWrapper(
        model=model,
        mode=cfg.mode,
        wavelet_type=cfg.wavelet_type,
        alpha=cfg.val_alpha,
        beta=cfg.val_beta,
    ).cuda()
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, cfg)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [
        first_total / len(image_list),
        second_total / len(image_list),
        third_total / len(image_list),
    ]
    return avg_metric


if __name__ == "__main__":
    metrics = inference(args)
    print(metrics)
    print("RV:", metrics[0], " | Myo:", metrics[1], " | LV:", metrics[2])
    print((metrics[0] + metrics[1] + metrics[2]) / 3)
