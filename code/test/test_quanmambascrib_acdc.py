import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.quan_mamba_scrib import QuanMambaScrib

np.bool = np.bool_

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--model_path", type=str, default="../../checkpoints/ACDC_QuanMambaScrib/quanmambascrib_best_model.pth")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--mode", type=str, default="ensemble", choices=["ensemble", "mamba", "unet"], help="inference mode")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256], help="patch size")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        out = self.model(x, scribble_label=None, update_memory=False, return_q=False)
        if self.mode == "ensemble":
            return torch.log(out["prob_ensemble"] + 1e-8)
        if self.mode == "mamba":
            return out["logits_m"]
        if self.mode == "unet":
            return out["logits_u"]
        raise ValueError("Unsupported mode: {}".format(self.mode))


def load_model_state(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def test_single_volume(case, net, cfg):
    h5f = h5py.File(os.path.join(cfg.root_path, "ACDC_training_volumes", "{}.h5".format(case)), "r")
    image = h5f["image"][:]
    label = h5f["label"][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        cur_slice = image[ind, :, :]
        x, y = cur_slice.shape[0], cur_slice.shape[1]
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
    with open(os.path.join(cfg.root_path, "test.txt"), "r") as f:
        image_list = f.readlines()
    image_list = sorted([item.replace("\n", "").split(".")[0] for item in image_list])

    model = QuanMambaScrib(in_chns=1, class_num=cfg.num_classes).cuda()
    load_model_state(model, cfg.model_path)
    print("init weight from {}".format(cfg.model_path))

    net = InferenceWrapper(model, cfg.mode).cuda()
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
