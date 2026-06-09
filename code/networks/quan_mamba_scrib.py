import torch
import torch.nn as nn

from networks.mamba_unet_2d import MambaUNet2D
from networks.qpim import QuantumPrototypeInteractionModule
from networks.unet import UNet_HL


class QuanMambaScrib(nn.Module):
    def __init__(
        self,
        in_chns,
        class_num,
        unet_type="unet_hl",
        mamba_variant="vmunet",
        qpim_backend="torch_angle_fidelity",
        qpim_dim=8,
        qpim_tau=0.5,
        qpim_momentum=0.99,
        qpim_normalize_z=True,
        qpim_detach_prototypes=False,
        ignore_index=None,
    ):
        super().__init__()
        if unet_type != "unet_hl":
            raise ValueError("QuanMambaScrib currently expects unet_type='unet_hl'")

        self.class_num = class_num
        self.ignore_index = class_num if ignore_index is None else ignore_index

        self.unet = UNet_HL(in_chns=in_chns, class_num=class_num)
        self.unet_feature_stage = "bottleneck"

        self.mamba_unet = MambaUNet2D(
            in_chns=in_chns,
            class_num=class_num,
            mamba_variant=mamba_variant,
            return_feature_stage="decoder_low",
        )

        self.qpim = QuantumPrototypeInteractionModule(
            in_channels_u=256,
            in_channels_m=self.mamba_unet.feature_channels,
            num_classes=class_num,
            proj_dim=qpim_dim,
            tau_q=qpim_tau,
            momentum=qpim_momentum,
            ignore_index=self.ignore_index,
            backend=qpim_backend,
            normalize_z=qpim_normalize_z,
            detach_prototypes=qpim_detach_prototypes,
        )

    def forward(self, x, scribble_label=None, update_memory=True, return_q=True):
        logits_u, feat_u = self.unet(x, return_features=True, feature_stage=self.unet_feature_stage)
        logits_m, feat_m = self.mamba_unet(x, return_features=True)

        prob_u = torch.softmax(logits_u, dim=1)
        prob_m = torch.softmax(logits_m, dim=1)

        output = {
            "logits_u": logits_u,
            "logits_m": logits_m,
            "prob_u": prob_u,
            "prob_m": prob_m,
            "feat_u": feat_u,
            "feat_m": feat_m,
        }

        if return_q and scribble_label is not None:
            q_out = self.qpim(
                feat_u=feat_u,
                feat_m=feat_m,
                scribble_label=scribble_label,
                target_size=x.shape[-2:],
                update_memory=update_memory,
            )
            output.update(q_out)

        if scribble_label is None:
            output["prob_ensemble"] = 0.5 * (prob_u + prob_m)

        return output
