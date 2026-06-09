from networks.unet import UNet, UNet_HL
from networks.unet_cct import UNet_CCT
from networks.unet_lgdt import UNet_LGDT

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_hl":
        net = UNet_HL(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_lgdt":
        net = UNet_LGDT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mamba_unet":
        from networks.mamba_unet_2d import MambaUNet2D
        net = MambaUNet2D(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "quanmambascrib":
        from networks.quan_mamba_scrib import QuanMambaScrib
        net = QuanMambaScrib(
            in_chns=in_chns,
            class_num=class_num,
            unet_type="unet_hl",
            mamba_variant="vmunet",
            qpim_backend="torch_angle_fidelity",
            ignore_index=class_num,
        ).cuda()
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    return net
