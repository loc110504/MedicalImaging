import numpy as np
import torch


def build_quantum_guided_reliable_masks(
    prob_u,
    prob_m,
    Q_u,
    Q_m,
    label,
    ignore_index,
    tau_u,
    tau_m,
    tau_q,
):
    conf_u, pseudo_u = torch.max(prob_u.detach(), dim=1)
    conf_m, pseudo_m = torch.max(prob_m.detach(), dim=1)
    qconf_u, qlabel_u = torch.max(Q_u.detach(), dim=1)
    qconf_m, qlabel_m = torch.max(Q_m.detach(), dim=1)

    unknown = label == ignore_index
    agreement_u = pseudo_u == qlabel_u
    agreement_m = pseudo_m == qlabel_m

    R_u = agreement_u & (conf_u > tau_u) & (qconf_u > tau_q) & unknown
    R_m = agreement_m & (conf_m > tau_m) & (qconf_m > tau_q) & unknown

    return {
        "pseudo_u": pseudo_u,
        "pseudo_m": pseudo_m,
        "R_u": R_u,
        "R_m": R_m,
        "conf_u": conf_u,
        "conf_m": conf_m,
        "qconf_u": qconf_u,
        "qconf_m": qconf_m,
        "agreement_u": agreement_u,
        "agreement_m": agreement_m,
    }


def safe_masked_mean(values, mask):
    if mask.sum() < 1:
        return np.nan
    return float(values[mask].mean().item())


def summarize_reliable_masks(pseudo_info):
    R_u = pseudo_info["R_u"]
    R_m = pseudo_info["R_m"]
    agreement_u = pseudo_info["agreement_u"]
    agreement_m = pseudo_info["agreement_m"]
    return {
        "reliable_ratio_u": R_u.float().mean(),
        "reliable_ratio_m": R_m.float().mean(),
        "agreement_ratio_u": agreement_u.float().mean(),
        "agreement_ratio_m": agreement_m.float().mean(),
        "conf_u_mean": safe_masked_mean(pseudo_info["conf_u"], R_u),
        "conf_m_mean": safe_masked_mean(pseudo_info["conf_m"], R_m),
        "qconf_u_mean": safe_masked_mean(pseudo_info["qconf_u"], R_u),
        "qconf_m_mean": safe_masked_mean(pseudo_info["qconf_m"], R_m),
    }


def masked_label_accuracy(pred, target, mask):
    if mask.sum() < 1:
        return np.nan
    return float((pred[mask] == target[mask]).float().mean().item())
