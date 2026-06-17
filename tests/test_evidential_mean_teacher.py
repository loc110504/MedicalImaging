import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.evidential import (  # noqa: E402
    asymmetric_uncertainty_mse_loss,
    build_mt_confidence_pseudo_label,
    evidential_prediction,
    masked_soft_ce_from_prob,
    partial_ce_from_prob,
)


def test_probability_normalization():
    raw = torch.randn(2, 4, 16, 16)
    result = evidential_prediction(raw, 4)

    assert result["prob"].shape == (2, 4, 16, 16)
    assert result["uncertainty"].shape == (2, 1, 16, 16)
    assert torch.allclose(
        result["prob"].sum(dim=1),
        torch.ones_like(result["prob"][:, 0]),
        atol=1e-5,
    )


def test_uncertainty_range():
    result = evidential_prediction(torch.randn(2, 4, 16, 16), 4)
    assert result["uncertainty"].min() > 0
    assert result["uncertainty"].max() <= 1 + 1e-6


def test_zero_evidence_limit():
    raw = torch.full((1, 4, 2, 2), -100.0)
    result = evidential_prediction(raw, 4)
    expected_prob = torch.full((1, 4, 2, 2), 0.25)
    expected_uncertainty = torch.ones((1, 1, 2, 2))

    assert torch.allclose(result["prob"], expected_prob, atol=1e-4)
    assert torch.allclose(result["uncertainty"], expected_uncertainty, atol=1e-4)


def test_more_evidence_means_less_uncertainty():
    low = evidential_prediction(torch.zeros(1, 4, 8, 8), 4)
    high = evidential_prediction(torch.full((1, 4, 8, 8), 10.0), 4)
    assert high["uncertainty"].mean() < low["uncertainty"].mean()


def test_one_sided_loss():
    student_u = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
    teacher_u = torch.tensor([[[[0.3, 0.7]]]])
    mask = torch.ones_like(student_u)

    loss = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.0)
    assert torch.allclose(loss, torch.tensor(0.125), atol=1e-6)


def test_teacher_receives_no_gradient():
    student_u = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
    teacher_u = torch.tensor([[[[0.3, 0.7]]]], requires_grad=True)
    mask = torch.ones_like(student_u)

    loss = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.0)
    loss.backward()

    assert student_u.grad is not None
    assert teacher_u.grad is None


def test_reliable_mask_excludes_loss():
    student_u = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
    teacher_u = torch.tensor([[[[0.3, 0.1]]]])
    mask = torch.tensor([[[[0.0, 1.0]]]])

    loss = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.0)
    assert torch.allclose(loss, torch.tensor(0.01), atol=1e-6)


def test_empty_mask_returns_zero():
    student_u = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
    teacher_u = torch.tensor([[[[0.3, 0.1]]]])
    mask = torch.zeros_like(student_u)

    loss = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.0)
    assert loss.shape == torch.Size([])
    assert torch.equal(loss, torch.tensor(0.0))


def test_margin_behavior():
    student_u = torch.tensor([[[[0.55]]]])
    teacher_u = torch.tensor([[[[0.50]]]])
    mask = torch.ones_like(student_u)

    loss_margin = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.10)
    loss_no_margin = asymmetric_uncertainty_mse_loss(student_u, teacher_u, mask, margin=0.0)

    assert torch.equal(loss_margin, torch.tensor(0.0))
    assert loss_no_margin > 0


def test_partial_ce_ignores_unlabeled_pixels():
    prob_a = torch.tensor(
        [[
            [[0.8, 0.1], [0.7, 0.2]],
            [[0.2, 0.9], [0.3, 0.8]],
        ]],
        dtype=torch.float32,
    )
    prob_b = prob_a.clone()
    prob_b[:, :, 1, 1] = torch.tensor([0.99, 0.01])
    label = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)

    loss_a = partial_ce_from_prob(prob_a, label, ignore_index=2)
    loss_b = partial_ce_from_prob(prob_b, label, ignore_index=2)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_integration_backward():
    torch.manual_seed(0)
    student_raw = torch.randn(2, 4, 8, 8, requires_grad=True)
    teacher_raw = torch.randn(2, 4, 8, 8, requires_grad=True)
    label = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)
    label[0, :, :] = 4
    label[1, :4, :4] = 4

    student_evi = evidential_prediction(student_raw, 4)
    with torch.no_grad():
        teacher_evi = evidential_prediction(teacher_raw, 4)

    loss_pce = partial_ce_from_prob(student_evi["prob"], label, ignore_index=4)
    pseudo_info = build_mt_confidence_pseudo_label(
        student_prob=student_evi["prob"],
        teacher_prob=teacher_evi["prob"],
        label=label,
        agree_thresh=0.2,
        disagree_thresh=0.2,
        margin_thresh=0.0,
        ignore_index=4,
        pseudo_mask_mode="unlabeled",
    )
    loss_pseudo = masked_soft_ce_from_prob(
        student_prob=student_evi["prob"],
        target_prob=pseudo_info["soft_pseudo_label"],
        mask=pseudo_info["reliable_mask"],
    )
    loss_unc = asymmetric_uncertainty_mse_loss(
        student_uncertainty=student_evi["uncertainty"],
        teacher_uncertainty=teacher_evi["uncertainty"],
        reliable_mask=pseudo_info["reliable_mask"],
        margin=0.0,
    )

    loss = loss_pce + loss_pseudo + loss_unc
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(student_raw.grad).all()
    assert teacher_raw.grad is None
