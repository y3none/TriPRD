import inspect

import torch

from models.triprd import (
    TriPRDLoss,
    TriPRDStudent,
    TriPRDTeacher,
    teacher_gate,
)


def _batch(size=16):
    torch.manual_seed(7)
    rgb = torch.randn(2, 3, size, size)
    thermal = torch.randn(2, 1, size, size)
    depth = torch.randn(2, 1, size, size)
    mask = torch.zeros(2, 1, size, size)
    mask[:, :, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    return rgb, thermal, depth, mask


def test_forward_signatures_and_shapes():
    rgb, thermal, depth, _ = _batch()
    student = TriPRDStudent(base_channels=4)
    teacher = TriPRDTeacher(base_channels=4)
    assert list(inspect.signature(student.forward).parameters) == ["rgb"]
    assert list(inspect.signature(teacher.forward).parameters) == ["rgb", "thermal", "depth"]
    sout = student(rgb)
    tout = teacher(rgb, thermal, depth)
    assert sout["logits"].shape == (2, 1, 16, 16)
    assert sout["boundary_logits"].shape == sout["logits"].shape
    assert sout["features"].ndim == 4
    assert tout["modality_logits"].shape == (2, 3, 16, 16)


def test_teacher_all_parameters_receive_supervised_gradients():
    rgb, thermal, depth, mask = _batch()
    teacher = TriPRDTeacher(base_channels=4)
    out = teacher(rgb, thermal, depth)
    losses = TriPRDLoss(mode="none")(out, mask)
    losses["total"].backward()
    missing = [name for name, p in teacher.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_kd_temperature_is_nonnegative_and_changes_value():
    rgb, thermal, depth, mask = _batch()
    teacher = TriPRDTeacher(base_channels=4).eval()
    student = TriPRDStudent(base_channels=4)
    with torch.no_grad():
        tout = teacher(rgb, thermal, depth)
    sout = student(rgb)
    low = TriPRDLoss(mode="logit", temperature=1.0)(sout, mask, tout)
    high = TriPRDLoss(mode="logit", temperature=4.0)(sout, mask, tout)
    assert low["logit_kd"].item() >= 0
    assert high["logit_kd"].item() >= 0
    assert not torch.allclose(low["logit_kd"], high["logit_kd"])


def test_gate_is_finite_and_detached():
    logits = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]]], requires_grad=True)
    gate = teacher_gate({"modality_logits": logits})
    assert gate.shape == (1, 1, 1, 1)
    assert torch.isfinite(gate).all()
    assert not gate.requires_grad
    assert 0.1 <= gate.item() <= 1.0


def test_relation_mode_uses_cosine_gram_geometry():
    # A channel rotation is orthogonal, hence preserves every cosine Gram
    # matrix.  Set logits equal so the relation component is isolated.
    torch.manual_seed(3)
    b, c, h, w = 1, 4, 8, 8
    features = torch.randn(b, c, h, w)
    q, _ = torch.linalg.qr(torch.randn(c, c))
    rotated = torch.einsum("ij,bjhw->bihw", q, features)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, 2:6, 2:6] = 1
    common_logits = torch.randn(b, 1, h, w)
    out = {"logits": common_logits, "boundary_logits": common_logits, "features": rotated}
    teacher = {"logits": common_logits, "boundary_logits": common_logits, "features": features}
    value = TriPRDLoss(mode="relation", relation_tokens=4)(out, mask, teacher)["relation_kd"]
    assert value.item() < 1e-5
