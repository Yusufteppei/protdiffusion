# tests/test_ipa.py

import torch
import pytest

from rfdiffusion.config import d_pair, d_res, device
from rfdiffusion.modules import InvariantPointAttention
from rfdiffusion.geometry import Rigid, Rotation


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def make_rigid(batch, length, device="cpu"):
    """
    Create random rigid frames.
    """
    rot = torch.randn(batch, length, 3, 3, device=device)

    # QR -> orthogonal matrices
    q, r = torch.linalg.qr(rot)

    # Make determinant +1 so these are proper rotations.
    det = torch.det(q)
    q[det < 0, :, 0] *= -1

    trans = torch.randn(batch, length, 3, device=device)

    return Rigid(
        Rotation(q),
        trans,
    )


def make_inputs(
    batch=2,
    length=8,
    single_dim=d_res,
    pair_dim=d_pair,
    device=device,
):
    single = torch.randn(
        batch, length, single_dim,
        device=device,
    )

    pair = torch.randn(
        batch, length, length, pair_dim,
        device=device,
    )

    rigids = make_rigid(
        batch,
        length,
        device=device,
    )

    mask = torch.ones(
        batch,
        length,
        dtype=torch.bool,
        device=device,
    )

    return single, pair, rigids, mask


def global_transform(rigids: Rigid, rotation: Rotation, translation):
    """
    Apply the same global rigid transformation to every residue frame.

    If the InvariantPointAttention implementation is invariant, applying this transformation
    to every frame should not change the attention output.
    """
    R = rotation
    t = translation

    new_R = torch.matmul(
        R,
        rigids.rotation.matrix,
    )

    new_t = torch.einsum(
        "...ij,...j->...i",
        R,
        rigids.translation,
    ) + t

    return Rigid(
        Rotation(new_R),
        new_t,
    )


# ------------------------------------------------------------
# Basic forward-pass tests
# ------------------------------------------------------------

def test_ipa_forward_shape():
    """
    InvariantPointAttention should preserve the residue dimension and produce the
    expected output dimensionality.
    """
    single, pair, rigids, mask = make_inputs()

    ipa = InvariantPointAttention()

    output = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    assert output.shape[0] == single.shape[0]
    assert output.shape[1] == single.shape[1]


def test_ipa_output_is_finite():
    """
    Normal inputs should not produce NaNs or infinities.
    """
    single, pair, rigids, mask = make_inputs()

    ipa = InvariantPointAttention()

    output = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    assert torch.isfinite(output).all()


# ------------------------------------------------------------
# Translation invariance
# ------------------------------------------------------------

def test_ipa_translation_invariance():
    """
    InvariantPointAttention's geometric attention should be invariant to translating
    the entire structure by the same vector.
    """
    torch.manual_seed(0)

    single, pair, rigids, mask = make_inputs()

    ipa = InvariantPointAttention()

    output_1 = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    translation = torch.tensor(
        [10.0, -7.0, 4.0]
    )

    translated = Rigid(
        rigids.rotation,
        rigids.translation + translation,
    )

    output_2 = ipa(
        single,
        pair,
        translated,
        mask,
    )

    torch.testing.assert_close(
        output_1,
        output_2,
        rtol=1e-5,
        atol=1e-5,
    )


# ------------------------------------------------------------
# Rotation invariance
# ------------------------------------------------------------

def test_ipa_rotation_invariance():
    """
    Rotating the entire structure by the same global rotation
    should not change InvariantPointAttention's output.
    """
    torch.manual_seed(0)

    single, pair, rigids, mask = make_inputs()

    ipa = InvariantPointAttention()

    output_1 = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    # Generate a global rotation.
    x = torch.randn(3, 3)
    Q, R = torch.linalg.qr(x)

    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    translation = torch.zeros(3)

    rotated = global_transform(
        rigids,
        Q,
        translation,
    )

    output_2 = ipa(
        single,
        pair,
        rotated,
        mask,
    )

    torch.testing.assert_close(
        output_1,
        output_2,
        rtol=1e-5,
        atol=1e-5,
    )


# ------------------------------------------------------------
# Combined SE(3) invariance
# ------------------------------------------------------------

def test_ipa_se3_invariance():
    """
    InvariantPointAttention should be invariant to an arbitrary global rigid-body
    transformation.
    """
    torch.manual_seed(42)

    single, pair, rigids, mask = make_inputs()

    ipa = InvariantPointAttention()

    output_1 = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    # Random global rotation.
    x = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(x)

    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    translation = torch.randn(3)

    transformed = global_transform(
        rigids,
        Q,
        translation,
    )

    output_2 = ipa(
        single,
        pair,
        transformed,
        mask,
    )

    torch.testing.assert_close(
        output_1,
        output_2,
        rtol=1e-5,
        atol=1e-5,
    )


# ------------------------------------------------------------
# Masking
# ------------------------------------------------------------

def test_ipa_masking():
    """
    Masked residues should not contribute to attention.
    """
    torch.manual_seed(0)

    single, pair, rigids, mask = make_inputs()

    # Mask the final residue.
    mask[:, -1] = False

    ipa = InvariantPointAttention()

    output = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    assert torch.isfinite(output).all()


def test_ipa_mask_changes_attention():
    """
    Changing a masked residue should not affect the unmasked
    residues.
    """
    torch.manual_seed(0)

    single, pair, rigids, mask = make_inputs()

    mask[:, -1] = False

    ipa = InvariantPointAttention()

    output_1 = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    # Completely change the masked residue.
    single_2 = single.clone()
    pair_2 = pair.clone()

    single_2[:, -1] = torch.randn_like(
        single_2[:, -1]
    )

    pair_2[:, -1, :] = torch.randn_like(
        pair_2[:, -1, :]
    )

    pair_2[:, :, -1] = torch.randn_like(
        pair_2[:, :, -1]
    )

    output_2 = ipa(
        single_2,
        pair_2,
        rigids,
        mask,
    )

    # Unmasked residues should remain unchanged.
    torch.testing.assert_close(
        output_1[:, :-1],
        output_2[:, :-1],
        rtol=1e-5,
        atol=1e-5,
    )


# ------------------------------------------------------------
# Gradient tests
# ------------------------------------------------------------

def test_ipa_gradients():
    """
    InvariantPointAttention should be differentiable with respect to its learned
    representations.
    """
    single, pair, rigids, mask = make_inputs()

    single.requires_grad_(True)
    pair.requires_grad_(True)

    ipa = InvariantPointAttention()

    output = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    loss = output.square().mean()

    loss.backward()

    assert single.grad is not None
    assert pair.grad is not None

    assert torch.isfinite(
        single.grad
    ).all()

    assert torch.isfinite(
        pair.grad
    ).all()


# ------------------------------------------------------------
# Batch independence
# ------------------------------------------------------------

def test_ipa_batch_independence():
    """
    Samples in a batch should not affect one another.
    """
    torch.manual_seed(0)

    single, pair, rigids, mask = make_inputs(
        batch=2
    )

    ipa = InvariantPointAttention()

    output_batch = ipa(
        single,
        pair,
        rigids,
        mask,
    )

    output_0 = ipa(
        single[:1],
        pair[:1],
        rigids[:1],
        mask[:1],
    )

    output_1 = ipa(
        single[1:2],
        pair[1:2],
        rigids[1:2],
        mask[1:2],
    )

    torch.testing.assert_close(
        output_batch[0],
        output_0[0],
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        output_batch[1],
        output_1[0],
        rtol=1e-5,
        atol=1e-5,
    )