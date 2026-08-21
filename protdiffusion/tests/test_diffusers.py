import torch
import pytest

from protdiffusion.diffusion import Diffuser
from protdiffusion.geometry import Rigid, Rotation


# ============================================================
# Helpers
# ============================================================

def random_rotation(batch_size, length):
    """
    Create valid random Rotation objects.

    Returns:
        Rotation
            matrix: (B, L, 3, 3)
    """

    omega = torch.randn(batch_size, length, 3)

    K = torch.zeros(
        batch_size,
        length,
        3,
        3,
    )

    K[..., 0, 1] = -omega[..., 2]
    K[..., 0, 2] =  omega[..., 1]

    K[..., 1, 0] =  omega[..., 2]
    K[..., 1, 2] = -omega[..., 0]

    K[..., 2, 0] = -omega[..., 1]
    K[..., 2, 1] =  omega[..., 0]

    matrix = torch.matrix_exp(K)

    return Rotation(matrix)


def random_rigid(batch_size, length):
    """
    Create a random Rigid object.
    """

    rotation = random_rotation(
        batch_size,
        length,
    )

    translation = torch.randn(
        batch_size,
        length,
        3,
    )

    return Rigid(
        rotation=rotation,
        translation=translation,
    )


def assert_is_so3(rotation, atol=1e-5):
    """
    Assert that a Rotation object contains matrices in SO(3).
    """

    matrix = rotation.matrix

    I = torch.eye(
        3,
        device=matrix.device,
        dtype=matrix.dtype,
    )

    RtR = (
        matrix.transpose(-1, -2)
        @ matrix
    )

    determinant = torch.linalg.det(matrix)

    assert torch.allclose(
        RtR,
        I,
        atol=atol,
    )

    assert torch.allclose(
        determinant,
        torch.ones_like(determinant),
        atol=atol,
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def diffuser():
    return Diffuser(
        num_timesteps=200,
    )


@pytest.fixture
def rigid():
    return random_rigid(
        batch_size=4,
        length=100,
    )


# ============================================================
# Diffuser Tests
# ============================================================

def test_diffuser_returns_rigid_objects(
    diffuser,
    rigid,
):

    timestep = torch.randint(
        0,
        diffuser.num_timesteps,
        (4,),
    )

    xt, noise = diffuser(
        rigid,
        timestep,
    )

    assert isinstance(xt, Rigid)
    assert isinstance(noise, Rigid)


def test_diffuser_preserves_shapes(
    diffuser,
    rigid,
):

    timestep = torch.randint(
        0,
        diffuser.num_timesteps,
        (4,),
    )

    xt, noise = diffuser(
        rigid,
        timestep,
    )

    # xt contains a Rotation object.
    assert xt.rotation.matrix.shape == (
        4,
        100,
        3,
        3,
    )

    assert xt.translation.shape == (
        4,
        100,
        3,
    )

    # Rotation noise is represented as a 3D vector.
    assert noise.rotation_vector.vector.shape == (
        4,
        100,
        3,
    )

    assert noise.translation.shape == (
        4,
        100,
        3,
    )


def test_noised_rotation_is_so3(
    diffuser,
    rigid,
):

    timestep = torch.randint(
        0,
        diffuser.num_timesteps,
        (4,),
    )

    xt, _ = diffuser(
        rigid,
        timestep,
    )

    assert_is_so3(
        xt.rotation,
    )


def test_input_rotation_is_so3(rigid):

    assert_is_so3(
        rigid.rotation,
    )


def test_rotation_diffuser_preserves_so3(
    diffuser,
    rigid,
):

    timestep = torch.randint(
        0,
        diffuser.num_timesteps,
        (4,),
    )

    # Pass the Rotation object, not rigid.rotation.matrix.
    Rt, _ = diffuser.rotation_diffuser(
        rigid.rotation,
        timestep,
    )

    assert isinstance(
        Rt,
        Rotation,
    )

    assert_is_so3(
        Rt,
    )


def test_zero_rotation_noise_preserves_rotation(
    diffuser,
    rigid,
):

    timestep = torch.tensor(
        [0, 50, 100, 199],
    )

    zero_noise = torch.zeros_like(
        rigid.translation,
    )

    Rt, returned_noise = diffuser.rotation_diffuser(
        rigid.rotation,
        timestep,
        noise=zero_noise,
    )

    assert torch.equal(
        returned_noise.vector,
        zero_noise,
    )

    assert torch.allclose(
        Rt.matrix,
        rigid.rotation.matrix,
        atol=1e-5,
    )


def test_translation_forward_equation(
    diffuser,
    rigid,
):

    timestep = torch.tensor(
        [0, 50, 100, 199],
    )

    explicit_noise = torch.randn_like(
        rigid.translation,
    )

    xt, returned_noise = diffuser.translation_diffuser(
        rigid.translation,
        timestep,
        noise=explicit_noise,
    )

    alpha_bar = (
        diffuser.translation_diffuser.alpha_bar[timestep]
    )

    while alpha_bar.ndim < rigid.translation.ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)

    expected = (
        alpha_bar.sqrt()
        * rigid.translation
        +
        (1.0 - alpha_bar).sqrt()
        * explicit_noise
    )

    assert torch.equal(
        returned_noise,
        explicit_noise,
    )

    assert torch.allclose(
        xt,
        expected,
    )