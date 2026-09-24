import pytest
import torch

from antstorch.registration import RegAdamState, reg_adam_direction


def test_first_reg_adam_direction_is_sign_of_gradient():
    gradient = torch.tensor([-4.0, 0.0, 2.0], dtype=torch.double)

    state, direction = reg_adam_direction(gradient, eps=1e-12)

    assert state.step == 1
    torch.testing.assert_close(direction, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double))


def test_reg_adam_direction_reuses_and_updates_state():
    first = torch.tensor([1.0, -2.0])
    second = torch.tensor([3.0, 4.0])

    state, first_direction = reg_adam_direction(first)
    same_state, second_direction = reg_adam_direction(second, state)

    assert same_state is state
    assert state.step == 2
    assert torch.isfinite(first_direction).all()
    assert torch.isfinite(second_direction).all()
    assert not torch.equal(first_direction, second_direction)


def test_reg_adam_direction_rejects_incompatible_state():
    state = RegAdamState.zeros_like(torch.zeros(2))

    with pytest.raises(ValueError, match="same shape"):
        reg_adam_direction(torch.zeros(3), state)


@pytest.mark.parametrize("betas", [(-0.1, 0.9), (0.9, 1.0)])
def test_reg_adam_direction_validates_betas(betas):
    with pytest.raises(ValueError, match="betas"):
        reg_adam_direction(torch.ones(2), betas=betas)
