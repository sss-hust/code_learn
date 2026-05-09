"""09_linear_by_hand 自动测试。"""
import pytest
import torch
import torch.nn as nn


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def mod(request):
    if request.config.getoption("--check-solution"):
        import solution as m
    else:
        import interview as m
    return m


def test_mylinear_shape(mod):
    layer = mod.MyLinear(8, 4)
    x = torch.randn(3, 8)
    assert tuple(layer(x).shape) == (3, 4)


def test_mylinear_has_parameters(mod):
    layer = mod.MyLinear(8, 4)
    assert isinstance(layer.weight, nn.Parameter)
    assert tuple(layer.weight.shape) == (4, 8)
    assert isinstance(layer.bias, nn.Parameter)
    assert tuple(layer.bias.shape) == (4,)


def test_mylinear_no_bias(mod):
    layer = mod.MyLinear(8, 4, bias=False)
    assert layer.bias is None


def test_mylinear_matches_nn_linear_when_weights_copied(mod):
    torch.manual_seed(0)
    my = mod.MyLinear(8, 4)
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        my.weight.copy_(ref.weight)
        my.bias.copy_(ref.bias)
    x = torch.randn(5, 8)
    assert torch.allclose(my(x), ref(x), atol=1e-6)
