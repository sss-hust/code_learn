"""08_nn_module_basics 自动测试。"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def mod(request):
    if request.config.getoption("--check-solution"):
        import solution as m
    else:
        import interview as m
    return m


def test_linear_no_bias_shape(mod):
    layer = mod.LinearNoBias(8, 4)
    x = torch.randn(2, 8)
    out = layer(x)
    assert tuple(out.shape) == (2, 4)


def test_linear_no_bias_weight_is_parameter(mod):
    layer = mod.LinearNoBias(8, 4)
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert tuple(layer.weight.shape) == (4, 8)


def test_linear_no_bias_buffer_registered(mod):
    layer = mod.LinearNoBias(8, 4)
    assert "forward_count" in dict(layer.named_buffers())
    fc = layer.forward_count
    assert fc.dtype == torch.long


def test_forward_count_increments(mod):
    layer = mod.LinearNoBias(8, 4)
    x = torch.randn(2, 8)
    for _ in range(3):
        layer(x)
    assert int(layer.forward_count.item()) == 3


def test_count_parameters(mod):
    layer = mod.LinearNoBias(8, 4)
    assert mod.count_parameters(layer) == 4 * 8


def test_freeze_module(mod):
    layer = mod.LinearNoBias(8, 4)
    mod.freeze_module(layer)
    assert mod.count_parameters(layer) == 0
    for p in layer.parameters():
        assert p.requires_grad is False
