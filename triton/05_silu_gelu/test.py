"""05_silu_gelu 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def activation_fns(request):
    if request.config.getoption("--check-solution"):
        from solution import silu, gelu
    else:
        from exercise import silu, gelu
    return silu, gelu


@pytest.mark.parametrize("size", [1024, 10000, 131072])
def test_silu(activation_fns, size):
    silu_fn, _ = activation_fns
    torch.manual_seed(42)
    x = torch.randn(size, device='cuda')
    output = silu_fn(x)
    expected = torch.nn.functional.silu(x)
    assert torch.allclose(output, expected, atol=1e-4), \
        f"SiLU size={size} 最大误差: {(output - expected).abs().max().item()}"


@pytest.mark.parametrize("size", [1024, 10000, 131072])
def test_gelu(activation_fns, size):
    _, gelu_fn = activation_fns
    torch.manual_seed(42)
    x = torch.randn(size, device='cuda')
    output = gelu_fn(x)
    expected = torch.nn.functional.gelu(x, approximate='tanh')
    assert torch.allclose(output, expected, atol=1e-4), \
        f"GELU size={size} 最大误差: {(output - expected).abs().max().item()}"


def test_silu_zero(activation_fns):
    """SiLU(0) = 0"""
    silu_fn, _ = activation_fns
    x = torch.zeros(128, device='cuda')
    output = silu_fn(x)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


def test_gelu_symmetry(activation_fns):
    """GELU 不是对称函数，但 GELU(0) ≈ 0"""
    _, gelu_fn = activation_fns
    x = torch.zeros(128, device='cuda')
    output = gelu_fn(x)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
