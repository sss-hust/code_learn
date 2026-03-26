"""03_layer_norm 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def layer_norm_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import layer_norm
    else:
        from exercise import layer_norm
    return layer_norm


@pytest.mark.parametrize("shape", [(128, 512), (1, 256), (64, 1024), (32, 333)])
def test_correctness(layer_norm_fn, shape):
    """测试不同形状的 LayerNorm 正确性"""
    torch.manual_seed(42)
    n_rows, n_cols = shape
    x = torch.randn(*shape, device='cuda')
    weight = torch.randn(n_cols, device='cuda')
    bias = torch.randn(n_cols, device='cuda')
    
    output = layer_norm_fn(x, weight, bias)
    expected = torch.nn.functional.layer_norm(x, (n_cols,), weight, bias)
    
    assert torch.allclose(output, expected, atol=1e-4), \
        f"shape={shape} 时最大误差: {(output - expected).abs().max().item()}"


def test_identity_params(layer_norm_fn):
    """weight=1, bias=0 时应等价于简单归一化"""
    x = torch.randn(64, 128, device='cuda')
    weight = torch.ones(128, device='cuda')
    bias = torch.zeros(128, device='cuda')
    
    output = layer_norm_fn(x, weight, bias)
    expected = torch.nn.functional.layer_norm(x, (128,), weight, bias)
    assert torch.allclose(output, expected, atol=1e-5)


def test_zero_mean(layer_norm_fn):
    """归一化后每行均值应接近 0（weight=1, bias=0 时）"""
    x = torch.randn(64, 128, device='cuda')
    weight = torch.ones(128, device='cuda')
    bias = torch.zeros(128, device='cuda')
    
    output = layer_norm_fn(x, weight, bias)
    row_means = output.mean(dim=-1)
    assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-4)
