"""04_rms_norm 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def rms_norm_ref(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


@pytest.fixture
def rms_norm_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import rms_norm
    else:
        from exercise import rms_norm
    return rms_norm


@pytest.mark.parametrize("shape", [(128, 512), (1, 256), (64, 1024), (32, 333)])
def test_correctness(rms_norm_fn, shape):
    torch.manual_seed(42)
    n_rows, n_cols = shape
    x = torch.randn(*shape, device='cuda')
    weight = torch.randn(n_cols, device='cuda')
    
    output = rms_norm_fn(x, weight)
    expected = rms_norm_ref(x, weight)
    
    assert torch.allclose(output, expected, atol=1e-4), \
        f"shape={shape} 时最大误差: {(output - expected).abs().max().item()}"


def test_unit_weight(rms_norm_fn):
    """weight=1 时的基本正确性"""
    x = torch.randn(64, 128, device='cuda')
    weight = torch.ones(128, device='cuda')
    
    output = rms_norm_fn(x, weight)
    expected = rms_norm_ref(x, weight)
    assert torch.allclose(output, expected, atol=1e-5)


def test_scale_invariance(rms_norm_fn):
    """RMSNorm 对输入缩放不完全不变，但输出量级应正确"""
    x = torch.randn(32, 64, device='cuda')
    weight = torch.ones(64, device='cuda')
    
    out1 = rms_norm_fn(x, weight)
    out2 = rms_norm_fn(x * 2, weight)
    
    # RMSNorm(2x) = 2x / rms(2x) = 2x / (2 * rms(x)) = x / rms(x) = RMSNorm(x)
    assert torch.allclose(out1, out2, atol=1e-4), "RMSNorm 应对输入缩放不变"
