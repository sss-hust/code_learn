"""09_fused_add_rmsnorm 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def ref_fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    x_new = x + residual
    rms = torch.sqrt(torch.mean(x_new ** 2, dim=-1, keepdim=True) + eps)
    return (x_new / rms) * weight, x_new


@pytest.fixture
def fused_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import fused_add_rmsnorm
    else:
        from exercise import fused_add_rmsnorm
    return fused_add_rmsnorm


@pytest.mark.parametrize("shape", [(128, 512), (1, 256), (64, 1024), (32, 333)])
def test_output_correctness(fused_fn, shape):
    torch.manual_seed(42)
    n_rows, n_cols = shape
    x = torch.randn(*shape, device='cuda')
    residual = torch.randn(*shape, device='cuda')
    weight = torch.randn(n_cols, device='cuda')
    
    expected_out, _ = ref_fused_add_rmsnorm(x.clone(), residual, weight)
    output = fused_fn(x, residual, weight)
    
    assert torch.allclose(output, expected_out, atol=1e-4), \
        f"shape={shape} 输出最大误差: {(output - expected_out).abs().max().item()}"


def test_inplace_update(fused_fn):
    """x 应被原地更新为 x + residual"""
    x = torch.randn(32, 128, device='cuda')
    residual = torch.randn(32, 128, device='cuda')
    weight = torch.ones(128, device='cuda')
    
    x_orig = x.clone()
    expected_x = x_orig + residual
    
    fused_fn(x, residual, weight)
    
    assert torch.allclose(x, expected_x, atol=1e-5), "x 未被正确原地更新"
