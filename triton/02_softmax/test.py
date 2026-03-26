"""02_softmax 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def softmax_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import softmax
    else:
        from exercise import softmax
    return softmax


@pytest.mark.parametrize("shape", [(128, 256), (1, 1024), (64, 64), (32, 781)])
def test_correctness(softmax_fn, shape):
    """测试不同形状的 softmax 正确性"""
    torch.manual_seed(42)
    x = torch.randn(*shape, device='cuda')
    output = softmax_fn(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, atol=1e-5), \
        f"shape={shape} 时结果不正确，最大误差: {(output - expected).abs().max().item()}"


def test_sum_to_one(softmax_fn):
    """softmax 每行之和应为 1"""
    x = torch.randn(64, 128, device='cuda')
    output = softmax_fn(x)
    row_sums = output.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_numerical_stability(softmax_fn):
    """大数值下的数值稳定性"""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
    output = softmax_fn(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, atol=1e-5), "大数值下数值不稳定"
