"""07_online_softmax 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def online_softmax_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import online_softmax
    else:
        from exercise import online_softmax
    return online_softmax


@pytest.mark.parametrize("shape", [
    (128, 1024), (1, 512), (64, 2048), (32, 333)
])
def test_correctness(online_softmax_fn, shape):
    torch.manual_seed(42)
    x = torch.randn(*shape, device='cuda')
    output = online_softmax_fn(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, atol=1e-5), \
        f"shape={shape} 最大误差: {(output - expected).abs().max().item()}"


def test_sum_to_one(online_softmax_fn):
    x = torch.randn(64, 1024, device='cuda')
    output = online_softmax_fn(x)
    row_sums = output.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_large_values(online_softmax_fn):
    """大数值下的数值稳定性"""
    x = torch.tensor([[1000.0, 1001.0, 1002.0, 999.0]], device='cuda')
    output = online_softmax_fn(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, atol=1e-5)


def test_exceeds_block_size(online_softmax_fn):
    """列数远大于 BLOCK_SIZE=256 时的正确性"""
    x = torch.randn(16, 4096, device='cuda')
    output = online_softmax_fn(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, atol=1e-5)
