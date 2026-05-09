"""04_reduction_ops 自动测试。"""
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


def test_row_mean(mod):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = mod.row_mean(x)
    assert torch.allclose(out, torch.tensor([2.0, 5.0]))


def test_argmax_per_row(mod):
    x = torch.tensor([[1.0, 9.0, 3.0], [4.0, 0.0, -1.0]])
    out = mod.argmax_per_row(x)
    assert torch.equal(out, torch.tensor([1, 0]))


def test_softmax_along_basic(mod):
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    out = mod.softmax_along(x, dim=-1)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_softmax_along_handles_large_values(mod):
    x = torch.tensor([[1000.0, 1001.0, 999.0]])
    out = mod.softmax_along(x, dim=-1)
    assert torch.isfinite(out).all(), "数值不稳定：大数让 exp 溢出了，需要先减 max"
    assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)


def test_logsumexp_stable_basic(mod):
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    out = mod.logsumexp_stable(x, dim=-1)
    ref = torch.logsumexp(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_logsumexp_stable_large_values(mod):
    x = torch.tensor([[1000.0, 1001.0, 999.0]])
    out = mod.logsumexp_stable(x, dim=-1)
    assert torch.isfinite(out).all(), "数值不稳定：需要先减 max 再 exp"
