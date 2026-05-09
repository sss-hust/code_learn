"""06_matmul_einsum 自动测试。"""
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


def test_mm_2d(mod):
    torch.manual_seed(0)
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    out = mod.mm_2d(a, b)
    assert tuple(out.shape) == (3, 5)
    assert torch.allclose(out, a @ b, atol=1e-5)


def test_bmm_3d(mod):
    torch.manual_seed(0)
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 5)
    out = mod.bmm_3d(a, b)
    assert tuple(out.shape) == (2, 3, 5)
    assert torch.allclose(out, torch.bmm(a, b), atol=1e-5)


def test_attention_scores_two_ways_agree(mod):
    torch.manual_seed(0)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 5, 8)
    s_mm = mod.attention_scores_matmul(q, k)
    s_ein = mod.attention_scores_einsum(q, k)
    assert tuple(s_mm.shape) == (2, 3, 5)
    assert torch.allclose(s_mm, s_ein, atol=1e-5)


def test_weighted_combine(mod):
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(2, 3, 5), dim=-1)
    values = torch.randn(2, 5, 8)
    out = mod.weighted_combine(weights, values)
    expected = torch.einsum("bts,bsd->btd", weights, values)
    assert tuple(out.shape) == (2, 3, 8)
    assert torch.allclose(out, expected, atol=1e-5)
