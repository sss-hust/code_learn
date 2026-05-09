"""03_broadcasting 自动测试。"""
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


def test_add_row_bias(mod):
    matrix = torch.zeros(3, 4)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = mod.add_row_bias(matrix, bias)
    assert tuple(out.shape) == (3, 4)
    assert torch.equal(out[0], bias)
    assert torch.equal(out[1], bias)


def test_add_col_bias(mod):
    matrix = torch.zeros(3, 4)
    bias = torch.tensor([10.0, 20.0, 30.0])
    out = mod.add_col_bias(matrix, bias)
    assert tuple(out.shape) == (3, 4)
    assert torch.equal(out[:, 0], bias)
    assert torch.equal(out[:, 3], bias)


def test_outer_add(mod):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = mod.outer_add(a, b)
    expected = a.unsqueeze(1) + b.unsqueeze(0)
    assert tuple(out.shape) == (3, 4)
    assert torch.equal(out, expected)


def test_pairwise_squared_distance(mod):
    torch.manual_seed(0)
    x = torch.randn(3, 2)
    y = torch.randn(4, 2)
    out = mod.pairwise_squared_distance(x, y)
    expected = torch.cdist(x, y).pow(2)
    assert tuple(out.shape) == (3, 4)
    assert torch.allclose(out, expected, atol=1e-5)
