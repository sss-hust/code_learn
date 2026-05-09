"""02_indexing_slicing 自动测试。"""
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


def test_select_rows(mod):
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    out = mod.select_rows(x, torch.tensor([0, 2, 4]))
    expected = torch.stack([x[0], x[2], x[4]], dim=0)
    assert torch.equal(out, expected)


def test_top_k_values_per_row(mod):
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 0.0, 9.0, 6.0]])
    out = mod.top_k_values_per_row(x, 2)
    assert tuple(out.shape) == (2, 2)
    assert torch.equal(out[0], torch.tensor([5.0, 3.0]))
    assert torch.equal(out[1], torch.tensor([9.0, 6.0]))


def test_mask_below(mod):
    x = torch.tensor([-1.0, 0.5, -2.0, 3.0])
    out = mod.mask_below(x, 0.0, -99.0)
    assert torch.equal(out, torch.tensor([-99.0, 0.5, -99.0, 3.0]))


def test_gather_per_row(mod):
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    idx = torch.tensor([[3, 0], [1, 2]])
    out = mod.gather_per_row(x, idx)
    expected = torch.tensor([[40.0, 10.0], [2.0, 3.0]])
    assert torch.equal(out, expected)


def test_where_positive_else(mod):
    x = torch.tensor([-1.0, 0.5, -2.0, 3.0])
    fb = torch.tensor([100.0, 200.0, 300.0, 400.0])
    out = mod.where_positive_else(x, fb)
    assert torch.equal(out, torch.tensor([100.0, 0.5, 300.0, 3.0]))
