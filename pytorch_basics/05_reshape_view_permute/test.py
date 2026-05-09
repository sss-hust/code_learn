"""05_reshape_view_permute 自动测试。"""
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


def test_flatten_btc_to_nc(mod):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = mod.flatten_btc_to_nc(x)
    assert tuple(out.shape) == (6, 4)
    assert torch.equal(out[0], x[0, 0])
    assert torch.equal(out[3], x[1, 0])


def test_round_trip(mod):
    x = torch.randn(2, 3, 4)
    flat = mod.flatten_btc_to_nc(x)
    back = mod.unflatten_nc_to_btc(flat, batch_size=2)
    assert tuple(back.shape) == tuple(x.shape)
    assert torch.equal(back, x)


def test_swap_last_two(mod):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = mod.swap_last_two(x)
    assert tuple(out.shape) == (2, 4, 3)
    for i in range(2):
        assert torch.equal(out[i], x[i].t())


def test_nhwc_to_nchw_shape_and_values(mod):
    x = torch.randn(1, 2, 2, 3)
    out = mod.nhwc_to_nchw(x)
    assert tuple(out.shape) == (1, 3, 2, 2)
    assert out.is_contiguous(), "nhwc_to_nchw 必须返回 contiguous 张量"
    expected = x.permute(0, 3, 1, 2).contiguous()
    assert torch.equal(out, expected)
