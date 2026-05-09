"""07_autograd_basics 自动测试。"""
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


def test_numerical_grad_polynomial(mod):
    fn = lambda x: x ** 3 + 2 * x
    g = mod.numerical_grad(fn, 1.5)
    expected = 3 * 1.5 ** 2 + 2.0
    assert abs(g - expected) < 1e-3


def test_autograd_grad_polynomial(mod):
    fn = lambda x: x ** 3 + 2 * x
    g = mod.autograd_grad(fn, 1.5)
    expected = 3 * 1.5 ** 2 + 2.0
    assert abs(g - expected) < 1e-5


def test_autograd_matches_numerical(mod):
    fn = lambda x: torch.sin(x) * x + x ** 2
    for x_value in [-1.0, 0.0, 0.5, 2.0]:
        g_auto = mod.autograd_grad(fn, x_value)
        g_num = mod.numerical_grad(fn, x_value)
        assert abs(g_auto - g_num) < 1e-3, f"x={x_value}: auto={g_auto}, num={g_num}"


def test_manual_sgd_converges(mod):
    param = torch.tensor(0.0, requires_grad=True)
    target = torch.tensor(3.0)
    loss_fn = lambda p: (p - target) ** 2
    for _ in range(100):
        mod.manual_sgd_step(param, loss_fn, lr=0.1)
    assert abs(param.item() - 3.0) < 1e-2


def test_detach_returns_true(mod):
    x = torch.randn(4, requires_grad=True)
    assert mod.detached_does_not_track_grad(x) is True
