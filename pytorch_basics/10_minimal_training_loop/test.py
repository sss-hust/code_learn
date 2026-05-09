"""10_minimal_training_loop 自动测试。"""
import pytest
import torch
import torch.nn as nn


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def mod(request):
    if request.config.getoption("--check-solution"):
        import solution as m
    else:
        import interview as m
    return m


def test_make_simple_mlp_structure(mod):
    model = mod.make_simple_mlp(4, 16, 1)
    x = torch.randn(3, 4)
    out = model(x)
    assert tuple(out.shape) == (3, 1)
    children = list(model.children())
    assert len(children) >= 3, "至少要有 Linear、激活、Linear 三层"


def test_train_step_decreases_loss(mod):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    x = torch.randn(20, 4)
    y = torch.randn(20, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    l0 = mod.train_step(model, x, y, loss_fn, optimizer)
    for _ in range(20):
        l_last = mod.train_step(model, x, y, loss_fn, optimizer)
    assert l_last < l0, f"训练 20 步后 loss 应下降：{l0} -> {l_last}"


def test_fit_regression_converges(mod):
    torch.manual_seed(0)
    X = torch.randn(200, 4)
    W_true = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    b_true = torch.tensor([0.5])
    y = X @ W_true + b_true + 0.1 * torch.randn(200, 1)

    model = mod.make_simple_mlp(4, 16, 1)
    losses = mod.fit_regression(model, X, y, epochs=200, lr=0.05)
    assert len(losses) == 200
    assert losses[-1] < 0.5 * losses[0], (
        f"训练 200 epoch 后 loss 应至少减半：{losses[0]} -> {losses[-1]}"
    )
