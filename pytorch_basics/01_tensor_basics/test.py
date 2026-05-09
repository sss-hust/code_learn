"""01_tensor_basics 自动测试。

默认测试 interview.py；加 --check-solution 改测 solution.py。
"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--check-solution",
        action="store_true",
        default=False,
        help="跑参考答案而不是 interview.py",
    )


@pytest.fixture
def mod(request):
    if request.config.getoption("--check-solution"):
        import solution as m
    else:
        import interview as m
    return m


def test_make_arange_2d_shape_and_values(mod):
    x = mod.make_arange_2d(3, 4)
    assert tuple(x.shape) == (3, 4)
    assert torch.allclose(x.flatten().to(torch.float64), torch.arange(12, dtype=torch.float64))


def test_make_arange_2d_dtype(mod):
    x = mod.make_arange_2d(2, 2, dtype=torch.int64)
    assert x.dtype == torch.int64


def test_as_dtype(mod):
    x = torch.randn(3, 3)
    y = mod.as_dtype(x, torch.float64)
    assert y.dtype == torch.float64
    assert torch.allclose(y.float(), x, atol=1e-6)


def test_tensor_info(mod):
    info = mod.tensor_info(torch.zeros(2, 5))
    assert info["shape"] == (2, 5)
    assert info["numel"] == 10
    assert info["ndim"] == 2
    assert "float32" in info["dtype"].lower() or "float" in info["dtype"].lower()


def test_safe_to_device_fallback(mod):
    x = torch.zeros(4)
    y = mod.safe_to_device(x, "cuda")
    assert isinstance(y, torch.Tensor)
    assert torch.cuda.is_available() or y.device.type == "cpu"
