"""01b_axpy 自动测试"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def axpy_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import axpy
    else:
        from exercise import axpy
    return axpy


@pytest.mark.parametrize("size", [1024, 98432, 100000])
@pytest.mark.parametrize("a,b", [(1.0, 0.0), (2.5, -0.5), (-1.0, 7.0)])
def test_correctness(axpy_fn, size, a, b):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    torch.manual_seed(0)
    x = torch.rand(size, device="cuda")
    out = axpy_fn(x, a, b)
    ref = a * x + b
    assert torch.allclose(out, ref, atol=1e-5), \
        f"size={size} a={a} b={b} 最大误差 {(out-ref).abs().max().item()}"
