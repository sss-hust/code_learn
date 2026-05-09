"""01c_vector_sum 自动测试"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def vector_sum_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import vector_sum
    else:
        from exercise import vector_sum
    return vector_sum


@pytest.mark.parametrize("size", [1024, 4096, 100000])
def test_correctness(vector_sum_fn, size):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    torch.manual_seed(0)
    x = torch.rand(size, device="cuda")
    out = vector_sum_fn(x)
    ref = x.sum()
    err = (out - ref).abs().item()
    assert err < 1e-2, f"size={size} 误差 {err}"
