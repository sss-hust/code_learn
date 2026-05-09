"""01d_row_max 自动测试"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def row_max_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import row_max
    else:
        from exercise import row_max
    return row_max


@pytest.mark.parametrize("shape", [(8, 64), (64, 200), (32, 1024), (16, 333)])
def test_correctness(row_max_fn, shape):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    out = row_max_fn(x)
    ref = x.max(dim=-1).values
    assert torch.allclose(out, ref, atol=1e-6), \
        f"shape={shape} max_err={(out-ref).abs().max().item()}"
