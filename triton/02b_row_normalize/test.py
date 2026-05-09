"""02b_row_normalize 自动测试"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def row_normalize_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import row_normalize
    else:
        from exercise import row_normalize
    return row_normalize


@pytest.mark.parametrize("shape", [(8, 64), (64, 200), (32, 1024), (16, 333)])
def test_correctness(row_normalize_fn, shape):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    torch.manual_seed(0)
    x = torch.rand(*shape, device="cuda") + 1e-3
    out = row_normalize_fn(x)
    ref = x / x.sum(dim=-1, keepdim=True)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"shape={shape} max_err={(out-ref).abs().max().item()}"


def test_rows_sum_to_one(row_normalize_fn):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    torch.manual_seed(0)
    x = torch.rand(16, 100, device="cuda") + 1e-3
    out = row_normalize_fn(x)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
