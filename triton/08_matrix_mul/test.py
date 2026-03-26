"""08_matrix_mul 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


@pytest.fixture
def matmul_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import matmul
    else:
        from exercise import matmul
    return matmul


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128), (256, 256, 256), (512, 512, 512),
    (64, 128, 256), (333, 444, 555),
])
def test_correctness(matmul_fn, M, N, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    output = matmul_fn(a, b)
    expected = a @ b
    
    # FP16 矩阵乘法允许较大容差
    assert torch.allclose(output, expected, atol=1.0, rtol=1e-2), \
        f"({M},{N},{K}) 最大误差: {(output - expected).abs().max().item()}"


def test_shape(matmul_fn):
    a = torch.randn(64, 128, device='cuda', dtype=torch.float16)
    b = torch.randn(128, 256, device='cuda', dtype=torch.float16)
    c = matmul_fn(a, b)
    assert c.shape == (64, 256)


def test_identity(matmul_fn):
    """与单位矩阵相乘应返回原矩阵"""
    M, K = 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    eye = torch.eye(K, device='cuda', dtype=torch.float16)
    c = matmul_fn(a, eye)
    assert torch.allclose(c, a, atol=1e-2)
