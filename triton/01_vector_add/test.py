"""01_vector_add 自动测试"""

import pytest
import torch


def get_module(request):
    """根据命令行参数选择测试 exercise 还是 solution"""
    if request.config.getoption("--check-solution", default=False):
        from solution import vector_add
    else:
        from exercise import vector_add
    return vector_add


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False,
                     help="测试参考答案而非练习代码")


@pytest.fixture
def vector_add_fn(request):
    return get_module(request)


@pytest.mark.parametrize("size", [1024, 98432, 131072, 100000])
def test_correctness(vector_add_fn, size):
    """测试不同大小的向量加法正确性"""
    torch.manual_seed(42)
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    output = vector_add_fn(x, y)
    expected = x + y
    
    assert torch.allclose(output, expected, atol=1e-5), \
        f"size={size} 时结果不正确，最大误差: {(output - expected).abs().max().item()}"


def test_shape(vector_add_fn):
    """测试输出形状正确"""
    x = torch.rand(256, device='cuda')
    y = torch.rand(256, device='cuda')
    output = vector_add_fn(x, y)
    assert output.shape == x.shape


def test_dtype(vector_add_fn):
    """测试不同数据类型"""
    for dtype in [torch.float32, torch.float16]:
        x = torch.rand(1024, device='cuda', dtype=dtype)
        y = torch.rand(1024, device='cuda', dtype=dtype)
        output = vector_add_fn(x, y)
        expected = x + y
        assert torch.allclose(output, expected, atol=1e-3), \
            f"dtype={dtype} 时结果不正确"
