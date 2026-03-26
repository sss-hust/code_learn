"""10_flash_attention 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def attention_ref(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@pytest.fixture
def flash_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import flash_attention
    else:
        from exercise import flash_attention
    return flash_attention


@pytest.mark.parametrize("batch,heads,seq_len,head_dim", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (2, 4, 256, 64),
    (1, 2, 128, 128),
])
def test_correctness(flash_fn, batch, heads, seq_len, head_dim):
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    output = flash_fn(q, k, v)
    expected = attention_ref(q, k, v)
    
    # FP16 + 在线 softmax 允许较大容差
    assert torch.allclose(output, expected, atol=0.5, rtol=0.05), \
        f"({batch},{heads},{seq_len},{head_dim}) 最大误差: {(output - expected).abs().max().item()}"


def test_output_shape(flash_fn):
    q = torch.randn(1, 2, 64, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 2, 64, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 2, 64, 64, device='cuda', dtype=torch.float16)
    output = flash_fn(q, k, v)
    assert output.shape == q.shape
