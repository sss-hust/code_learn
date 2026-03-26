"""06_rope 自动测试"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def precompute_freqs(head_dim, seq_len, base=10000.0, device='cuda'):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def rope_ref(x, cos, sin):
    x0 = x[..., ::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack([out0, out1], dim=-1).flatten(-2)


@pytest.fixture
def rope_fn(request):
    if request.config.getoption("--check-solution"):
        from solution import rope
    else:
        from exercise import rope
    return rope


@pytest.mark.parametrize("batch,seq_len,head_dim", [
    (1, 32, 64), (2, 64, 128), (4, 128, 256)
])
def test_correctness(rope_fn, batch, seq_len, head_dim):
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, head_dim, device='cuda')
    cos, sin = precompute_freqs(head_dim, seq_len)
    
    output = rope_fn(x, cos, sin)
    expected = rope_ref(x, cos, sin)
    
    assert torch.allclose(output, expected, atol=1e-4), \
        f"({batch},{seq_len},{head_dim}) 最大误差: {(output - expected).abs().max().item()}"


def test_position_zero(rope_fn):
    """位置 0 时 cos=1, sin=0，输出应与输入相同"""
    head_dim = 64
    x = torch.randn(1, 1, head_dim, device='cuda')
    cos, sin = precompute_freqs(head_dim, 1)
    
    output = rope_fn(x, cos, sin)
    # 位置 0: cos(0)=1, sin(0)=0, 所以 output = x
    assert torch.allclose(output, x, atol=1e-5), "位置 0 时输出应等于输入"
