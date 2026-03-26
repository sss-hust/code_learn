"""01_vector_add CUDA 自动测试

通过 subprocess 编译并运行 CUDA 代码，检查输出是否包含"通过"。
"""

import pytest
import subprocess
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def compile_and_run(source_file):
    """编译并运行 CUDA 源文件"""
    src = os.path.join(DIR, source_file)
    exe = os.path.join(DIR, "test_output")
    
    # 编译
    result = subprocess.run(
        ["nvcc", "-O2", "-o", exe, src],
        capture_output=True, text=True, cwd=DIR
    )
    if result.returncode != 0:
        pytest.fail(f"编译失败:\n{result.stderr}")
    
    # 运行
    result = subprocess.run(
        [exe], capture_output=True, text=True, cwd=DIR
    )
    
    # 清理
    if os.path.exists(exe):
        os.remove(exe)
    
    return result.stdout


def test_vector_add(request):
    source = "solution.cu" if request.config.getoption("--check-solution") else "exercise.cu"
    output = compile_and_run(source)
    print(output)
    assert "通过" in output, f"测试未通过，输出:\n{output}"
