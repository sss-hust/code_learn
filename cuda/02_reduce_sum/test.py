"""02_reduce_sum CUDA 自动测试"""

import pytest
import subprocess
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def pytest_addoption(parser):
    parser.addoption("--check-solution", action="store_true", default=False)


def compile_and_run(source_file):
    src = os.path.join(DIR, source_file)
    exe = os.path.join(DIR, "test_output")
    result = subprocess.run(["nvcc", "-O2", "-o", exe, src],
                            capture_output=True, text=True, cwd=DIR)
    if result.returncode != 0:
        pytest.fail(f"编译失败:\n{result.stderr}")
    result = subprocess.run([exe], capture_output=True, text=True, cwd=DIR)
    if os.path.exists(exe):
        os.remove(exe)
    return result.stdout


def test_reduce_sum(request):
    source = "solution.cu" if request.config.getoption("--check-solution") else "exercise.cu"
    output = compile_and_run(source)
    print(output)
    assert "通过" in output, f"测试未通过，输出:\n{output}"
