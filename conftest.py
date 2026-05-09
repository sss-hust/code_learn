"""code_learn 仓库根级 pytest 配置。

注意：pytest 的 ``pytest_addoption`` hook 只在 ``conftest.py`` 里有效，
写在 ``test.py`` 里相当于没写。本文件统一注册 ``--check-solution``，
任意子目录下的 ``test.py`` 都能用 ``request.config.getoption('--check-solution')``。
"""


def pytest_addoption(parser):
    parser.addoption(
        "--check-solution",
        action="store_true",
        default=False,
        help="跑 solution.* 而不是 interview.* / exercise.*",
    )
