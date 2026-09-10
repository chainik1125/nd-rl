"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import pytest
from pytest import Config, Item, Parser

CURRENT_EXPERIMENTS = Path(__file__).resolve().parent / "experiments" / "current"
R1_VENDOR = CURRENT_EXPERIMENTS / "r1_cot" / "vendor"
sys.path[:0] = [str(CURRENT_EXPERIMENTS), str(R1_VENDOR)]


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Config, items: Iterable[Item]) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
