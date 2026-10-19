"""Keep the vendored `intuit.py`'s recursion limit inside the ndbench tests.

The vendored take-home `intuit.py` (`code/ndbench/vendor/takehome/`, frozen bytes) calls
`sys.setrecursionlimit(10000)` when it is imported, for the whole interpreter.
`logic.skills.classical` imports it lazily, at its first call, and `test_wp0_vendor` imports it
directly. In one pytest process the raised limit then stayed set for every later test. That flipped
tests elsewhere which assume Python's default limit of 1000, such as
`test_state_env.py::test_deep_nesting_never_raises`.

- Every ndbench test now gets the recursion limit put back after it runs (autouse), wherever the
  first import happens.
- The tests that reach `intuit_provable` run at intuit's own limit (`intuit_recursion_limit`, used
  by `test_wp2_skills` and `test_wp2_existing`), as they would in an ndbench process that has
  imported it.

Non-test code is unchanged: an ndbench run that calls `classical` still raises the limit for its
whole process.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

#: The limit `intuit.py` sets at import.
INTUIT_RECURSION_LIMIT = 10_000


@pytest.fixture(autouse=True)
def _restore_recursion_limit() -> Iterator[None]:
    before = sys.getrecursionlimit()
    try:
        yield
    finally:
        sys.setrecursionlimit(before)


@pytest.fixture
def intuit_recursion_limit() -> Iterator[None]:
    """Run the test at the recursion limit `intuit.py` sets at import; restore the old one."""
    before = sys.getrecursionlimit()
    sys.setrecursionlimit(max(before, INTUIT_RECURSION_LIMIT))
    try:
        yield
    finally:
        sys.setrecursionlimit(before)
