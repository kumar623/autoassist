"""Fixtures every test file gets.

The caches are module-level and the whole suite runs in one process, so without
this a test can be "passed" by work an earlier test did. That is not a test of
anything, and it is not hypothetical: adding the answer cache turned six router
tests green for the wrong reason, because several of them ask "what does P0420
mean" and the second one onwards was answered from the first one's reply without
ever reaching the (faked) agents.

Cleared before and after each test: before, so a test never inherits; after, so
a failing test does not leave a surprise for the next one.
"""

import pytest

from services.orchestrator import retrieval, router


@pytest.fixture(autouse=True)
def empty_caches():
    router.ANSWERS.clear()
    retrieval._SEARCHES.clear()
    yield
    router.ANSWERS.clear()
    retrieval._SEARCHES.clear()
