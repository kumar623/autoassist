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

# Everything that would let a test reach a real service. app.py calls
# load_dotenv() when it is imported, so on a machine with a filled-in .env these
# are in the environment for the whole run - and a test that forgot to stub the
# pre-search was making real embedding and search calls. Six did, on every run
# on a laptop, while the README said the suite needs no Azure. Removed from each
# test's environment; a test that wants one sets it with monkeypatch.
LIVE_SETTINGS = (
    "PROJECT_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "SEARCH_ENDPOINT",
    "SEARCH_API_KEY",
    "TYPESAFE_API_KEY",
    "TRIAGE_BACKEND",
    "ZOHO_MCP_URL",
    "ZOHO_MCP_CLIENT_ID",
    "ZOHO_MCP_CLIENT_SECRET",
    "ZOHO_MCP_REFRESH_TOKEN",
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
)


@pytest.fixture(autouse=True)
def no_live_services(monkeypatch):
    for name in LIVE_SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def empty_caches():
    router.ANSWERS.clear()
    retrieval._SEARCHES.clear()
    yield
    router.ANSWERS.clear()
    retrieval._SEARCHES.clear()
