"""Tests for retrieval over plain HTTPS: what is sent to Azure OpenAI and AI
Search, and the relevance floor and per-source cap applied to what comes back.
The expected requests match the SDKs' recorded traffic."""

import json

import httpx
import pytest

from services.orchestrator import retrieval

VECTOR = [0.01] * 1536


def hit(i, score, source="TSB-015.pdf", doc_type="bulletin"):
    return {"@search.score": score, "id": f"c{i}", "title": f"t{i}", "content": f"text {i}",
            "doc_type": doc_type, "source_file": source, "section": "SUMMARY", "page": 1, "severity": "medium"}


@pytest.fixture
def azure(monkeypatch):
    """Fake Azure OpenAI and AI Search. Set .results to what the search returns."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://rg-autoassist.services.ai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "oai-key")
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://autoassist-search.search.windows.net")
    monkeypatch.setenv("SEARCH_API_KEY", "search-key")

    state = {"results": [], "seen": []}

    def handler(request):
        state["seen"].append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": VECTOR, "index": 0}]})
        return httpx.Response(200, json={"value": state["results"]})

    monkeypatch.setattr(retrieval, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    retrieval._SEARCHES.clear()
    return state


def test_the_question_is_embedded_then_searched_both_ways(azure):
    azure["results"] = [hit(1, 0.03)]
    retrieval.search("P0420")
    emb, srch = azure["seen"]

    assert emb.url.path == "/openai/deployments/text-embedding-3-small/embeddings"
    assert emb.headers["api-key"] == "oai-key"
    assert json.loads(emb.content) == {"input": ["P0420"]}

    assert srch.url.path == "/indexes('service-docs')/docs/search.post.search"
    assert srch.url.params["api-version"] == "2026-04-01"
    assert srch.headers["api-key"] == "search-key"
    sent = json.loads(srch.content)
    assert sent["search"] == "P0420", "the keyword half"
    assert sent["vectorQueries"] == [{"kind": "vector", "vector": VECTOR, "k": 20, "fields": "content_vector"}]
    assert sent["top"] == 20, "4x wider than the 5 we keep"
    assert "filter" not in sent


def test_a_valid_doc_type_becomes_a_filter(azure):
    retrieval.search("P0420", doc_type="dtc")
    assert json.loads(azure["seen"][1].content)["filter"] == "doc_type eq 'dtc'"


@pytest.mark.parametrize("bad", ["x' or doc_type ne 'x", "DTC", "manual"])
def test_any_other_doc_type_is_refused_before_anything_is_sent(azure, bad):
    """The model writes this value, and it would go into the filter as written."""
    with pytest.raises(ValueError):
        retrieval.search("P0420", doc_type=bad)
    assert azure["seen"] == []


def test_the_tool_turns_a_refused_filter_into_an_error_the_agent_can_read(azure):
    from services.orchestrator import tools
    out = tools.execute("search_service_docs", '{"query": "P0420", "doc_type": "manual"}')
    assert out.startswith("ERROR:") and "doc_type must be one of" in out


def test_results_below_the_relevance_floor_are_dropped(azure):
    azure["results"] = [hit(1, 0.030, "a.pdf"), hit(2, 0.010, "b.pdf")]
    r = retrieval.search("q")
    assert [c.id for c in r.chunks] == ["c1"]
    assert r.dropped_below_floor == 1


def test_no_file_contributes_more_than_the_cap(azure):
    azure["results"] = [hit(i, 0.03) for i in range(5)] + [hit(9, 0.02, "TSB-001.pdf")]
    r = retrieval.search("q")
    assert [c.source_file for c in r.chunks] == ["TSB-015.pdf", "TSB-015.pdf", "TSB-001.pdf"]
    assert r.dropped_by_source_cap == 3


def test_nothing_relevant_means_nothing_returned(azure):
    azure["results"] = [hit(1, 0.005)]
    r = retrieval.search("P9999")
    assert not r.found_anything
    assert retrieval.format_for_agent(r).startswith("NOTHING USABLE")


def test_the_closing_block_names_the_safety_systems_and_the_bulletin_trap(azure):
    """The last thing the agent reads before writing is this block, and that
    position is worth using.

    'If the question involves a safety system' on its own did not reach the
    case it is for: asked about a smell of petrol, the agent found the
    hard-starting bulletin, wrote what it said, and left the warning out on 1
    run in 5 (finding 13). Naming the systems, and saying that a bulletin
    describing the symptom does not settle it, closed that.
    """
    azure["results"] = [hit(1, 0.03)]
    out = retrieval.format_for_agent(retrieval.search("smell of petrol"))
    tail = out.split("THIS TOOL ONLY TELLS YOU")[1]
    for system in ("brakes", "steering", "airbags", "seat belts", "smell of petrol"):
        assert system in tail, system
    assert "known condition" in tail


def test_fetch_all_reads_the_whole_index_without_a_vector(azure):
    azure["results"] = [hit(1, 1.0)]
    assert len(retrieval.fetch_all(["doc_type", "content"])) == 1
    [req] = azure["seen"]
    assert json.loads(req.content) == {"search": "*", "select": "doc_type,content", "top": 1000}


def test_the_same_question_is_searched_once(azure):
    """An agent often searches the same thing twice in one conversation."""
    azure["results"] = [hit(1, 0.03)]
    first = retrieval.search("P0420")
    second = retrieval.search(" p0420 ")
    assert second is first
    assert len(azure["seen"]) == 2, "one embedding, one search - not four"


def test_a_different_question_is_searched_again(azure):
    azure["results"] = [hit(1, 0.03)]
    retrieval.search("P0420")
    retrieval.search("P0300")
    assert len(azure["seen"]) == 4


def test_keys_are_sent_without_the_whitespace_a_paste_brings(azure, monkeypatch):
    """A trailing space or newline is an illegal header value: httpx refuses to
    send the request at all, and search fails on every message."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "oai-key\n")
    monkeypatch.setenv("SEARCH_API_KEY", "  search-key ")
    azure["results"] = [hit(1, 0.03)]
    retrieval.search("P0420")
    emb, srch = azure["seen"]
    assert emb.headers["api-key"] == "oai-key"
    assert srch.headers["api-key"] == "search-key"
