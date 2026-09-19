"""Hybrid retrieval over the service-docs index.

This is the same search that scripts/search_test.py proved out in week 1, moved
into a reusable module and given two things the agent's built-in search tool
could not do:

  1. A relevance floor. Top-k search ALWAYS returns k results, even when nothing
     in the index is relevant. That is how the agent ended up explaining a
     clutch bulletin as a brake fault (docs/evaluation.md, finding 2). Below the
     floor we return nothing at all.

  2. A per-source cap. One strongly matching document can take every slot: for
     "my catalytic converter light is on", four of five results came from
     TSB-015 and pushed the P0420 definition to rank 4. We cap how many chunks
     any single file may contribute.

Scores here are reciprocal rank fusion sums, so they sit around 0.016 for a
chunk found by one method and 0.032 for one found by both. The absolute number
is meaningless on its own - only the ordering and the relative size matter.

Both calls are plain HTTPS (docs/decisions/007): one POST to Azure OpenAI to
embed the question, one POST to AI Search with the text and the vector. Same
URLs, API versions and bodies the SDKs sent - recorded, then reproduced.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

from . import azure_http

log = logging.getLogger(__name__)

INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "service-docs")
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")
SEARCH_API_VERSION = os.getenv("SEARCH_API_VERSION", "2026-04-01")  # what azure-search-documents 12.0.0 sent

# The only values doc_type may take. The agent's tool schema says so, but the
# model writes the value, and it goes into a search filter - so it is checked
# here rather than trusted.
DOC_TYPES = ("dtc", "maintenance", "bulletin")

RESULT_FIELDS = ["id", "title", "content", "doc_type", "source_file", "section", "page", "severity"]

# Tuning. Defaults chosen by hand in week 2; week 3 measures them properly
# against evals/golden_set.jsonl before anyone claims they are right.
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RELEVANCE_FLOOR = float(os.getenv("RETRIEVAL_FLOOR", "0.0155"))
MAX_PER_SOURCE = int(os.getenv("RETRIEVAL_MAX_PER_SOURCE", "2"))
CANDIDATE_MULTIPLIER = 4  # retrieve wider than we need, then filter and trim


@dataclass
class Chunk:
    """One retrieved piece of a document."""

    id: str
    title: str
    content: str
    doc_type: str
    source_file: str
    section: str
    page: int
    severity: str
    score: float

    def citation(self) -> str:
        """How this chunk should be referred to in an answer."""
        if self.doc_type == "dtc":
            return f"fault code list, {self.section}"
        if self.doc_type == "maintenance":
            return f"maintenance schedule, {self.section}"
        return f"{self.source_file.replace('.pdf', '')}, {self.section.title()}"

    def as_context(self) -> str:
        return f"[{self.citation()}]\n{self.content}"


@dataclass
class RetrievalResult:
    """What a search returned, plus why it returned that."""

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    candidates_seen: int = 0
    dropped_below_floor: int = 0
    dropped_by_source_cap: int = 0

    @property
    def found_anything(self) -> bool:
        return bool(self.chunks)

    def summary(self) -> str:
        return (
            f"{len(self.chunks)} kept of {self.candidates_seen} candidates "
            f"({self.dropped_below_floor} below floor, "
            f"{self.dropped_by_source_cap} over source cap)"
        )


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    """One connection pool for both APIs, reused across requests and threads."""
    return azure_http.new_client()


def embed(text: str) -> list[float]:
    """The question as a vector, from the same model that embedded the documents."""
    base = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    data = azure_http.request(
        _client(),
        "POST",
        f"{base}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings",
        headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
        params={"api-version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")},
        json={"input": [text]},
    )
    return data["data"][0]["embedding"]


def _query_index(body: dict) -> list[dict]:
    """POST a search to the index and return its results ("value")."""
    base = os.environ["SEARCH_ENDPOINT"].rstrip("/")
    data = azure_http.request(
        _client(),
        "POST",
        f"{base}/indexes('{INDEX_NAME}')/docs/search.post.search",
        headers={"api-key": os.environ["SEARCH_API_KEY"]},
        params={"api-version": SEARCH_API_VERSION},
        json=body,
    )
    return data.get("value") or []


def fetch_all(fields: list[str], top: int = 1000) -> list[dict]:
    """Every piece in the index, with the chosen fields. For the Library tab.

    One search returns at most 1000 results; library.py warns if it gets that many.
    """
    return _query_index({"search": "*", "select": ",".join(fields), "top": top})


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_type: str | None = None,
    floor: float = RELEVANCE_FLOOR,
    max_per_source: int = MAX_PER_SOURCE,
) -> RetrievalResult:
    """Hybrid search: BM25 keyword + vector similarity, fused by RRF.

    Retrieves `top_k * CANDIDATE_MULTIPLIER` candidates, drops anything below
    the relevance floor, caps how many chunks any one file may contribute, then
    returns the best `top_k` of what survives.

    Returns an empty result rather than weak matches when nothing clears the
    floor. That is the point: a wrong answer that looks sourced is worse than
    no answer.
    """
    if doc_type and doc_type not in DOC_TYPES:
        # Refused, not ignored: the agent gets an error it can read and retry
        # without the filter. Never pasted into the filter as written.
        raise ValueError(f"doc_type must be one of {', '.join(DOC_TYPES)}, or left out; got {doc_type!r}")

    result = RetrievalResult(query=query)
    want = top_k * CANDIDATE_MULTIPLIER

    body: dict = {
        "search": query,
        "top": want,
        "select": ",".join(RESULT_FIELDS),
        # Hybrid: the text above goes to BM25, this vector to HNSW, and Azure
        # merges the two rankings with reciprocal rank fusion.
        "vectorQueries": [{"kind": "vector", "vector": embed(query), "k": want, "fields": "content_vector"}],
    }
    if doc_type:
        body["filter"] = f"doc_type eq '{doc_type}'"

    per_source: dict[str, int] = {}

    for raw in _query_index(body):
        result.candidates_seen += 1
        score = float(raw["@search.score"])

        if score < floor:
            result.dropped_below_floor += 1
            continue

        source = raw["source_file"]
        if per_source.get(source, 0) >= max_per_source:
            result.dropped_by_source_cap += 1
            continue

        per_source[source] = per_source.get(source, 0) + 1
        result.chunks.append(
            Chunk(
                id=raw["id"],
                title=raw["title"],
                content=raw["content"],
                doc_type=raw["doc_type"],
                source_file=source,
                section=raw["section"],
                page=int(raw.get("page") or 0),
                severity=raw.get("severity") or "unknown",
                score=score,
            )
        )

        if len(result.chunks) >= top_k:
            break

    log.info("retrieval query=%r -> %s", query, result.summary())
    return result


def format_for_agent(result: RetrievalResult) -> str:
    """Turn a result into the string the agent receives as tool output.

    When nothing survived filtering, say so plainly and explain why. The agent
    needs to be able to tell 'the library does not cover this' apart from
    'the search failed', because those call for different answers.
    """
    if not result.found_anything:
        if result.candidates_seen == 0:
            return (
                "NO DOCUMENTS FOUND. The search returned nothing at all for "
                f"'{result.query}'. The service library does not cover this topic."
            )
        return (
            f"NOTHING USABLE. The search looked at {result.candidates_seen} possible "
            f"matches for '{result.query}' and every one scored below the relevance "
            "threshold. The service library does not cover this. Tell the person so. "
            "Do not use near-misses and do not answer from your own knowledge."
        )

    parts = [
        f"{len(result.chunks)} CANDIDATE document(s) for '{result.query}'.\n"
        "\n"
        "These are the closest text matches in the library. THE SEARCH CANNOT TELL "
        "WHETHER THEY ARE ABOUT THE RIGHT THING. It matches wording, not meaning. "
        "A document here may be about a completely different part of the vehicle "
        "that happens to be described with the same words.\n"
        "\n"
        "You must decide. For each one, ask: is this about the exact component the "
        "person asked about? If it is not, it is not usable, no matter how similar "
        "it sounds.\n"
    ]

    for n, c in enumerate(result.chunks, start=1):
        # Roughly 0.016 means one search method found it; roughly 0.032 means both
        # did. Agreement is a weak signal of quality, and no signal at all about
        # whether the topic is right.
        agreement = "both keyword and vector search found this" if c.score >= 0.028 else "only one search method found this"
        parts.append(
            f"--- CANDIDATE {n} ---\n"
            f"About: {c.title}\n"
            f"Cite as: ({c.citation()})\n"
            f"Type: {c.doc_type} | Severity: {c.severity} | Match: {agreement}\n"
            f"{c.content}\n"
        )

    parts.append(
        "\nBEFORE YOU WRITE YOUR ANSWER:\n"
        "1. Name to yourself which candidates are about the component asked about.\n"
        "2. Discard the rest. Do not mention them, do not reason from them, do not "
        "say 'this is similar to' or 'a related bulletin suggests'.\n"
        "3. If NONE of them is about the right component, then the library does not "
        "cover this question. Say that plainly - it is far better than an answer "
        "built from the wrong document.\n"
        "4. Every claim you make must come from a candidate you kept. Nothing may "
        "come from your own knowledge, including general explanations of how a "
        "part works.\n"
        "\n"
        "THIS TOOL ONLY TELLS YOU WHAT IS IN THE LIBRARY. It does not tell you what "
        "your answer should contain. Every other instruction you have still applies "
        "in full - in particular, if the question involves a safety system, your "
        "safety warning comes first whether or not anything here was usable. "
        "'Not covered' is never the whole answer to a safety question."
    )
    return "\n".join(parts)
