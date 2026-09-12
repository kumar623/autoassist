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
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

log = logging.getLogger(__name__)

INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "service-docs")
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")

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
def _openai() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )


@lru_cache(maxsize=1)
def _search() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"]),
    )


def embed(text: str) -> list[float]:
    return _openai().embeddings.create(model=EMBED_DEPLOYMENT, input=[text]).data[0].embedding


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
    result = RetrievalResult(query=query)
    want = top_k * CANDIDATE_MULTIPLIER

    kwargs: dict = {
        "search_text": query,
        "top": want,
        "select": [
            "id", "title", "content", "doc_type",
            "source_file", "section", "page", "severity",
        ],
        "vector_queries": [
            VectorizedQuery(vector=embed(query), k_nearest_neighbors=want, fields="content_vector")
        ],
    }
    if doc_type:
        kwargs["filter"] = f"doc_type eq '{doc_type}'"

    per_source: dict[str, int] = {}

    for raw in _search().search(**kwargs):
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
