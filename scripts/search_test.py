"""Prove retrieval works before you build any agent on top of it.

Runs hybrid search (keyword + vector + semantic ranker) and prints what came back.

Usage:
    python scripts/search_test.py "my catalytic converter light is on"
    python scripts/search_test.py "how often should I change brake fluid" --k 3
"""

import argparse
import os
import sys

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "service-docs")
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "vector", "semantic"],
        default="hybrid",
        help="hybrid = keyword + vector (works on Free tier). "
        "semantic = hybrid plus the reranker (needs Basic tier or higher).",
    )
    args = parser.parse_args()

    oai = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    sc = SearchClient(
        endpoint=os.environ["SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"]),
    )

    kwargs: dict = {"top": args.k, "select": ["title", "content", "doc_type", "source_file", "section", "severity"]}

    if args.mode in ("hybrid", "vector", "semantic"):
        vec = oai.embeddings.create(model=EMBED_DEPLOYMENT, input=[args.query]).data[0].embedding
        kwargs["vector_queries"] = [
            VectorizedQuery(vector=vec, k_nearest_neighbors=args.k * 2, fields="content_vector")
        ]

    kwargs["search_text"] = args.query if args.mode in ("hybrid", "keyword", "semantic") else None

    if args.mode == "semantic":
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = "semantic-config"

    print(f"\nquery : {args.query}")
    print(f"mode  : {args.mode}\n" + "-" * 70)

    try:
        results = list(sc.search(**kwargs))
    except HttpResponseError as e:
        if "Semantic search is not enabled" in str(e):
            print(
                "\n  Semantic ranker is not available on this search service (Free tier).\n"
                "  Falling back to hybrid without the reranker.\n"
                "  To use it: upgrade the service to Basic or higher.\n"
            )
            kwargs.pop("query_type", None)
            kwargs.pop("semantic_configuration_name", None)
            results = list(sc.search(**kwargs))
        else:
            raise

    if not results:
        print("\n  no results\n")
        return 1

    for n, r in enumerate(results, start=1):
        score = r.get("@search.reranker_score") or r["@search.score"]
        print(f"\n[{n}] score {score:.4f}  ({r['doc_type']} / {r['source_file']} / {r['section']})")
        print(f"    {r['title']}")
        body = r["content"].replace("\n", " ")
        print(f"    {body[:220]}{'...' if len(body) > 220 else ''}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
