"""What the diagnostics agent would be handed for a question, with no agent involved.

Runs the service's own retrieval (services/orchestrator/retrieval.py): hybrid
search, then the relevance floor and the per-source cap. The week 1 version of
this script queried the index directly through the SDKs and skipped both
filters, so what it printed was not what the agent saw.

Usage:
    python scripts/search_test.py "my catalytic converter light is on"
    python scripts/search_test.py "brake fluid" --doc-type maintenance --k 3
    python scripts/search_test.py "spongy brakes" --as-agent    # the exact tool output
    make search Q="spongy brakes"

Read-only: one embedding call and one search. Needs the Azure OpenAI and Search
endpoints and keys in .env.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Before the import below: retrieval reads its settings - index name, floor,
# cap - when it is imported.
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from services.orchestrator import azure_http, retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=retrieval.DEFAULT_TOP_K,
                        help="chunks to keep; the agent gets %(default)s")
    parser.add_argument("--doc-type", choices=retrieval.DOC_TYPES,
                        help="only fault codes, maintenance items or bulletins")
    parser.add_argument("--as-agent", action="store_true",
                        help="print the tool output exactly as the agent receives it")
    args = parser.parse_args()

    try:
        result = retrieval.search(args.query, top_k=args.k, doc_type=args.doc_type)
    except azure_http.AzureError as e:
        # Azure's own message and the path, never the request headers.
        print(f"search failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        # The type only. A client-side error can quote the request it refused,
        # and the request's headers carry the keys.
        print(f"search failed: {type(e).__name__}. Check the endpoints and keys in .env.", file=sys.stderr)
        return 1

    if args.as_agent:
        print(retrieval.format_for_agent(result))
        return 0

    print(f"\nquery : {args.query}" + (f"   (doc_type={args.doc_type})" if args.doc_type else ""))
    print(f"kept  : {result.summary()}")
    print(f"rules : floor {retrieval.RELEVANCE_FLOOR}, at most {retrieval.MAX_PER_SOURCE} per source")
    print("-" * 70)

    if not result.found_anything:
        print("\n  nothing survived: the agent is told the library does not cover this\n")
        return 1

    for n, c in enumerate(result.chunks, start=1):
        print(f"\n[{n}] score {c.score:.4f}  cite as ({c.citation()})")
        print(f"    {c.title}")
        body = c.content.replace("\n", " ")
        print(f"    {body[:220]}{'...' if len(body) > 220 else ''}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
