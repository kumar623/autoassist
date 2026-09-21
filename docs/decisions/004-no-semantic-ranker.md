# 004 - No semantic ranker on the Free search tier

## Status
Accepted (week 1)

## Context
The retrieval pipeline was built to use Azure AI Search's semantic ranker: a
second-stage model that re-reads the top keyword+vector results and reorders
them by how well they actually answer the question.

Microsoft's documentation states semantic ranker is "available on all pricing
tiers" under the free billing plan. In practice, a Free tier search service
rejects the request:

    (FeatureNotSupportedInService) Semantic search is not enabled for this service.
    Parameter name: queryType

The free *billing plan* for semantic ranker is available on all tiers that
support the feature at all - but the Free *service tier* does not support it.
Those are two different things with confusingly similar names.

## Decision
Run hybrid retrieval without the reranker: BM25 keyword search merged with
vector similarity, fused by reciprocal rank fusion.

Both the search test script and the agent tool default to simple hybrid.
Setting `SEARCH_SEMANTIC=true` switches both to the semantic variant, so
upgrading the service to Basic is a one-line config change, not a code change.

> **Note, 21 September 2026.** The paragraph above stopped being true in week 2,
> when retrieval moved into `services/orchestrator/retrieval.py`. Nothing reads
> `SEARCH_SEMANTIC` any more, and it has been removed from `.env.example`:
> `retrieval.py` always sends a plain hybrid query, and `scripts/search_test.py`
> now runs that same code. The index was built with a semantic configuration
> (`semantic-config`, in `scripts/ingest.py`), so after an upgrade to Basic the
> ranker would be a change in `retrieval.py` - the query type and that
> configuration's name in the request - and a re-run of the evals, not a setting.

## Consequences
- Retrieval quality is lower than it would be with reranking, particularly for
  vaguely worded questions where lexical overlap is weak.
- Exact-token lookups (fault codes like P0420) are unaffected - those are
  carried by the keyword half of the hybrid.
- The evaluation in week 3 measures hybrid-without-reranker as the baseline.
  If retrieval quality proves to be the limiting factor, upgrading to Basic
  (about $75/month) is the first lever to pull, and the eval set will show
  whether it was worth it.

## Lesson
Verify tier limits against the service, not the documentation. The error only
appeared at query time, not when creating the index - the index accepted a
semantic configuration it could never use.
