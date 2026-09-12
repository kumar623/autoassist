"""Build the AI Search index and fill it.

Reads three sources, turns them into chunks, embeds each chunk, and uploads
them to one hybrid (keyword + vector) index called `service-docs`.

  1. data/dtc_codes.csv          -> one chunk per fault code
  2. data/maintenance.csv        -> one chunk per maintenance item
  3. data/synthetic_bulletins/*.pdf -> chunked by section heading

Usage:
    python scripts/ingest.py --recreate     # drop and rebuild the index
    python scripts/ingest.py                # upsert into the existing index
"""

import argparse
import csv
import hashlib
import os
import pathlib
import re
import sys
import time

import fitz  # pymupdf
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "service-docs")
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")
# The base model behind the deployment. Must match EMBED_DEPLOYMENT's model, or
# documents and questions land in different meaning-spaces and retrieval quietly
# returns nonsense with no error.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
EMBED_DIMS = 1536
CHUNK_TOKENS = 400  # roughly; we count words and multiply
CHUNK_OVERLAP_WORDS = 60


# ---------------------------------------------------------------- clients


def openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )


def index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=os.environ["SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"]),
    )


def search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"]),
    )


# ---------------------------------------------------------------- index


def build_index(recreate: bool) -> None:
    ic = index_client()
    existing = [i for i in ic.list_index_names()]

    if INDEX_NAME in existing:
        if not recreate:
            print(f"index '{INDEX_NAME}' already exists, keeping it")
            return
        print(f"deleting index '{INDEX_NAME}'")
        ic.delete_index(INDEX_NAME)

    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="severity", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="vehicle_model", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBED_DIMS,
                vector_search_profile_name="hnsw-profile",
            ),
        ],
        # The vectorizer lets the INDEX turn query text into a vector by calling
        # Azure OpenAI itself. Our scripts embed questions in Python and do not
        # need it, but the agent's Azure AI Search tool only sends text, so
        # without a vectorizer it fails with:
        #   "Query type vector_simple_hybrid requires a vector field with
        #    integrated vectorizer, but none was found"
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-config",
                    vectorizer_name="aoai-vectorizer",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="aoai-vectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=os.environ["AZURE_OPENAI_ENDPOINT"],
                        deployment_name=EMBED_DEPLOYMENT,
                        model_name=EMBED_MODEL_NAME,
                        api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    ),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="semantic-config",
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ],
        ),
    )
    ic.create_index(index)
    print(f"created index '{INDEX_NAME}'")


# ---------------------------------------------------------------- chunking


def chunk_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:24]


def load_dtc_chunks() -> list[dict]:
    path = DATA / "dtc_codes.csv"
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["code"].strip()
            if not code:
                continue
            content = (
                f"Fault code {code} ({row['system']}): {row['title']}.\n"
                f"What it means: {row['plain_meaning']}\n"
                f"Common causes: {row['common_causes']}.\n"
                f"Severity: {row['severity']}. Safe to drive: {row['safe_to_drive']}."
            )
            out.append(
                {
                    "id": chunk_id("dtc", code),
                    "title": f"{code} - {row['title']}",
                    "content": content,
                    "doc_type": "dtc",
                    "source_file": "dtc_codes.csv",
                    "section": code,
                    "page": 0,
                    "severity": row["severity"],
                    "vehicle_model": "generic",
                }
            )
    print(f"dtc chunks: {len(out)}")
    return out


def load_maintenance_chunks() -> list[dict]:
    path = DATA / "maintenance.csv"
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = row["item"].strip()
            if not item:
                continue
            km = row["interval_km"]
            months = row["interval_months"]
            when = []
            if km and km != "0":
                when.append(f"every {int(km):,} km")
            if months and months != "0":
                when.append(f"every {months} months")
            when_str = " or ".join(when) if when else "no fixed interval"
            content = (
                f"Maintenance item: {item}.\n"
                f"Replacement or service interval: {when_str}.\n"
                f"Notes: {row['notes']}\n"
                f"Typical workshop time: {row['typical_duration_min']} minutes."
            )
            out.append(
                {
                    "id": chunk_id("maint", item),
                    "title": f"Maintenance: {item}",
                    "content": content,
                    "doc_type": "maintenance",
                    "source_file": "maintenance.csv",
                    "section": item,
                    "page": 0,
                    "severity": "low",
                    "vehicle_model": "generic",
                }
            )
    print(f"maintenance chunks: {len(out)}")
    return out


HEADING_RE = re.compile(r"^[A-Z][A-Z \-/]{4,}$")


def split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start : start + size]))
        start += size - overlap
    return out


def load_bulletin_chunks() -> list[dict]:
    pdf_dir = DATA / "synthetic_bulletins"
    out = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(pdf)
        # collect (section, text, page) by walking lines and watching for headings
        sections: list[tuple[str, list[str], int]] = []
        current = ("HEADER", [], 1)
        for pno, page in enumerate(doc, start=1):
            for line in page.get_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                if HEADING_RE.match(line) and len(line.split()) <= 6:
                    sections.append(current)
                    current = (line, [], pno)
                else:
                    current[1].append(line)
        sections.append(current)
        doc.close()

        title_guess = pdf.stem
        for section, lines, pno in sections:
            body = " ".join(lines).strip()
            if len(body) < 40:
                continue
            for n, piece in enumerate(split_words(body, CHUNK_TOKENS, CHUNK_OVERLAP_WORDS)):
                out.append(
                    {
                        "id": chunk_id("tsb", pdf.name, section, str(n)),
                        "title": f"{title_guess} - {section.title()}",
                        "content": f"[{title_guess}, section {section}] {piece}",
                        "doc_type": "bulletin",
                        "source_file": pdf.name,
                        "section": section,
                        "page": pno,
                        "severity": "medium",
                        "vehicle_model": "corvale",
                    }
                )
    print(f"bulletin chunks: {len(out)}")
    return out


# ---------------------------------------------------------------- embed + upload


def embed_all(chunks: list[dict]) -> None:
    oai = openai_client()
    batch = 64
    for i in range(0, len(chunks), batch):
        window = chunks[i : i + batch]
        resp = oai.embeddings.create(
            model=EMBED_DEPLOYMENT,
            input=[c["content"] for c in window],
        )
        for c, item in zip(window, resp.data):
            c["content_vector"] = item.embedding
        print(f"embedded {min(i + batch, len(chunks))}/{len(chunks)}")
        time.sleep(0.2)


def upload(chunks: list[dict]) -> None:
    sc = search_client()
    batch = 200
    for i in range(0, len(chunks), batch):
        result = sc.upload_documents(documents=chunks[i : i + batch])
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"  {len(failed)} documents failed in this batch")
            for r in failed[:3]:
                print(f"    {r.key}: {r.error_message}")
        print(f"uploaded {min(i + batch, len(chunks))}/{len(chunks)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the index")
    parser.add_argument("--skip-bulletins", action="store_true")
    args = parser.parse_args()

    build_index(recreate=args.recreate)

    chunks = load_dtc_chunks() + load_maintenance_chunks()
    if not args.skip_bulletins:
        chunks += load_bulletin_chunks()

    if not chunks:
        print("nothing to ingest")
        return 1

    print(f"\ntotal chunks: {len(chunks)}")
    embed_all(chunks)
    upload(chunks)

    time.sleep(2)
    count = search_client().get_document_count()
    print(f"\nindex '{INDEX_NAME}' now holds {count} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
