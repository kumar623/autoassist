"""The document library, laid out for people to browse.

The chat page's Library tab lists everything the assistant can cite. It reads
the SEARCH INDEX, not the files in data/, for the same reason the eval scorer
does: the index is what the agents can actually retrieve. It is also the only
place the bulletins still exist - their PDFs are generated, gitignored, and gone
- and the container image does not include data/ at all.

Each document type is stored as text in a fixed shape by scripts/ingest.py.
The parsers here read that shape back into fields. They are pure functions,
tested offline in tests/test_library.py against real stored text.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from . import retrieval
from .cache import TimedCache

log = logging.getLogger(__name__)

# The index changes only when someone runs ingest, so re-reading all of it on
# every page view would be waste. Ten minutes keeps a fresh ingest visible soon.
CACHE_SECONDS = 600
_LIBRARY = TimedCache(CACHE_SECONDS, "library")

# A search returns at most 1000 results. The library is 370 pieces today; this
# is where it would silently start losing documents, so it is checked and logged.
MAX_PIECES = 1000

# The order the bulletin generator writes sections in. Anything unexpected goes
# after these, in page order, rather than being dropped.
SECTION_ORDER = [
    "TECHNICAL SERVICE BULLETIN",
    "SUMMARY",
    "AFFECTED VEHICLES",
    "SYMPTOMS",
    "PROBABLE CAUSE",
    "DIAGNOSTIC PROCEDURE",
    "REPAIR PROCEDURE",
    "PARTS REQUIRED",
    "LABOUR TIME",
    "WARRANTY INFORMATION",
]

_PIECE_PREFIX = re.compile(r"^\[[^\]]*\]\s*")  # "[TSB-010, section SYMPTOMS] "
_HEADER = re.compile(r"SERVICE BULLETIN\s+(TSB-\d+)\s+(.*?)\s+[—-]\s+(.*?)\s+SYNTHETIC DOCUMENT", re.S)
_DTC_FIRST_LINE = re.compile(r"^Fault code (\S+) \((.*?)\): (.*)\.$")


def _field(text: str, label: str) -> str:
    """The value on a 'Label: value' line, or ''."""
    m = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def fault_code(piece: dict) -> dict:
    content = piece.get("content") or ""
    first = content.splitlines()[0] if content else ""
    m = _DTC_FIRST_LINE.match(first)
    code, system, title = m.groups() if m else (piece.get("section", ""), "", piece.get("title", ""))
    safe = re.search(r"Safe to drive:\s*([^.\n]+)", content)
    return {
        "code": code,
        "system": system,
        "title": title,
        "meaning": _field(content, "What it means"),
        "causes": _field(content, "Common causes").rstrip("."),
        "severity": piece.get("severity") or "",
        "safe_to_drive": safe.group(1).strip() if safe else "",
        "cite_as": f"fault code list, {code}",
    }


def maintenance_item(piece: dict) -> dict:
    content = piece.get("content") or ""
    minutes = re.search(r"Typical workshop time:\s*(\d+)", content)
    item = piece.get("section", "")
    return {
        "item": item,
        "interval": _field(content, "Replacement or service interval").rstrip("."),
        "notes": _field(content, "Notes"),
        "workshop_minutes": int(minutes.group(1)) if minutes else None,
        "cite_as": f"maintenance schedule, {item}",
    }


def _section_rank(section: str) -> int:
    return SECTION_ORDER.index(section) if section in SECTION_ORDER else len(SECTION_ORDER)


def bulletins(pieces: list[dict]) -> list[dict]:
    """Reassemble bulletins from their pieces, one entry per source file."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for p in pieces:
        by_file[p["source_file"]].append(p)

    out = []
    for source in sorted(by_file):
        number = source.rsplit(".", 1)[0]
        vehicle = topic = ""
        sections: list[dict] = []

        for p in sorted(by_file[source], key=lambda p: (_section_rank(p["section"]), p.get("page") or 0)):
            text = _PIECE_PREFIX.sub("", p.get("content") or "").strip()
            if p["section"] == "HEADER":
                m = _HEADER.search(text)
                if m:
                    number, vehicle, topic = m.group(1), m.group(2).strip(), m.group(3).strip()
                continue
            name = p["section"].title()
            if sections and sections[-1]["name"] == name:
                # A long section is split across pieces; show it as one.
                sections[-1]["text"] += " " + text
            else:
                # Cited the same way retrieval.Chunk.citation() words it.
                sections.append({"name": name, "text": text, "cite_as": f"{number}, {name}"})

        out.append({"number": number, "vehicle": vehicle, "topic": topic, "sections": sections})

    out.sort(key=lambda b: b["number"])
    return out


def build(pieces: list[dict]) -> dict:
    """Group raw index pieces into the three lists the page shows."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in pieces:
        by_type[p.get("doc_type", "")].append(p)

    codes = sorted((fault_code(p) for p in by_type["dtc"]), key=lambda c: c["code"])
    items = [maintenance_item(p) for p in by_type["maintenance"]]
    items.sort(key=lambda i: i["item"].lower())
    tsbs = bulletins(by_type["bulletin"])

    return {
        "fault_codes": codes,
        "maintenance": items,
        "bulletins": tsbs,
        "counts": {
            "fault_codes": len(codes),
            "maintenance": len(items),
            "bulletins": len(tsbs),
            "pieces": len(pieces),
        },
    }


def load() -> dict:
    """The library, from the search index, cached for CACHE_SECONDS.

    The search runs OUTSIDE the cache's lock: TimedCache.get_or_call holds it
    only to read and to write, never while producing the value. That rule
    matters more here than anywhere. /library is a sync endpoint, so every
    reader holds one of Starlette's 40 threadpool threads for as long as this
    takes, and it is exempt from the rate limiter (decision 009). When this
    module kept its own cache and the lock spanned the search, a fetch that
    failed handed the lock to the next reader, which started its own full
    attempt from the front of the queue: six readers on a cold cache cost six
    fetches end to end, not one. azure_http retries a 5xx up to three times at
    up to 20s, so each of those is worth up to a minute of a held thread. The
    threadpool empties, /health - sync too - cannot get a thread, the Dockerfile
    HEALTHCHECK fails its three 4s tries, and the container is restarted, all
    because people opened the Library tab on a replica that had just scaled from
    zero.

    Two readers arriving on a cold cache therefore both search, rather than one
    waiting on the other. That is the trade, taken deliberately: a duplicate
    search costs a fraction of a second against an index that changes only when
    someone runs ingest, and blocking costs a thread that /health needs.
    """
    return _LIBRARY.get_or_call("library", _read_index)


def _read_index() -> dict:
    fields = ["doc_type", "source_file", "section", "page", "severity", "title", "content"]
    pieces = retrieval.fetch_all(fields, top=MAX_PIECES)
    if len(pieces) >= MAX_PIECES:
        log.warning("library read %d pieces, the most one search returns; some are missing", len(pieces))

    data = build(pieces)
    log.info("library loaded: %s", data["counts"])
    return data
