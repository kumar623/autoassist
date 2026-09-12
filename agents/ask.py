"""Ask one agent one question, in a fresh thread, and show exactly what it did.

Usage:
    python3 agents/ask.py "what does P0420 mean"
    python3 agents/ask.py "my brakes feel spongy" --agent diagnostics
    python3 agents/ask.py "any slots on saturday" --agent booking
    python3 agents/ask.py "P0420 and can I come in saturday" --agent triage
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.orchestrator import runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--agent", default="diagnostics")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--verbose", action="store_true", help="show tool logging as it happens")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    with AgentsClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = next((a for a in client.list_agents() if a.name == args.agent), None)
        if agent is None:
            print(f"agent '{args.agent}' not found. Run: python3 agents/deploy_agents.py")
            return 1

        result = runner.ask(client, agent.id, args.question, timeout=args.timeout, agent_name=args.agent)

    print("\n" + "=" * 74)
    print(result.answer or "(no answer)")
    print("=" * 74)

    if result.error:
        print(f"\nERROR: {result.error}")

    print(f"\nagent      : {result.agent_name}")
    print(f"status     : {result.status}")
    print(f"searched   : {'YES' if result.searched else 'NO'}")
    print(f"tool calls : {result.tool_summary()}")

    for c in result.tool_calls:
        flag = "  FAILED" if c.failed else ""
        print(f"  - {c.name}({c.arguments}){flag}")
        preview = c.output_preview.replace("\n", " ")[:150]
        print(f"      -> {preview}...")

    if not result.searched and args.agent in ("diagnostics",):
        print("\n  WARNING: no search call. Any technical claim above is ungrounded -")
        print("  it came from the model, not from your documents.")

    print(f"\ntokens     : prompt {result.prompt_tokens}, completion {result.completion_tokens}")
    print(f"took       : {result.duration_ms}ms")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
