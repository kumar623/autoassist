"""Create or update Foundry agents from JSON definitions.

Idempotent: an agent with the same name is updated in place, so a prompt change
is a git commit plus one command, not a container rebuild.

Week 2 change: agents no longer use Azure AI Search's built-in tool. They get
FUNCTION tools that our own orchestrator executes (services/orchestrator/tools.py).
That was forced by finding 5 - the built-in tool hung on every vector query type
- but it is the better design anyway: we control the relevance floor, the
per-source cap and the citation format, none of which the built-in tool exposes.

Usage:
    python3 agents/deploy_agents.py                 # deploy all definitions
    python3 agents/deploy_agents.py --only booking
    python3 agents/deploy_agents.py --list
    python3 agents/deploy_agents.py --delete-all    # clean slate
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Import the tool schemas from the orchestrator package. Run from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.orchestrator import tools  # noqa: E402

import os  # noqa: E402

DEFS = pathlib.Path(__file__).resolve().parent / "definitions"
ENDPOINT = os.environ["PROJECT_ENDPOINT"]


def client() -> AgentsClient:
    return AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())


def deploy_one(c: AgentsClient, path: pathlib.Path) -> None:
    spec = json.loads(path.read_text())
    name = spec["name"]
    tool_names = spec.get("tools", [])

    kwargs: dict = {
        "model": spec["model"],
        "name": name,
        "description": spec.get("description"),
        "instructions": spec["instructions"],
        "temperature": spec.get("temperature", 0.2),
        "tools": tools.schemas_for(tool_names),
    }

    if spec.get("response_format") == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    existing = next((a for a in c.list_agents() if a.name == name), None)
    tool_list = ", ".join(tool_names) if tool_names else "no tools"

    if existing:
        c.update_agent(agent_id=existing.id, **kwargs)
        print(f"updated  {name:<12} [{tool_list}]  {existing.id}")
    else:
        agent = c.create_agent(**kwargs)
        print(f"created  {name:<12} [{tool_list}]  {agent.id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="deploy just this agent name")
    parser.add_argument("--list", action="store_true", help="list agents in the project")
    parser.add_argument("--delete-all", action="store_true", help="delete every agent, then stop")
    args = parser.parse_args()

    with client() as c:
        if args.list:
            found = False
            for a in c.list_agents():
                names = [t.get("function", {}).get("name", t.get("type")) for t in (a.tools or [])]
                print(f"{a.id}  {a.name:<12} {a.model:<14} tools={names}")
                found = True
            if not found:
                print("(no agents in this project)")
            return 0

        if args.delete_all:
            for a in list(c.list_agents()):
                c.delete_agent(a.id)
                print(f"deleted  {a.name}  {a.id}")
            return 0

        files = sorted(DEFS.glob("*.json"))
        if args.only:
            files = [f for f in files if f.stem == args.only]
            if not files:
                print(f"no definition named {args.only}")
                return 1

        for f in files:
            deploy_one(c, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
