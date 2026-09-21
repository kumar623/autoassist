.PHONY: setup data ingest reindex search agents ask lint test clean \
        agents-list triage book serve docker-build docker-run \
        evals evals-smoke eval-one redteam compare-triage

# requirements.txt starts with requirements-service.txt, so this installs the
# service as well as the scripts and dev tools. constraints.txt stops pip
# spending minutes resolving the OpenTelemetry packages; see the file.
setup:
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -c constraints.txt -r requirements.txt
	@echo "\nNow: cp .env.example .env and fill it in"

# --- the search index ---
#
# The live index is the only copy of the bulletins it was built from: the PDFs
# are gone. `data` and `reindex` are kept because they are how the index was
# built, but each would destroy it, so both refuse unless CONFIRM=1.
#
#   data     writes 30 NEW bulletins under the same TSB numbers (the model runs
#            at temperature 0.8, so the text differs). An ingest after it
#            overwrites the live chunks in place: chunk ids are a hash of file
#            name and section, not of content.
#   reindex  drops the index and rebuilds it from data/synthetic_bulletins/.
#            Without those PDFs it rebuilds 91 fault-code and maintenance chunks
#            and loses the 279 bulletin chunks.
#
# `ingest` on its own is safe: without PDFs it re-uploads the fault codes and
# maintenance items unchanged.

data:
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "Refusing: this writes new bulletins over the names the live index uses."; \
	  echo "An ingest after it replaces the live bulletin chunks. See the Makefile."; \
	  echo "Run 'make data CONFIRM=1' only for a fresh environment."; \
	  exit 1; \
	fi
	python scripts/generate_bulletins.py --count 30

ingest:
	python scripts/ingest.py

reindex:
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "Refusing: this DELETES the search index and rebuilds it from data/."; \
	  echo "The bulletin PDFs the live index came from no longer exist. See the Makefile."; \
	  echo "Run 'make reindex CONFIRM=1' only for a fresh environment."; \
	  exit 1; \
	fi
	python scripts/ingest.py --recreate

# what the diagnostics agent would be handed for a question, with no agent involved
search:
	python scripts/search_test.py "$(Q)"

# --- agents ---

# create or update the agents in Foundry
agents:
	python agents/deploy_agents.py

# ask the diagnostics agent a question
ask:
	python agents/ask.py "$(Q)"

agents-list:
	python3 agents/deploy_agents.py --list

triage:
	python3 agents/ask.py "$(Q)" --agent triage

book:
	python3 agents/ask.py "$(Q)" --agent booking

# --- code ---

# The same check CI runs. `ruff format` is not enforced: most files were written
# before it was adopted, and reformatting them all would bury real diffs.
lint:
	ruff check .

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

# --- the service ---
serve:
	uvicorn services.orchestrator.app:app --reload --port 8000

docker-build:
	docker build -t autoassist:local .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env autoassist:local

# --- evals ---
# Needs Azure. Not part of `make test`, which is offline and runs in CI.
evals:
	python3 evals/run_evals.py --save

evals-smoke:
	python3 evals/run_evals.py --smoke

eval-one:
	python3 evals/run_evals.py --only $(ID)

# Attacks on the real agents: poisoned documents, other people's bookings,
# prompt extraction. Needs Azure; ~60k tokens. Scratch stores, never live data.
redteam:
	python3 evals/red_team.py

# Measure Jev (TypeSafe) against the triage agent over the labelled routing set.
# Needs Azure, and TYPESAFE_API_KEY for the Jev side - without the key it
# reports the triage numbers and says Jev was not asked. The service can route
# with either (TRIAGE_BACKEND); this measures both, it does not switch anything.
compare-triage:
	python3 evals/compare_triage.py $(ARGS)
