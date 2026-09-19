.PHONY: setup data ingest reindex search agents ask lint test clean \
        agents-list triage book serve docker-build docker-run \
        evals evals-smoke eval-one redteam

setup:
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	@echo "\nNow: cp .env.example .env and fill it in"

# generate the 30 synthetic service bulletins as PDFs
data:
	python scripts/generate_bulletins.py --count 30

# fill the search index (keeps the index if it exists)
ingest:
	python scripts/ingest.py

# drop the index and rebuild it from scratch
reindex:
	python scripts/ingest.py --recreate

# try a search without any agent involved
search:
	python scripts/search_test.py "$(Q)"

# create or update the agents in Foundry
agents:
	python agents/deploy_agents.py

# ask the diagnostics agent a question
ask:
	python agents/ask.py "$(Q)"

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

# --- week 2 ---
agents-list:
	python3 agents/deploy_agents.py --list

triage:
	python3 agents/ask.py "$(Q)" --agent triage

book:
	python3 agents/ask.py "$(Q)" --agent booking

# --- week 2: service ---
serve:
	uvicorn services.orchestrator.app:app --reload --port 8000

docker-build:
	docker build -t autoassist:local .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env autoassist:local

# --- week 3: evals ---
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
