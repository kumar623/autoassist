# Deploying

`.github/workflows/deploy.yml` builds an image, pushes it, deploys it, checks it
is really serving, and puts the old one back if it is not.

It runs **only after CI passes on a push to main in this repository**. Green CI
is the entry condition, not a second opinion. CI on a pull request - including
one from a fork whose branch happens to be called `main` - never deploys.

## The shape of it

```
push to main
   └─ CI: lint, 112 tests, agent definitions, Docker build   (no Azure, no cost)
        └─ Deploy:  build → push → update → smoke test
                                                └─ failed? roll back, verify, fail loudly
```

## What makes the smoke test worth having

`az containerapp update` exits 0 when Azure **accepts** the request. That is not
the same as the new container starting, and not the same as it serving traffic.
A pipeline that stops there reports success while a crash-looping revision sits
behind an old one that is still answering.

So `/health` reports the git SHA it was built from (`GIT_SHA`, set by the deploy
step), and the smoke test waits for **that specific build** to answer. Polling
for HTTP 200 would pass immediately against the old revision.

Then it calls `/ready`, which checks the service reached Azure and found all
four agents. That is the check that catches a wrong endpoint, a missing role
assignment, or agents that were never deployed — all things a liveness probe
happily reports as fine.

## One-time setup

### 1. Register the app and let GitHub log in as it

No client secret is created. GitHub mints a short-lived token per run and Azure
trusts it because of a *federated credential* naming this exact repo. Nothing to
store, nothing to leak, nothing to rotate.

```bash
APP_ID=$(az ad app create --display-name autoassist-github --query appId -o tsv)
az ad sp create --id "$APP_ID"
SP_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)
echo "client id: $APP_ID"
```

Two federated credentials are needed, because GitHub's token says something
different depending on how the job runs:

```bash
# The deploy job declares `environment: production`, which makes the subject the
# environment rather than the branch.
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-production",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:kumar623/autoassist:environment:production",
  "audiences": ["api://AzureADTokenExchange"]
}'

# CI's eval job has no environment, so its subject is the branch.
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:kumar623/autoassist:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'
```

A mismatched `subject` is the usual cause of `AADSTS70021: No matching
federated identity record found`. The string must match character for character.

### 2. Give it the two permissions it needs, and no more

```bash
SUB=$(az account show --query id -o tsv)
RG=Ai_solution            # the resource group the app actually runs in
                          # (rg-autoassist is the Foundry ACCOUNT's name, not a group)
ACR=<your registry name>
APP=<your container app name>

# Push images.
az role assignment create --assignee-object-id "$SP_ID" \
  --assignee-principal-type ServicePrincipal \
  --role AcrPush \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/$ACR"

# Change one container app. Scoped to the app, not the resource group: this
# identity cannot touch the search service, the models or the storage account.
az role assignment create --assignee-object-id "$SP_ID" \
  --assignee-principal-type ServicePrincipal \
  --role Contributor \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/$APP"
```

### 3. Tell the repo where to deploy

Settings → Secrets and variables → Actions.

**Secrets** (hidden in logs):

| name | value |
|---|---|
| `AZURE_CLIENT_ID` | the `appId` printed above |
| `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | `az account show --query id -o tsv` |

None of these are really secret — they are identifiers, not credentials, and the
federated credential is what grants access. They are stored as secrets out of
habit and because there is no reason to publish them.

**Variables** (visible, and useful in logs):

| name | value |
|---|---|
| `AZURE_RESOURCE_GROUP` | e.g. `Ai_solution` |
| `AZURE_CONTAINER_APP` | e.g. `ca-autoassist` |
| `AZURE_REGISTRY` | the login server, e.g. `crautoassist.azurecr.io` |

### 4. For the weekly evals

`.github/workflows/evals.yml` runs real questions against the agents, so it needs
more than the deploy does. Without these it skips with a notice rather than
failing.

**Variables:** `PROJECT_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`, `SEARCH_ENDPOINT`
**Secrets:** `AZURE_OPENAI_API_KEY`, `SEARCH_API_KEY`

And the GitHub identity needs to list and run agents - the same role the app
needs, on the Foundry account only:

```bash
AI_ID=$(az cognitiveservices account show -g "$RG" -n rg-autoassist --query id -o tsv)
az role assignment create --assignee-object-id "$SP_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Foundry User" --scope "$AI_ID"
```

Name the account. The resource group holds other Foundry accounts, and taking
"the first AIServices account in the group" can pick the wrong one.

### 5. Optional: require an approval

Settings → Environments → `production` → required reviewers. Every deploy then
waits for a person. Nothing in the workflow file changes.

## Rolling back

Actions → Deploy → Run workflow → put a git SHA in **image_tag**.

That skips the build entirely and deploys an image that already exists, which is
the whole reason images are tagged by SHA rather than only `latest`. A `latest`
tag cannot be rolled back to, because it has already moved.

The pipeline also rolls back on its own if a smoke test fails — and then
verifies the rollback with the same two checks as the smoke test: the previous
build's SHA is what `/health` reports, and `/ready` passes. A rollback that
quietly failed is worse than the original fault, because by then nobody is
watching.

It has fired once, on 12 September (95bc9e5). The new build came up, then
`/ready` reported `agents not deployed: triage, diagnostics, booking,
escalation` — the app's identity could reach the project but not list agents.
Two things were wrong with the rollback that followed, both fixed since:

- It restored the image but not `GIT_SHA`, so `/health` kept reporting the
  failed build while the previous one ran.
- It counted any HTTP 200 from `/health` as success. That passed in 5 seconds
  and said nothing about whether the rolled-back app could serve. Here it could
  not have been the code: 95bc9e5 changed only `infra/` and a doc, so the two
  images were identical. When both builds fail `/ready`, the rollback now says
  the fault is in Azure rather than reporting success.

## Things this does not do

- **No blue/green or canary.** The container app is in `Single` revision mode, so
  a deploy replaces the revision and the rollback is a second deploy — perhaps
  ninety seconds of the old build if something is wrong. Real zero-downtime
  would mean `Multiple` revision mode and shifting `traffic_weight` gradually,
  which is a better answer for a product and more moving parts than this needs.
- **No database migrations.** There is no database. Bookings are a JSON file in
  the container, which is also why they do not survive a deploy — a known
  limitation, and the reason Table Storage is the next step.
- **No infrastructure changes.** Terraform is run by hand (`infra/README.md`).
  Deploying code and changing infrastructure on the same trigger means a bad
  application commit can delete a search index. They are kept apart on purpose.
- **The agents are not redeployed.** `agents/deploy_agents.py` is still manual.
  Prompt changes are the highest-risk change in this system — `docs/evaluation.md`
  findings 4, 7 and 9 are all prompts — so they go out deliberately, with the
  full eval set run afterwards, not as a side effect of a push.


## Choosing the triage classifier

Routing is decided either by the triage agent (gpt-4.1-mini writing JSON) or by
Jev (four probabilities, thresholds in code). See docs/evaluation.md, findings 15
and 16, for the measurement behind offering the choice.

It is a repository **variable**, so flipping it needs no code change:

    Settings -> Secrets and variables -> Actions -> Variables -> TRIAGE_BACKEND

| value | what routes |
|---|---|
| `agent` (default) | the triage agent |
| `jev` | Jev, falling back to the agent for anything it cannot answer |

Jev also needs the repository **secret** `TYPESAFE_API_KEY`. The deploy copies it
onto the container app as a secret on every run and reads it through
`secretref`, so GitHub is the one place to rotate it and the value never appears
in `az containerapp show`. Without the secret the deploy stays on the agent and
says so, rather than failing.

The change takes effect on the next deploy. To flip it immediately, without one:

    az containerapp update -g Ai_solution -n ca-autoassist --set-env-vars TRIAGE_BACKEND=agent

**Check it is really answering.** A fallback is silent - the customer gets a
reply either way - so a revoked key would quietly send everything back to the
agent. `/metrics` reports `triage_backend`, `jev_answered` and `jev_fell_back`;
if `jev_fell_back` is climbing, Jev is configured but not the thing deciding.
