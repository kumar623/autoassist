# Deploying

`.github/workflows/deploy.yml` builds an image, pushes it, deploys it, checks it
is really serving, and puts the old one back if it is not.

It runs **only after CI passes on a push to main in this repository**. Green CI
is the entry condition, not a second opinion. CI on a pull request - including
one from a fork whose branch happens to be called `main` - never deploys.

## The shape of it

```
push to main
   └─ CI: lint, offline tests, agent definitions, Docker build   (no Azure, no cost)
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

Two scripts do most of it. Each explains itself at the top; this section says
what they are for and what is still done by hand.

### 1. The keys, in Key Vault (by hand)

The app's six keys live in Key Vault `kv-autoassist-kk` and the container app
holds only references to them. Why, and what was rejected:
`docs/decisions/010-keys-in-key-vault.md`.

| Secret | The app's environment variable, and the `.env` line it comes from |
|---|---|
| `openai-key` | `AZURE_OPENAI_API_KEY` |
| `search-key` | `SEARCH_API_KEY` |
| `zoho-mcp-url` | `ZOHO_MCP_URL` |
| `zoho-client-id` | `ZOHO_MCP_CLIENT_ID` |
| `zoho-refresh-token` | `ZOHO_MCP_REFRESH_TOKEN` |
| `typesafe-key` | `TYPESAFE_API_KEY` |

```bash
RG=Ai_solution
KV=kv-autoassist-kk
az keyvault create -g "$RG" -n "$KV" --enable-rbac-authorization true
KV_ID=$(az keyvault show -n "$KV" --query id -o tsv)

# You add and replace values. The app will only read them; GitHub gets nothing.
az role assignment create --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Key Vault Secrets Officer" --scope "$KV_ID"

# From .env, so no value is typed or pasted. -o none matters: the command's
# output includes the value it just stored.
set -a; source .env; set +a
az keyvault secret set --vault-name "$KV" -n openai-key         --value "$AZURE_OPENAI_API_KEY"   -o none
az keyvault secret set --vault-name "$KV" -n search-key         --value "$SEARCH_API_KEY"         -o none
az keyvault secret set --vault-name "$KV" -n zoho-mcp-url       --value "$ZOHO_MCP_URL"           -o none
az keyvault secret set --vault-name "$KV" -n zoho-client-id     --value "$ZOHO_MCP_CLIENT_ID"     -o none
az keyvault secret set --vault-name "$KV" -n zoho-refresh-token --value "$ZOHO_MCP_REFRESH_TOKEN" -o none
az keyvault secret set --vault-name "$KV" -n typesafe-key       --value "$TYPESAFE_API_KEY"       -o none
```

A new role assignment can take a minute to apply; if `secret set` is refused
straight after the one above, wait and run it again.

Changing a key later is a new version of the secret, not a deploy: see
`docs/runbook.md`, "Rotating a key".

### 2. Somewhere to deploy to

```bash
KEY_VAULT_NAME=kv-autoassist-kk ./scripts/setup_deploy_target.sh Ai_solution
```

Creates the container registry, the app's identity (`id-autoassist`), the
container app environment and the container app, and builds a first image in
Azure. It gives the identity three roles: `AcrPull` on the registry, **Foundry
User** on the Foundry account (`rg-autoassist` by default - name it, because
the resource group holds more than one), and **Key Vault Secrets User** on the
vault. The app starts with references to `openai-key` and `search-key`.

The three Zoho references, which the live app has because it books into Zoho
(`BOOKING_BACKEND=zoho`), are added the same way: `az containerapp secret set`
with `zoho-mcp-url=keyvaultref:<vault url>/secrets/zoho-mcp-url,identityref:<identity id>`,
then `ZOHO_MCP_URL=secretref:zoho-mcp-url` among the app's environment
variables. `typesafe-key` is referenced by the deploy itself; see "Choosing the
triage classifier" below.

### 3. Let GitHub log in as a deploy identity

```bash
./scripts/setup_github_oidc.sh Ai_solution <registry-name> ca-autoassist
```

No client secret is created. GitHub mints a short-lived token per run and Azure
trusts it because of a *federated credential* naming this exact repo. Nothing to
store, nothing to leak, nothing to rotate.

The script creates the `autoassist-github` app registration and two federated
credentials, because GitHub's token says something different depending on how
the job runs:

| credential | subject | used by |
|---|---|---|
| `github-production` | `repo:kumar623/autoassist:environment:production` | the deploy job, which declares `environment: production` |
| `github-main` | `repo:kumar623/autoassist:ref:refs/heads/main` | the weekly evals workflow, which has no environment |

A mismatched `subject` is the usual cause of `AADSTS70021: No matching
federated identity record found`. The string must match character for character.

It then grants two roles, each scoped to one resource: `AcrPush` on the
registry, and Contributor on the one container app - not the resource group, so
this identity cannot touch the search service or the vault. (The live identity
holds Container Apps Contributor on the app instead.) Either role
includes `listSecrets` on the app, which is why no key value is kept there:
listing now returns vault addresses.

### 4. Tell the repo where to deploy

Settings → Secrets and variables → Actions. The two scripts print these values.

**Secrets** (hidden in logs):

| name | value |
|---|---|
| `AZURE_CLIENT_ID` | the `autoassist-github` app id |
| `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | `az account show --query id -o tsv` |

None of these are really secret — they are identifiers, not credentials, and the
federated credential is what grants access. They are stored as secrets out of
habit and because there is no reason to publish them. They are the only three
things GitHub holds: no application key is stored there.

**Variables** (visible, and useful in logs):

| name | value |
|---|---|
| `AZURE_RESOURCE_GROUP` | e.g. `Ai_solution` |
| `AZURE_CONTAINER_APP` | e.g. `ca-autoassist` |
| `AZURE_REGISTRY` | the login server, e.g. `crautoassist.azurecr.io` |
| `TRIAGE_BACKEND` | optional: `agent` or `jev`; see below |

### 5. For the weekly evals

`.github/workflows/evals.yml` runs real questions against the agents, so it needs
more than the deploy does.

**Variables:** `PROJECT_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`, `SEARCH_ENDPOINT`
**Secrets:** `AZURE_OPENAI_API_KEY`, `SEARCH_API_KEY`

Those two secrets are no longer kept in GitHub (decision 010). The workflow's
own check looks only for the login and the endpoint variables, so it will still
start, and every search in it will fail on the missing keys. Until retrieval
uses managed identity - the next step in decision 010 - run `make evals`
locally after a prompt change.

The GitHub identity also needs to list and run agents - the same role the app
needs, on the Foundry account only:

```bash
AI_ID=$(az cognitiveservices account show -g "$RG" -n rg-autoassist --query id -o tsv)
SP_ID=$(az ad sp list --display-name autoassist-github --query "[0].id" -o tsv)
az role assignment create --assignee-object-id "$SP_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Foundry User" --scope "$AI_ID"
```

Name the account. The resource group holds other Foundry accounts, and taking
"the first AIServices account in the group" can pick the wrong one.

### 6. Optional: require an approval

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
- **No database migrations.** There is no database. Bookings are in Zoho
  Bookings (decision 008), so they survive a deploy. Tickets are still a JSON
  file in the container (`TICKET_STORE`), and a deploy loses them — a known
  limitation.
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
| `agent` (the default when the variable is unset) | the triage agent |
| `jev` (what the live app runs) | Jev, falling back to the agent for anything it cannot answer |

Jev also needs its key. The value is `typesafe-key` in Key Vault, and the app
holds a reference to it under the same name (step 1, and decision 010). The
deploy only references it, as `TYPESAFE_API_KEY=secretref:typesafe-key`, and
never writes it. That was learned the hard way on 21 September - copying the
key across from a GitHub secret needed the deploy identity to hold
`managedEnvironments/join/action`, and after that was granted it failed on a
further linked scope. Each step that writes a secret asks for more power for the
identity GitHub logs in as; referencing one needs nothing it does not already
have.

If `TRIAGE_BACKEND=jev` but the app has no `typesafe-key` secret, the deploy
stays on the agent and says so in a warning, rather than failing.

The change takes effect on the next deploy. To flip it immediately, without one:

    az containerapp update -g Ai_solution -n ca-autoassist --set-env-vars TRIAGE_BACKEND=agent

**Check it is really answering.** A fallback is silent - the customer gets a
reply either way - so a revoked key would quietly send everything back to the
agent. `/metrics` reports `triage_backend`, `jev_answered` and `jev_fell_back`;
if `jev_fell_back` is climbing, Jev is configured but not the thing deciding.
