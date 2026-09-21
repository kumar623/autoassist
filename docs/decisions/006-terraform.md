# 006 - Terraform for infrastructure, describing a fresh environment

## Status
Accepted (week 3)

## Context
Every Azure resource this project runs on was created by hand, in the portal,
over about two weeks: a resource group, an AI Foundry resource, two model
deployments, a search service, Application Insights.

That worked, and it was the right way to learn what the resources are. But the
environment now exists only as a sequence of clicks nobody wrote down. If the
subscription were lost, or a second environment were needed for staging, or a
reviewer asked "how do I run this", the honest answer would be "I remember most
of it".

## Decision
Describe the whole environment in Terraform, in `infra/`.

Do **not** import the hand-created resources into Terraform state. Write the
code for a *fresh* environment and leave the current one alone.

## Why not import

Importing is possible. `terraform import` attaches an existing resource to a
state entry, and from then on Terraform manages it. Two reasons not to:

- **It is fiddly and risky.** Every imported resource has to match the written
  configuration exactly, attribute by attribute, or the next `plan` proposes
  changes to a working system. Getting an AI Services account and two model
  deployments to line up takes several rounds of plan-diff-adjust, and a
  mistake modifies live resources.
- **It demonstrates less.** The claim worth making is "this environment can be
  rebuilt from nothing". A fresh `terraform apply` into an empty resource group
  proves that. An import proves only that Terraform can be pointed at something
  that already works.

The cost of this choice is real: the code in `infra/` has never created the
environment the demo actually runs on. It has been planned, not applied. That
is stated plainly rather than glossed over, and applying it is a ten-minute job
whenever a second environment is wanted.

## Why Terraform rather than Bicep or ARM

The job ad names "Terraform or Bicep". Either would do here.

- **Bicep** is Azure-only and the better choice in an Azure-only shop: no state
  file to manage, because Azure Resource Manager is the state. Simpler.
- **Terraform** is not Azure-only, which matters if the same tooling has to
  reach GitHub, Cloudflare or another cloud, and it has an explicit plan step
  that reads as a diff.

Terraform was chosen for the plan step and the wider applicability. If this
were a real Azure-only platform team, Bicep would be an equally defensible
answer and would remove the state problem below entirely.

## State

Terraform records what it created in a state file. That file is how it knows
that `azurerm_search_service.main` means *that particular* search service.

State here is **local** - a `terraform.tfstate` next to the code, gitignored.
Correct for one person on one laptop, and wrong for a team:

- Two people running `apply` at the same time interleave writes and corrupt it.
- The file contains secrets in plain text (the search admin key, the AI
  services key - Terraform stores every attribute it read).

The remote backend for a team is written in `versions.tf`, commented out: an
Azure Storage container, which gives both shared state and blob-lease locking
so the second `apply` waits instead of colliding. Uncommenting it is the whole
change. It is commented out rather than deleted because a reviewer should be
able to see that the problem is understood, not solved by accident.

## What Terraform cannot do here

Worth being exact, because "infrastructure as code" is usually claimed as
complete and rarely is.

1. **The Foundry project and its AI Search connection.** `azurerm_cognitive_account`
   (kind `AIServices`) creates the account and the model deployments. The
   *project* inside it, and that project's connection to the search service,
   are still portal steps, listed in `terraform output next_steps`.

   (21 September 2026: the connection is no longer needed. The agents search
   through a function tool the service runs, not through a Foundry connection,
   so `next_steps` now lists only the project.)

   To be precise, because "the provider does not support it" is too broad:
   `azurerm_ai_foundry_project` exists, but it models the older **hub-based**
   Foundry project - a Machine Learning workspace under a hub. This project is
   the newer **account-based** kind, which lives inside the Cognitive Services
   account itself. Provider support for that shape was still landing when this
   was written. Worth re-checking on a provider upgrade; it is the one gap that
   is likely to close on its own.

2. **Data.** Terraform creates an empty search service. The index, its schema,
   its vectorizer and its 370 chunks come from `scripts/ingest.py`. That is
   correct - data is not infrastructure - but it means `apply` alone does not
   give a working system.

3. **The agents.** Their definitions live in `agents/definitions/*.json` and
   are deployed by `agents/deploy_agents.py`. Same reasoning: a prompt is
   application code that happens to be stored in Azure.

4. **The image.** The container app starts on a Microsoft placeholder image, and
   `lifecycle.ignore_changes` on the image means Terraform will never touch it
   again. The deploy pipeline owns it from then on. Without that block,
   infrastructure changes would roll the app back to whatever image was current
   when Terraform last ran - a genuinely surprising outage.

So: Terraform builds the environment, three scripts fill it, and the pipeline
deploys into it. Four steps, all written down, none of them clicks.

## Choices inside the code worth defending

- **No VNET.** Private endpoints and a virtual network are how a real
  production deployment of this would be locked down, and they roughly triple
  the cost and the setup time. This is a demonstration project on a personal
  subscription. Public endpoints, keys in Key Vault or environment variables,
  and managed identity where it is free. Stated as a limitation in the README
  rather than pretended away.
- **Managed identity for the container app, keys for the scripts.** The
  deployed service authenticates as a user-assigned identity with three role
  assignments (pull images, call the AI account, read the search index). The
  ingestion scripts still use keys, because they run from a laptop that has no
  managed identity.

  (21 September 2026: this said "no long-lived secret in the running system",
  and that was wrong. Managed identity covers only the Foundry agents.
  `retrieval.py` sends `api-key` headers to Azure OpenAI, for embeddings, and to
  AI Search - which is why `main.tf` puts both keys on the container app, and
  why the search-reader role above goes unused by the code. In the live
  environment those keys are in Key Vault and the app holds references
  (decision 010); moving retrieval to managed identity is the step that would
  make the original sentence true.)
- **`min_replicas = 0`.** The container app scales to zero, so an idle demo
  costs nothing. The price is a cold start of a few seconds on the first request
  after a quiet period, which is the correct trade for this and the wrong one
  for a product.
- **Random suffix on globally-unique names.** Registry, search service and AI
  subdomain names are unique across all of Azure. (This first named a storage
  account; `main.tf` created one that nothing used, and it was removed on
  21 September 2026.)
- **`prevent_deletion_if_contains_resources`.** A `destroy` that would remove
  resources Terraform did not create fails instead.
- **`purge_soft_delete_on_destroy` on cognitive accounts.** Soft delete reserves
  a deleted AI resource's name for 48 hours. For something torn down and rebuilt
  in a day, that turns the next `apply` into a confusing "name already taken"
  on a resource that is not visible anywhere.

## Revisit when

- A second environment is actually needed (then apply it, and this document's
  "never applied" caveat goes away)
- More than one person runs it (uncomment the remote backend, first)
- The azurerm provider gains a Foundry project resource (gap 1 closes)
- The project needs network isolation (VNET, private endpoints, and a real
  budget)
