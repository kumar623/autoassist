# infra

Everything AutoAssist runs on, described as code.

`terraform apply` in an empty subscription gives you the whole environment:
resource group, AI Foundry resource with two model deployments, AI Search,
storage, container registry, a container app, and Application Insights wired to
a Log Analytics workspace.

Why it is written for a *fresh* environment rather than imported from the one
that exists: `docs/decisions/006-terraform.md`.

## Files

| file | what it holds |
|---|---|
| `versions.tf` | Terraform and provider versions, provider behaviour, the commented-out remote backend |
| `variables.tf` | Everything you might want to change, with validation |
| `main.tf` | The 17 resources |
| `outputs.tf` | Endpoints, keys, and a ready-made `.env` |
| `terraform.tfvars.example` | Copy to `terraform.tfvars` and edit |

## Running it

Install Terraform (macOS, no Homebrew needed):

```bash
curl -fsSL -o /tmp/tf.zip \
  https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_darwin_arm64.zip
unzip -o /tmp/tf.zip -d ~/.local/bin
terraform version
```

Then:

```bash
az login
az account set --subscription "<your subscription>"

cd infra
cp terraform.tfvars.example terraform.tfvars   # edit if you like

terraform init      # downloads providers, writes .terraform.lock.hcl
terraform fmt       # canonical formatting
terraform validate  # syntax and type checking, no Azure calls
terraform plan      # what it WOULD do. Read this.
```

**`plan` is free and changes nothing.** `apply` creates real resources that cost
real money. Read the plan first, every time.

```bash
terraform apply     # asks for confirmation
```

Roughly ten minutes, most of it the AI Services account and the container app
environment.

## After apply

`terraform output next_steps` prints this, but in short:

1. **Two portal steps.** Terraform cannot create the Foundry *project* inside
   the AI resource, nor the project's connection to the search service. Both are
   a few clicks at <https://ai.azure.com>.
2. `terraform output -raw env_file > ../.env` — writes a complete `.env`,
   secrets included. It is gitignored. Do not paste it anywhere.
3. `make data && make reindex` — generate the bulletins, build the index.
4. `python3 agents/deploy_agents.py` — create the four agents.
5. `make evals` — 16 cases; they should all pass.

## Shutting it down

```bash
terraform destroy
```

Removes everything in the resource group. The free search tier and a
scale-to-zero container app cost approximately nothing when idle, but the AI
Services account and Log Analytics do not, so destroy it when you are done.

`purge_soft_delete_on_destroy` is on for cognitive accounts, so the name is
released immediately rather than reserved for 48 hours.

## What it costs

With the defaults (free search, scale to zero, 30-day log retention) and light
demo use:

| resource | cost |
|---|---|
| AI Search, free tier | £0 |
| Container app, idle | £0 (scales to zero) |
| Container registry, Basic | ~£4/month |
| Log Analytics + App Insights | ~£0–2/month at this volume |
| Storage | pennies |
| Model calls | pay per token; the full eval suite is a few pence |

Call it £500–900/month in rupees terms — well under ₹1,000 — provided you do not
leave a load test running. The single biggest cost risk is switching
`search_sku` to `basic`, which is about $75/month on its own.

## Things that are deliberately not here

- **No VNET or private endpoints.** Public endpoints and managed identity. A
  real production deployment would isolate the network; this is a demo on a
  personal subscription and the isolation would cost more than everything else
  combined.
- **No Key Vault for application secrets.** The container app gets its
  configuration as environment variables and its Azure access through a managed
  identity, so there is no application secret to store. The ingestion scripts
  use keys from `.env` on a laptop.
- **State is local.** Fine for one person. See `versions.tf` for the remote
  backend a team needs, and 006 for why it matters.
- **The container image is owned by the deploy pipeline**, not by Terraform.
  `lifecycle.ignore_changes` on the image is what stops infrastructure changes
  from rolling the app back.

## Common problems

**`InvalidTemplateDeployment: ... not available in region`**
The model is not offered in `location`. Check
<https://learn.microsoft.com/azure/ai-foundry/openai/concepts/models> and set
`location` to a region that has it.

**`You may only have one free search service per subscription`**
You already have one — probably the hand-made one. Either set
`search_sku = "basic"` (paid) or delete the old service.

**`StorageAccountAlreadyTaken`**
The random suffix collided, which is unlucky. `terraform taint
random_string.unique && terraform apply` rolls a new one.

**`plan` wants to change the container image every time**
The `lifecycle.ignore_changes` block is missing or mistyped. It should be
`template[0].container[0].image`.

**`Error: Insufficient quota`**
The subscription has no capacity left for the model. Lower
`chat_model.capacity` from 100, or request more quota in the portal.
