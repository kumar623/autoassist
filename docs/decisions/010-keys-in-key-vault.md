# 010 - Keys live in Key Vault; the container app holds only references

## Status
Accepted (week 4). 21 September 2026.

## Context
The running service still needs six secret values:

| Secret | What it is | Used by |
|---|---|---|
| `openai-key` | Azure OpenAI key | `retrieval.py`, to embed each question |
| `search-key` | AI Search key | `retrieval.py` and the Library tab |
| `zoho-mcp-url` | Zoho's MCP address, which contains an access key | `zoho_bookings.py`, `zoho_auth.py` |
| `zoho-client-id` | Zoho OAuth client | `zoho_auth.py` |
| `zoho-refresh-token` | Zoho OAuth refresh token | `zoho_auth.py` |
| `typesafe-key` | TypeSafe key, for Jev triage | `typesafe.py` |

Managed identity covers only the Foundry agents. The first two keys are sent as
`api-key` headers on every search.

They were container app secrets. Those are hidden from `az containerapp show`
and from the portal's environment variable list, but not from anyone allowed to
call `listSecrets` on the app - and the identity GitHub deploys as has Container
Apps Contributor on the app, which includes that action. Every key the app used
was one API call away from any workflow that logs in as that identity.

## Decision
The values live in Key Vault `kv-autoassist-kk` (resource group `Ai_solution`,
RBAC permission model, soft delete on, purge protection off). The container app
holds six Key Vault **references** under the same secret names, so the code and
the environment variables did not change:

    openai-key  ->  keyvaultref:https://kv-autoassist-kk.vault.azure.net/secrets/openai-key,
                    identityref:<id-autoassist>

The app's user-assigned identity, `id-autoassist`, resolves them. Who can do what:

| Identity | Role on the vault | Can |
|---|---|---|
| `id-autoassist` (the app) | Key Vault Secrets User | read the values |
| the owner | Key Vault Secrets Officer | add and replace values |
| `autoassist-github` (the deploy) | none | nothing: `listSecrets` on the app now returns vault addresses |

The references are **versionless**, so rotating a key is adding a new version in
the vault. Container Apps picks it up within 30 minutes - 12 minutes when it was
first tried - and restarts the revision. No deploy, no pipeline, no GitHub
secret. The steps are in `docs/runbook.md`, "Rotating a key".

The deploy workflow still never writes a secret. It references `typesafe-key`
through `secretref` when `TRIAGE_BACKEND=jev` and the app has that secret, which
works the same whether the secret is a value or a reference. GitHub holds only
the three OIDC identifiers it logs in with, none of which is a credential.

## Rejected
- **The app reads Key Vault itself at startup**, through the Key Vault SDK. The
  same result for more cost: a code change, a new package in the image, a
  second sign-in path to get right, and a new way for startup to fail. Container
  Apps references need no code at all.
- **Keys kept in GitHub and written into the app on deploy.** Tried for the
  TypeSafe key on 21 September. Writing a secret needed the deploy identity to
  gain `managedEnvironments/join/action`, then failed on a further scope. Every
  secret-writing step asks for more power for the identity every push to main
  runs as, and the key still ended up readable through `listSecrets`.
- **Leaving them as container app secrets.** The `listSecrets` exposure above.

## Consequences
- The deploy identity cannot read a key through Azure's API any more. It can
  still deploy an image, and an image runs with the keys in its environment -
  which is why only CI on `main` deploys (`docs/deploy.md`). The vault narrows
  who can read a key; it does not make the code that uses one trustworthy.
- Key Vault stores whatever it is given. On 21 September a portal paste put page
  text into `openai-key`; httpx refused to send it as a header, and search failed
  on every message until a new version was added. The fix took no deploy, which
  is the design working, but the vault did nothing to stop the bad value. The
  runbook now says how to check a value's shape without printing it.
- A rotated key can take up to half an hour to reach the app. Revoking the old
  key before the app has the new one would mean an outage of up to that long.
- One more thing that must work before the app can start: a reference that
  does not resolve can keep a revision from starting (runbook, "Key Vault
  reference not resolving").
- Zoho's refresh token is still read-only to the app. Decision 008's limitation
  stands: a token Zoho rotates is kept only in memory.

## Next
Managed identity for Azure OpenAI and AI Search, so `openai-key` and
`search-key` stop existing rather than being stored better. That is a change in
`retrieval.py` - bearer tokens from the credential the app already has, instead
of `api-key` headers - and two role assignments for `id-autoassist`. The index's
own vectorizer holds a key too (README, known limitations) and would move with it.
