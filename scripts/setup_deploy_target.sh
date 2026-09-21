#!/usr/bin/env bash
#
# Create the two things the deploy pipeline needs: a container registry to push
# images to, and a container app to deploy them to.
#
# Everything else - the Foundry resource, the models, AI Search, Application
# Insights - already exists and is reused. This adds to your environment; it
# changes nothing that is already there.
#
# BEFORE running it, by hand (docs/deploy.md, step 1): create the Key Vault,
# give yourself "Key Vault Secrets Officer" on it, and add the secrets. This
# script checks that openai-key and search-key are there, then gives the app's
# identity "Key Vault Secrets User" on the vault and points the app at them.
# The app never holds a key value, only the vault address of one.
#
# Usage:
#     KEY_VAULT_NAME=<vault> ./scripts/setup_deploy_target.sh <resource-group>
#
# Optional: AI_ACCOUNT (the Foundry account; default rg-autoassist), LOCATION.
#
# Run it from the repo root. It reads .env for the endpoints; the keys come
# from the vault, so they never have to be typed or pasted anywhere.
#
# Cost: the registry is about 400 rupees a month. The container app scales to
# zero and costs nothing while idle.

set -euo pipefail

RG="${1:-}"
KV="${KEY_VAULT_NAME:-}"
# Named, not "the first AIServices account in the group": the resource group
# holds more than one Foundry account, and picking the wrong one grants the
# role somewhere the app never calls.
AI_ACCOUNT="${AI_ACCOUNT:-rg-autoassist}"

if [ -z "$RG" ] || [ -z "$KV" ]; then
  echo "Usage: KEY_VAULT_NAME=<vault> ./scripts/setup_deploy_target.sh <resource-group>"
  echo
  echo "Your resource groups:"
  az group list --query "[].name" -o tsv | sed 's/^/  /'
  exit 1
fi

if [ ! -f .env ]; then
  echo "No .env in this directory. Run this from the repo root." >&2
  exit 1
fi

# Load .env without printing it.
set -a
# shellcheck disable=SC1091
source .env
set +a

for required in PROJECT_ENDPOINT AZURE_OPENAI_ENDPOINT SEARCH_ENDPOINT SEARCH_INDEX_NAME; do
  if [ -z "${!required:-}" ]; then
    echo "$required is missing from .env" >&2
    exit 1
  fi
done

# The vault and its secrets come first; see the top of this file. `list` shows
# names only - `show` would return the value.
KV_ID=$(az keyvault show -n "$KV" --query id -o tsv)
KV_URI=$(az keyvault show -n "$KV" --query properties.vaultUri -o tsv)
KV_URI="${KV_URI%/}"
HAVE=$(az keyvault secret list --vault-name "$KV" --query "[].name" -o tsv)
for secret in openai-key search-key; do
  if ! grep -qx "$secret" <<< "$HAVE"; then
    echo "Key Vault $KV has no '$secret' secret. Add it first (docs/deploy.md)." >&2
    exit 1
  fi
done

LOCATION="${LOCATION:-$(az group show -g "$RG" --query location -o tsv)}"

# Registry names must be unique across all of Azure and allow no hyphens, so a
# random suffix is not optional.
SUFFIX=$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')
ACR="crautoassist${SUFFIX}"
CAE="cae-autoassist"
APP="ca-autoassist"
UAI="id-autoassist"

echo "Resource group : $RG"
echo "Location       : $LOCATION"
echo "Key Vault      : $KV"
echo "AI account     : $AI_ACCOUNT"
echo "Registry       : $ACR"
echo "Container app  : $APP"
echo
read -r -p "Create these? [y/N] " ok
[ "$ok" = "y" ] || { echo "Nothing created."; exit 0; }


echo
echo "==> Container registry"
# admin_enabled stays off. The app pulls with a managed identity and the
# pipeline pushes with a federated credential, so no registry password exists
# anywhere to be leaked.
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled false -o none
ACR_ID=$(az acr show -g "$RG" -n "$ACR" --query id -o tsv)
ACR_SERVER=$(az acr show -g "$RG" -n "$ACR" --query loginServer -o tsv)


echo "==> Building the image in Azure"
# az acr build uploads the source and builds it on Azure's machines, so this
# works whether or not Docker is installed locally. It also proves the
# Dockerfile builds before anything depends on it.
az acr build --registry "$ACR" --image "autoassist:bootstrap" . -o none


echo "==> Identity for the app"
az identity create -g "$RG" -n "$UAI" -o none
UAI_ID=$(az identity show -g "$RG" -n "$UAI" --query id -o tsv)
UAI_CLIENT=$(az identity show -g "$RG" -n "$UAI" --query clientId -o tsv)
UAI_PRINCIPAL=$(az identity show -g "$RG" -n "$UAI" --query principalId -o tsv)


echo "==> Permissions"
# Pull images.
az role assignment create \
  --assignee-object-id "$UAI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope "$ACR_ID" -o none

# Call the Foundry project as itself rather than with a key. This is what
# DefaultAzureCredential picks up inside the container. "Foundry User", not
# "Cognitive Services User": that one covers model calls, and with it the app
# authenticates and then gets back an EMPTY list of agents (infra/main.tf).
AI_ID=$(az cognitiveservices account show -g "$RG" -n "$AI_ACCOUNT" --query id -o tsv)
az role assignment create \
  --assignee-object-id "$UAI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Foundry User" --scope "$AI_ID" -o none
echo "    granted Foundry User on $AI_ACCOUNT"

# Read the keys, and nothing else: not list them, not change them.
az role assignment create \
  --assignee-object-id "$UAI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$KV_ID" -o none
echo "    granted Key Vault Secrets User on $KV"

# Role assignments take a moment to reach the service that checks them. Creating
# the app immediately can fail an image pull that would work a minute later.
echo "    waiting 60s for permissions to take effect"
sleep 60


echo "==> Container app environment"
az containerapp env create -g "$RG" -n "$CAE" --location "$LOCATION" -o none


echo "==> Container app"
# The app's secrets are Key Vault REFERENCES, resolved by its identity. Versionless
# URLs, so a new version in the vault reaches the app without a deploy
# (docs/decisions/010). The environment variables name the secrets, so no value
# appears in `az containerapp show` or in the portal's env var list.
az containerapp create \
  -g "$RG" -n "$APP" \
  --environment "$CAE" \
  --image "$ACR_SERVER/autoassist:bootstrap" \
  --registry-server "$ACR_SERVER" \
  --registry-identity "$UAI_ID" \
  --user-assigned "$UAI_ID" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 0.5 --memory 1Gi \
  --secrets \
      "openai-key=keyvaultref:$KV_URI/secrets/openai-key,identityref:$UAI_ID" \
      "search-key=keyvaultref:$KV_URI/secrets/search-key,identityref:$UAI_ID" \
  --env-vars \
      "PROJECT_ENDPOINT=$PROJECT_ENDPOINT" \
      "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
      "AZURE_OPENAI_API_KEY=secretref:openai-key" \
      "AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2025-04-01-preview}" \
      "EMBED_DEPLOYMENT=${EMBED_DEPLOYMENT:-text-embedding-3-small}" \
      "SEARCH_ENDPOINT=$SEARCH_ENDPOINT" \
      "SEARCH_API_KEY=secretref:search-key" \
      "SEARCH_INDEX_NAME=$SEARCH_INDEX_NAME" \
      "APPLICATIONINSIGHTS_CONNECTION_STRING=${APPLICATIONINSIGHTS_CONNECTION_STRING:-}" \
      "AZURE_CLIENT_ID=$UAI_CLIENT" \
      "AZURE_LOG_LEVEL=WARNING" \
      "GIT_SHA=bootstrap" \
  -o none

FQDN=$(az containerapp show -g "$RG" -n "$APP" \
         --query "properties.configuration.ingress.fqdn" -o tsv)


echo
echo "==> Waiting for it to answer"
for i in $(seq 1 36); do
  if curl -sf --max-time 10 "https://$FQDN/health" > /dev/null; then
    echo "    healthy after $((i * 5))s"
    break
  fi
  [ "$i" = "36" ] && echo "    never became healthy - check: az containerapp logs show -g $RG -n $APP --tail 50"
  sleep 5
done

echo
echo "    /health:"
curl -s --max-time 10 "https://$FQDN/health" || true
echo
echo "    /ready:"
curl -s --max-time 30 "https://$FQDN/ready" || true
echo
echo
echo "======================================================================"
echo "  Live at: https://$FQDN"
echo
echo "  Put these in GitHub > Settings > Secrets and variables > Actions,"
echo "  under the Variables tab:"
echo
echo "    AZURE_RESOURCE_GROUP = $RG"
echo "    AZURE_CONTAINER_APP  = $APP"
echo "    AZURE_REGISTRY       = $ACR_SERVER"
echo
echo "  Then run scripts/setup_github_oidc.sh $RG $ACR $APP"
echo "======================================================================"
