#!/usr/bin/env bash
#
# Create the two things the deploy pipeline needs: a container registry to push
# images to, and a container app to deploy them to.
#
# Everything else - the Foundry resource, the models, AI Search, Application
# Insights - already exists and is reused. This adds to your environment; it
# changes nothing that is already there.
#
# Usage:
#     ./scripts/setup_deploy_target.sh <resource-group>
#
# Run it from the repo root. It reads .env for the endpoints and keys, so those
# values never have to be typed or pasted anywhere.
#
# Cost: the registry is about 400 rupees a month. The container app scales to
# zero and costs nothing while idle.

set -euo pipefail

RG="${1:-}"
if [ -z "$RG" ]; then
  echo "Usage: ./scripts/setup_deploy_target.sh <resource-group>"
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

for required in PROJECT_ENDPOINT AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY \
                SEARCH_ENDPOINT SEARCH_API_KEY SEARCH_INDEX_NAME; do
  if [ -z "${!required:-}" ]; then
    echo "$required is missing from .env" >&2
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
# DefaultAzureCredential picks up inside the container.
AI_ID=$(az cognitiveservices account list -g "$RG" \
          --query "[?kind=='AIServices'] | [0].id" -o tsv)
if [ -n "$AI_ID" ]; then
  az role assignment create \
    --assignee-object-id "$UAI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --role "Cognitive Services User" --scope "$AI_ID" -o none
  echo "    granted Cognitive Services User on the AI resource"
else
  echo "    WARNING: no AIServices account found in $RG - grant this by hand"
fi

# Role assignments take a moment to reach the service that checks them. Creating
# the app immediately can fail an image pull that would work a minute later.
echo "    waiting 60s for permissions to take effect"
sleep 60


echo "==> Container app environment"
az containerapp env create -g "$RG" -n "$CAE" --location "$LOCATION" -o none


echo "==> Container app"
# Secrets go in as container app secrets and are referenced by name, so they do
# not appear in `az containerapp show` output or in the portal's env var list.
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
  --secrets "openai-key=$AZURE_OPENAI_API_KEY" "search-key=$SEARCH_API_KEY" \
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
