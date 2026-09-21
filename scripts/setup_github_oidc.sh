#!/usr/bin/env bash
#
# Let GitHub Actions deploy to Azure, without storing a password.
#
# GitHub mints a short-lived token for each workflow run. Azure trusts it
# because of a "federated credential" that names this exact repository and
# branch. Nothing is stored in GitHub that could be stolen and reused, and there
# is no secret to rotate.
#
# Usage:
#     ./scripts/setup_github_oidc.sh <resource-group> <registry-name> <container-app-name>

set -euo pipefail

RG="${1:-}"; ACR="${2:-}"; APP="${3:-}"
REPO="${REPO:-kumar623/autoassist}"

if [ -z "$RG" ] || [ -z "$ACR" ] || [ -z "$APP" ]; then
  echo "Usage: ./scripts/setup_github_oidc.sh <resource-group> <registry-name> <container-app-name>"
  echo "       (registry name only, not the .azurecr.io login server)"
  exit 1
fi

SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)

echo "Repository     : $REPO"
echo "Subscription   : $SUB"
echo


echo "==> Application registration"
APP_ID=$(az ad app list --display-name autoassist-github --query "[0].appId" -o tsv)
if [ -z "$APP_ID" ]; then
  APP_ID=$(az ad app create --display-name autoassist-github --query appId -o tsv)
  az ad sp create --id "$APP_ID" -o none
  echo "    created $APP_ID"
else
  echo "    reusing $APP_ID"
fi
SP_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)


echo "==> Federated credentials"
# Two of them, because the token GitHub sends says something different
# depending on how the job runs. The deploy job declares
# `environment: production`, which makes the subject the environment. The
# weekly evals workflow (.github/workflows/evals.yml) has no environment, so its
# subject is the branch. A mismatch here is the cause of "AADSTS70021: No
# matching federated identity record found".
add_credential() {
  local name="$1" subject="$2"
  if az ad app federated-credential list --id "$APP_ID" \
       --query "[?name=='$name'] | [0].name" -o tsv | grep -q .; then
    echo "    $name already exists"
    return
  fi
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"$name\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$subject\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" -o none
  echo "    $name -> $subject"
}

add_credential "github-production" "repo:$REPO:environment:production"
add_credential "github-main"       "repo:$REPO:ref:refs/heads/main"


echo "==> Permissions"
# Two, both scoped as narrowly as they can be. This identity can push images and
# change one container app. It cannot touch the search service, the Key Vault,
# or anything else in the subscription. (The weekly evals need one more role, on
# the Foundry account only - docs/deploy.md, step 4.)
az role assignment create \
  --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role AcrPush \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/$ACR" \
  -o none 2>/dev/null || echo "    AcrPush already assigned"

az role assignment create \
  --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/$APP" \
  -o none 2>/dev/null || echo "    Contributor already assigned"


echo
echo "======================================================================"
echo "  GitHub > Settings > Secrets and variables > Actions > Secrets tab:"
echo
echo "    AZURE_CLIENT_ID       = $APP_ID"
echo "    AZURE_TENANT_ID       = $TENANT"
echo "    AZURE_SUBSCRIPTION_ID = $SUB"
echo
echo "  These are identifiers, not passwords. The federated credential is what"
echo "  grants access - nobody can use these without being GitHub Actions"
echo "  running in $REPO."
echo "======================================================================"
