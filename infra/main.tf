# AutoAssist infrastructure.
#
# Everything the system needs, in one region, in one resource group.
#
# This describes a FRESH environment. The resources the project currently runs
# on were created by hand in the portal; importing them into state is possible
# but fiddly, and the point of this file is to show the environment can be
# rebuilt from nothing. See docs/decisions/006-terraform.md.
#
#   terraform init
#   terraform plan     # read this before applying, every time
#   terraform apply
#   terraform destroy  # when you are finished, so it stops costing money

locals {
  # Every resource carries the same suffix, so dev and prod cannot collide and
  # anything belonging to this project is obvious in a shared subscription.
  suffix = "${var.name}-${var.environment}"

  # Some Azure resources reject hyphens (storage accounts, container registries)
  # and cap at 24 characters.
  compact = substr(replace(local.suffix, "-", ""), 0, 20)

  tags = merge(
    {
      project     = var.name
      environment = var.environment
      managed_by  = "terraform"
      repo        = "github.com/kumar623/autoassist"
    },
    var.tags,
  )
}

# A random suffix on globally-unique names. Storage account and registry names
# are unique across all of Azure, so "stautoassistdev" is very likely taken.
resource "random_string" "unique" {
  length  = 5
  upper   = false
  special = false
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.suffix}"
  location = var.location
  tags     = local.tags
}


# ---------------------------------------------------------------- observability
#
# Created first: everything else sends telemetry here, so it has to exist before
# the things that depend on it.

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days
  tags                = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  # Sample everything. At this volume it costs almost nothing, and a sampled-out
  # trace is exactly the one you wanted during an incident.
  sampling_percentage = 100

  # Azure defaults this to 100 GB/day. This project produces a few MB. The cap
  # is not a budget, it is a circuit breaker: a retry loop logging in anger
  # stops ingesting at 1 GB instead of billing for 100.
  daily_data_cap_in_gb = 1

  # Workspace-based App Insights actually retains for as long as the workspace
  # does, so this mirrors var.log_retention_days rather than setting its own.
  retention_in_days = var.log_retention_days

  tags = local.tags
}


# ---------------------------------------------------------------- storage

resource "azurerm_storage_account" "main" {
  name                = "st${local.compact}${random_string.unique.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = "LRS" # one region is enough for a demo; GRS costs double

  # Defaults that are off unless you ask.
  https_traffic_only_enabled      = true
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  tags = local.tags
}

# Raw documents. Week 3 moves the booking store here too, as a Table.
resource "azurerm_storage_container" "documents" {
  name                  = "documents"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}


# ---------------------------------------------------------------- AI Foundry + models

resource "azurerm_cognitive_account" "main" {
  name                = "ai-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  # "AIServices" is the multi-service account Foundry projects sit inside. Using
  # kind "OpenAI" instead would give a plain Azure OpenAI resource - models would
  # work, agents would not.
  #
  # This was azurerm_ai_services until the provider deprecated it in favour of
  # azurerm_cognitive_account with this kind. Same resource in Azure; the
  # dedicated one was feature-frozen.
  kind     = "AIServices"
  sku_name = "S0"

  # The agents authenticate as themselves rather than with a key. The key still
  # exists and the ingestion scripts use it; managed identity is what the
  # deployed service uses.
  identity {
    type = "SystemAssigned"
  }

  custom_subdomain_name = "ai-${local.suffix}-${random_string.unique.result}"

  tags = local.tags
}

resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.chat_model.name
  cognitive_account_id = azurerm_cognitive_account.main.id

  # Azure defaults to "OnceNewDefaultVersionAvailable", which silently swaps the
  # model under you when Microsoft promotes a new default. For a system whose
  # eval results are the evidence it works, the model changing without a commit
  # is the worst kind of change: nothing in git moved, and the numbers did.
  # Upgrades happen here, deliberately, by editing var.chat_model.
  version_upgrade_option = "NoAutoUpgrade"

  model {
    format  = "OpenAI"
    name    = var.chat_model.name
    version = var.chat_model.version
  }

  sku {
    name     = "GlobalStandard"
    capacity = var.chat_model.capacity
  }
}

resource "azurerm_cognitive_deployment" "embedding" {
  name                 = var.embedding_model.name
  cognitive_account_id = azurerm_cognitive_account.main.id

  # Same reasoning as the chat deployment, but worse if it happens: a changed
  # embedding model puts new questions in a different vector space from the
  # indexed documents. No error, just quietly worse retrieval.
  version_upgrade_option = "NoAutoUpgrade"

  model {
    format  = "OpenAI"
    name    = var.embedding_model.name
    version = var.embedding_model.version
  }

  sku {
    name     = "Standard"
    capacity = var.embedding_model.capacity
  }

  # Both deployments live on the same account. Azure rejects concurrent changes
  # to one account, so this one waits for the other rather than racing it.
  depends_on = [azurerm_cognitive_deployment.chat]
}


# ---------------------------------------------------------------- search

resource "azurerm_search_service" "main" {
  name                = "srch-${local.suffix}-${random_string.unique.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.search_sku

  # Free tier allows exactly one of each and rejects anything else.
  replica_count   = var.search_sku == "free" ? 1 : 1
  partition_count = var.search_sku == "free" ? 1 : 1

  # Replicas serve more queries at once; partitions hold more data. Different
  # problems, different levers - see the README.

  local_authentication_enabled = true # the ingestion scripts use the admin key

  tags = local.tags
}


# ---------------------------------------------------------------- container registry

resource "azurerm_container_registry" "main" {
  name                = "cr${local.compact}${random_string.unique.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"

  # Container Apps pulls with a managed identity, so no admin user is needed.
  admin_enabled = false

  tags = local.tags
}


# ---------------------------------------------------------------- container apps

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${local.suffix}"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

resource "azurerm_user_assigned_identity" "app" {
  name                = "id-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

# The identity needs to pull images and to call the AI project. Role assignments
# rather than keys: nothing long-lived to leak, and access can be revoked in one
# place.
resource "azurerm_role_assignment" "app_pulls_images" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_role_assignment" "app_uses_ai" {
  scope                = azurerm_cognitive_account.main.id
  role_definition_name = "Cognitive Services User"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_role_assignment" "app_reads_search" {
  scope                = azurerm_search_service.main.id
  role_definition_name = "Search Index Data Reader"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_container_app" "orchestrator" {
  name                         = "ca-${local.suffix}"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.app.id
  }

  template {
    # Scales to zero. An idle demo costs nothing, at the price of a cold start
    # on the first request after a quiet period.
    min_replicas = 0
    max_replicas = 3

    container {
      name = "orchestrator"

      # Until the pipeline pushes a real image, run a placeholder so the
      # environment is complete and testable. The pipeline replaces this.
      image = var.container_image != "" ? var.container_image : "mcr.microsoft.com/k8se/quickstart:latest"

      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "PROJECT_ENDPOINT"
        value = "https://${azurerm_cognitive_account.main.custom_subdomain_name}.services.ai.azure.com/api/projects/${var.name}"
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = azurerm_cognitive_account.main.endpoint
      }
      env {
        name  = "SEARCH_ENDPOINT"
        value = "https://${azurerm_search_service.main.name}.search.windows.net"
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
      env {
        name  = "AZURE_CLIENT_ID" # tells DefaultAzureCredential which identity to use
        value = azurerm_user_assigned_identity.app.client_id
      }
      env {
        name  = "CHAT_DEPLOYMENT"
        value = azurerm_cognitive_deployment.chat.name
      }
      env {
        name  = "EMBED_DEPLOYMENT"
        value = azurerm_cognitive_deployment.embedding.name
      }

      # Liveness, not readiness. A readiness failure means "do not send me
      # traffic"; restarting on it would turn a brief Azure blip into a
      # restart loop across every replica.
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"

        initial_delay           = 10
        interval_seconds        = 30
        timeout                 = 5
        failure_count_threshold = 3
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/ready"

        interval_seconds        = 10
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  lifecycle {
    ignore_changes = [
      # The deploy pipeline updates the image. Terraform should not fight it and
      # roll back to whatever was current when infrastructure last ran.
      template[0].container[0].image,
    ]
  }
}
