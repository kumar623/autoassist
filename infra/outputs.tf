# What you need after `terraform apply`. Secrets are marked sensitive, so they
# are hidden in the plan output and in CI logs; read them with
# `terraform output -raw <name>` when you actually need one.

output "resource_group" {
  description = "Everything lives here. Delete this to delete the lot."
  value       = azurerm_resource_group.main.name
}

output "app_url" {
  description = "The chat page."
  value       = "https://${azurerm_container_app.orchestrator.ingress[0].fqdn}"
}

output "project_endpoint" {
  description = "PROJECT_ENDPOINT for .env"
  value       = "https://${azurerm_cognitive_account.main.custom_subdomain_name}.services.ai.azure.com/api/projects/${var.name}"
}

output "azure_openai_endpoint" {
  description = "AZURE_OPENAI_ENDPOINT for .env"
  value       = azurerm_cognitive_account.main.endpoint
}

output "search_endpoint" {
  description = "SEARCH_ENDPOINT for .env"
  value       = "https://${azurerm_search_service.main.name}.search.windows.net"
}

output "container_registry" {
  description = "Where the deploy pipeline pushes images."
  value       = azurerm_container_registry.main.login_server
}

output "azure_openai_key" {
  description = "AZURE_OPENAI_API_KEY for .env"
  value       = azurerm_cognitive_account.main.primary_access_key
  sensitive   = true
}

output "search_admin_key" {
  description = "SEARCH_API_KEY for .env - admin, because ingestion writes"
  value       = azurerm_search_service.main.primary_key
  sensitive   = true
}

output "appinsights_connection_string" {
  description = "APPLICATIONINSIGHTS_CONNECTION_STRING for .env"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "env_file" {
  description = <<-EOT
    A ready-made .env, secrets included. Write it out with:
      terraform output -raw env_file > ../.env
  EOT
  sensitive   = true
  value       = <<-EOT
    PROJECT_ENDPOINT=https://${azurerm_cognitive_account.main.custom_subdomain_name}.services.ai.azure.com/api/projects/${var.name}
    AZURE_OPENAI_ENDPOINT=${azurerm_cognitive_account.main.endpoint}
    AZURE_OPENAI_API_KEY=${azurerm_cognitive_account.main.primary_access_key}
    AZURE_OPENAI_API_VERSION=2025-04-01-preview

    CHAT_DEPLOYMENT=${azurerm_cognitive_deployment.chat.name}
    EMBED_DEPLOYMENT=${azurerm_cognitive_deployment.embedding.name}

    SEARCH_ENDPOINT=https://${azurerm_search_service.main.name}.search.windows.net
    SEARCH_API_KEY=${azurerm_search_service.main.primary_key}
    SEARCH_INDEX_NAME=service-docs

    APPLICATIONINSIGHTS_CONNECTION_STRING=${azurerm_application_insights.main.connection_string}
    AZURE_LOG_LEVEL=WARNING
  EOT
}

output "next_steps" {
  description = "What to do once this has applied."
  value       = <<-EOT

    Infrastructure is up. Now:

      1. One portal step Terraform cannot do (see
         docs/decisions/006-terraform.md): create a project named
         "${var.name}" inside the AI resource ai-${local.suffix}, at
         https://ai.azure.com
      2. terraform output -raw env_file > ../.env
      3. cd .. && make data CONFIRM=1 && make reindex CONFIRM=1
         Only because this search service is new and empty. Check .env
         points at srch-${local.suffix}-${random_string.unique.result} first.
      4. python3 agents/deploy_agents.py
      5. make evals

    App URL (placeholder image until the pipeline deploys):
      https://${azurerm_container_app.orchestrator.ingress[0].fqdn}

    When you are finished: terraform destroy
  EOT
}
