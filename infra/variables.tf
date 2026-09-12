variable "name" {
  description = "Short name used as a prefix for every resource."
  type        = string
  default     = "autoassist"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,14}$", var.name))
    error_message = "Lowercase letters, digits and hyphens; 3-15 characters; must start with a letter."
  }
}

variable "environment" {
  description = "Environment name. Part of every resource name, so dev and prod cannot collide."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Must be dev, staging or prod."
  }
}

variable "location" {
  description = <<-EOT
    Azure region. Everything goes in one region: cross-region calls add latency
    and data transfer charges for no benefit at this size.

    Must support Azure OpenAI and the models below. South India works and is
    closest to Vizag; Sweden Central has the widest model catalogue.
  EOT
  type        = string
  default     = "southindia"
}

variable "chat_model" {
  description = "Model for all four agents."
  type = object({
    name     = string
    version  = string
    capacity = number # thousands of tokens per minute
  })
  default = {
    name     = "gpt-4.1-mini"
    version  = "2025-04-14"
    capacity = 100
  }
}

variable "embedding_model" {
  description = <<-EOT
    Model for embeddings. MUST match what was used at ingestion.

    If documents were embedded with one model and questions with another, they
    land in different vector spaces and retrieval degrades silently - no error,
    just worse answers. Changing this means re-indexing everything.
  EOT
  type = object({
    name     = string
    version  = string
    capacity = number
  })
  default = {
    name     = "text-embedding-3-small"
    version  = "1"
    capacity = 120
  }
}

variable "search_sku" {
  description = <<-EOT
    Azure AI Search tier.

    "free" costs nothing and holds 50MB, which fits ~370 chunks with room to
    spare. It does NOT support the semantic reranker - see
    docs/decisions/004-no-semantic-ranker.md. Also only one free search service
    is allowed per subscription.

    "basic" is the first tier with a dedicated machine and the paid semantic
    plan, at roughly $75/month.
  EOT
  type        = string
  default     = "free"

  validation {
    condition     = contains(["free", "basic", "standard"], var.search_sku)
    error_message = "Must be free, basic or standard."
  }
}

variable "log_retention_days" {
  description = "Log Analytics retention. 30 days is the free floor; longer costs per GB."
  type        = number
  default     = 30
}

variable "container_image" {
  description = "Image for the orchestrator. Left as a placeholder until the pipeline pushes a real one."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Extra tags, merged with the defaults below."
  type        = map(string)
  default     = {}
}
