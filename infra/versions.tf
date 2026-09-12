# Pinned. An unpinned provider means `terraform init` on a different day builds
# something different, which defeats most of the point of infrastructure as code.
terraform {
  required_version = "~> 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.14"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # State is local by default, which is fine for one person on one laptop.
  # A team needs remote state with locking, or two people running apply at the
  # same time corrupt it. See docs/decisions/006-terraform.md.
  #
  # backend "azurerm" {
  #   resource_group_name  = "rg-tfstate"
  #   storage_account_name = "sttfstateautoassist"
  #   container_name       = "tfstate"
  #   key                  = "autoassist.tfstate"
  # }
}

provider "azurerm" {
  features {
    cognitive_account {
      # Soft delete keeps a deleted AI resource's name reserved for 48 hours.
      # For a project that gets torn down and rebuilt, that means the next
      # apply fails with "name already taken" on a resource you cannot see.
      purge_soft_delete_on_destroy = true
    }
    resource_group {
      # Refuse to delete a resource group that still has things in it. Without
      # this, one wrong `destroy` takes resources Terraform never created.
      prevent_deletion_if_contains_resources = true
    }
  }
}
