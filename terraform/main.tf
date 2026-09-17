provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  # Every resource in this stack carries these. Teardown refuses to touch
  # anything that does not.
  tags = {
    Project   = var.project
    Ephemeral = "true"
    ManagedBy = "terraform"
    CreatedBy = "jreq-mlflow"
  }

  name        = "${var.project}-${random_id.suffix.hex}"
  bucket_name = "${var.project}-artifacts-${data.aws_caller_identity.current.account_id}-${random_id.suffix.hex}"
  account_id  = data.aws_caller_identity.current.account_id
  partition   = data.aws_partition.current.partition
}
