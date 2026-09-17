variable "aws_profile" {
  description = "Named AWS CLI profile used for every call in this stack."
  type        = string
  default     = "spectro"
}

variable "region" {
  description = "Region the tracking server and artifact bucket live in."
  type        = string
  default     = "us-east-2"
}

variable "project" {
  description = "Name prefix and value of the Project tag used by teardown."
  type        = string
  default     = "jreq-mlflow"
}

variable "expected_account_id" {
  description = <<-EOT
    Account the stack is allowed to build in. Left empty the guard is skipped;
    set it and a plan against any other account fails before anything is created.
  EOT
  type        = string
  default     = ""
}

variable "tracking_server_size" {
  description = "Small | Medium | Large. Small is ~$0.642/hr while running."
  type        = string
  default     = "Small"

  validation {
    condition     = contains(["Small", "Medium", "Large"], var.tracking_server_size)
    error_message = "tracking_server_size must be Small, Medium or Large."
  }
}

variable "mlflow_version" {
  description = <<-EOT
    MLflow version for the managed tracking server, e.g. "3.0". Leave empty to
    take whatever the service defaults to for this region.
  EOT
  type        = string
  default     = ""
}

variable "weekly_maintenance_window_start" {
  description = "DAY:HH:MM in UTC, e.g. Sun:03:30."
  type        = string
  default     = "Sun:03:30"
}

variable "artifact_retention_days" {
  description = "Days before objects under the mlflow/ prefix expire."
  type        = number
  default     = 30
}

variable "force_destroy" {
  description = <<-EOT
    Allow `terraform destroy` to delete a non-empty artifact bucket. Default
    false so a stray destroy cannot take the evidence with it; the teardown
    role syncs artifacts down before it passes this.
  EOT
  type        = bool
  default     = false
}

variable "create_operator_policy" {
  description = <<-EOT
    Create and attach a policy granting the calling principal MLflow access to
    the tracking server. Only needed when the caller is not already an admin;
    `make check` reports which case applies.
  EOT
  type        = bool
  default     = false
}

variable "operator_principal_arn" {
  description = "IAM user or role ARN to attach the operator policy to. Required when create_operator_policy is true."
  type        = string
  default     = ""
}
