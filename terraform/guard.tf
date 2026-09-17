# Fails at plan time when pointed at an account other than the expected one.
# Cheap insurance against a stale AWS_PROFILE building a tracking server in
# somebody else's account.
resource "terraform_data" "account_guard" {
  input = local.account_id

  lifecycle {
    precondition {
      condition     = var.expected_account_id == "" || var.expected_account_id == local.account_id
      error_message = "Refusing to apply: credentials resolve to account ${local.account_id}, expected ${var.expected_account_id}. Check your AWS profile."
    }

    precondition {
      condition     = !var.create_operator_policy || var.operator_principal_arn != ""
      error_message = "create_operator_policy is true but operator_principal_arn is empty (CONFIG_REQUIRED)."
    }
  }
}
