resource "aws_sagemaker_mlflow_tracking_server" "this" {
  tracking_server_name = local.name
  role_arn             = aws_iam_role.tracking_server.arn
  artifact_store_uri   = "s3://${aws_s3_bucket.artifacts.bucket}/mlflow"

  tracking_server_size            = var.tracking_server_size
  mlflow_version                  = var.mlflow_version != "" ? var.mlflow_version : null
  weekly_maintenance_window_start = var.weekly_maintenance_window_start

  # Models logged with mlflow.<flavor>.log_model land in the registry without
  # an explicit register_model call. The bake-offs still register the champion
  # explicitly so the alias is deterministic.
  automatic_model_registration = true

  depends_on = [
    aws_iam_role_policy.tracking_server_s3,
    aws_s3_bucket_public_access_block.artifacts,
    terraform_data.account_guard,
  ]
}
