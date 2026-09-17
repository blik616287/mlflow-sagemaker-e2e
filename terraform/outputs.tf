output "account_id" {
  description = "Account the stack was built in."
  value       = local.account_id
}

output "region" {
  description = "Region the stack was built in."
  value       = var.region
}

output "tracking_server_name" {
  description = "Name used by the aws sagemaker *-mlflow-tracking-server calls."
  value       = aws_sagemaker_mlflow_tracking_server.this.tracking_server_name
}

output "tracking_server_arn" {
  description = "Value of MLFLOW_TRACKING_URI for the sagemaker-mlflow plugin."
  value       = aws_sagemaker_mlflow_tracking_server.this.arn
}

output "tracking_server_url" {
  description = "Service endpoint of the tracking server (the UI still needs a presigned URL)."
  value       = aws_sagemaker_mlflow_tracking_server.this.tracking_server_url
}

output "artifact_bucket" {
  description = "S3 bucket backing the MLflow artifact store."
  value       = aws_s3_bucket.artifacts.bucket
}

output "artifact_store_uri" {
  description = "Artifact store prefix handed to the tracking server."
  value       = aws_sagemaker_mlflow_tracking_server.this.artifact_store_uri
}

output "tracking_server_role_arn" {
  description = "Role the tracking server assumes."
  value       = aws_iam_role.tracking_server.arn
}
