data "aws_iam_policy_document" "tracking_server_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "tracking_server" {
  name        = "${local.name}-tracking-server"
  description = "Role the managed MLflow tracking server assumes to reach its artifact store."

  assume_role_policy = data.aws_iam_policy_document.tracking_server_trust.json
}

# Scoped to this stack's bucket only -- the tracking server has no reason to
# read anything else in the account.
data "aws_iam_policy_document" "tracking_server_s3" {
  statement {
    sid    = "BucketLevel"
    effect = "Allow"

    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation",
      "s3:ListBucketMultipartUploads",
    ]

    resources = [aws_s3_bucket.artifacts.arn]
  }

  statement {
    sid    = "ObjectLevel"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]

    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "tracking_server_s3" {
  name   = "artifact-store-access"
  role   = aws_iam_role.tracking_server.id
  policy = data.aws_iam_policy_document.tracking_server_s3.json
}

# --- optional operator access -------------------------------------------------
# Only created when the calling principal is not already allowed to talk to the
# tracking server (see `make check`).

data "aws_iam_policy_document" "operator" {
  count = var.create_operator_policy ? 1 : 0

  statement {
    sid    = "MlflowDataPlane"
    effect = "Allow"

    actions = ["sagemaker-mlflow:*"]

    resources = [aws_sagemaker_mlflow_tracking_server.this.arn]
  }

  statement {
    sid    = "MlflowControlPlane"
    effect = "Allow"

    actions = [
      "sagemaker:DescribeMlflowTrackingServer",
      "sagemaker:ListMlflowTrackingServers",
      "sagemaker:CreatePresignedMlflowTrackingServerUrl",
      "sagemaker:StartMlflowTrackingServer",
      "sagemaker:StopMlflowTrackingServer",
    ]

    resources = [aws_sagemaker_mlflow_tracking_server.this.arn]
  }
}

resource "aws_iam_policy" "operator" {
  count = var.create_operator_policy ? 1 : 0

  name        = "${local.name}-operator"
  description = "MLflow access to the ${local.name} tracking server."
  policy      = data.aws_iam_policy_document.operator[0].json
}

locals {
  operator_is_role = var.create_operator_policy ? can(regex(":role/", var.operator_principal_arn)) : false
  operator_name    = var.create_operator_policy ? element(split("/", var.operator_principal_arn), length(split("/", var.operator_principal_arn)) - 1) : ""
}

resource "aws_iam_role_policy_attachment" "operator_role" {
  count = var.create_operator_policy && local.operator_is_role ? 1 : 0

  role       = local.operator_name
  policy_arn = aws_iam_policy.operator[0].arn
}

resource "aws_iam_user_policy_attachment" "operator_user" {
  count = var.create_operator_policy && !local.operator_is_role ? 1 : 0

  user       = local.operator_name
  policy_arn = aws_iam_policy.operator[0].arn
}

# automatic_model_registration mirrors MLflow registered models into the
# SageMaker Model Registry, which the tracking server does under its own role.
# Without these the mirror fails and mlflow.register_model raises, so the model
# never lands in the MLflow registry either.
data "aws_iam_policy_document" "tracking_server_registry" {
  statement {
    sid    = "ModelPackageGroups"
    effect = "Allow"

    actions = [
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:DeleteModelPackageGroup",
      "sagemaker:ListModelPackages",
    ]

    resources = ["arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package-group/*"]
  }

  statement {
    sid    = "ModelPackages"
    effect = "Allow"

    actions = [
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DeleteModelPackage",
    ]

    resources = ["arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package/*"]
  }

  statement {
    sid    = "Tagging"
    effect = "Allow"

    actions = [
      "sagemaker:AddTags",
      "sagemaker:ListTags",
      "sagemaker:DeleteTags",
    ]

    resources = [
      "arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package-group/*",
      "arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package/*",
    ]
  }
}

resource "aws_iam_role_policy" "tracking_server_registry" {
  name   = "model-registry-mirror"
  role   = aws_iam_role.tracking_server.id
  policy = data.aws_iam_policy_document.tracking_server_registry.json
}
