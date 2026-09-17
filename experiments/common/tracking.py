"""Shared MLflow wiring for the jreq bake-offs.

The tracking server is SageMaker-managed, so the tracking URI is the server
ARN and every request is SigV4-signed by the ``sagemaker-mlflow`` plugin. The
only thing callers need to get right is that ``AWS_PROFILE``/region resolve to
the account that owns the server.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_DIR = REPO_ROOT / "terraform"


class ConfigRequired(RuntimeError):
    """A prerequisite is missing. Never downgraded to a warning or a skip."""


@dataclass(frozen=True)
class TrackingTarget:
    uri: str
    region: str
    profile: str | None
    source: str

    def describe(self) -> str:
        who = self.profile or "<ambient credentials>"
        return f"{self.uri} (region={self.region}, profile={who}, from {self.source})"


def _terraform_outputs() -> dict:
    """Read stack outputs. Returns {} when no state exists yet."""
    if not (TERRAFORM_DIR / "terraform.tfstate").exists():
        return {}
    try:
        raw = subprocess.run(
            ["terraform", f"-chdir={TERRAFORM_DIR}", "output", "-json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {k: v.get("value") for k, v in parsed.items()}


def resolve_target() -> TrackingTarget:
    """Environment first, Terraform state second, then fail loudly."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return TrackingTarget(
            uri=env_uri,
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or _region_from_arn(env_uri) or "us-east-2",
            profile=os.environ.get("AWS_PROFILE"),
            source="MLFLOW_TRACKING_URI",
        )

    outputs = _terraform_outputs()
    arn = outputs.get("tracking_server_arn")
    if not arn:
        raise ConfigRequired(
            "CONFIG_REQUIRED: no MLFLOW_TRACKING_URI in the environment and no "
            "tracking_server_arn in terraform state. Run `make up EXECUTE=--execute` first, "
            "or export MLFLOW_TRACKING_URI=<tracking server arn>."
        )
    return TrackingTarget(
        uri=arn,
        region=outputs.get("region") or _region_from_arn(arn) or "us-east-2",
        profile=os.environ.get("AWS_PROFILE") or "spectro",
        source="terraform output",
    )


def _region_from_arn(arn: str) -> str | None:
    parts = arn.split(":")
    return parts[3] if len(parts) > 4 and parts[0] == "arn" else None


def connect(experiment_name: str) -> TrackingTarget:
    """Point MLflow at the managed server and select/create the experiment."""
    target = resolve_target()

    if target.profile:
        os.environ.setdefault("AWS_PROFILE", target.profile)
    os.environ.setdefault("AWS_DEFAULT_REGION", target.region)
    os.environ.setdefault("AWS_REGION", target.region)

    mlflow.set_tracking_uri(target.uri)
    mlflow.set_registry_uri(target.uri)
    mlflow.set_experiment(experiment_name)
    return target


@contextmanager
def timed(into: dict, key: str):
    """Record wall-clock seconds for a block into ``into[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        into[key] = round(time.perf_counter() - start, 4)


def wait_until_ready(server_name: str, profile: str, region: str, timeout_s: int = 3600) -> str:
    """Block until a tracking server reports Created. Returns its final status."""
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    sm = session.client("sagemaker")
    deadline = time.time() + timeout_s
    status = "Unknown"

    while time.time() < deadline:
        status = sm.describe_mlflow_tracking_server(TrackingServerName=server_name)["TrackingServerStatus"]
        if status == "Created":
            return status
        if status in {"CreateFailed", "DeleteFailed", "UpdateFailed", "Stopped"}:
            raise RuntimeError(f"tracking server {server_name} entered terminal status {status}")
        time.sleep(20)

    raise TimeoutError(f"tracking server {server_name} still {status} after {timeout_s}s")
