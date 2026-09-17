#!/usr/bin/env python3
"""Delete every object version and delete-marker in a bucket.

Terraform's force_destroy only empties a bucket if the provider saw that flag on
the resource *before* the destroy. Passing -var force_destroy=true at destroy
time is too late: the delete fails with BucketNotEmpty on any versioned bucket.
So the teardown empties it explicitly first.

    python bin/empty-bucket.py <bucket>
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    bucket = sys.argv[1]
    s3 = boto3.client("s3")

    try:
        pages = s3.get_paginator("list_object_versions").paginate(Bucket=bucket)
        total = 0
        for page in pages:
            batch = [
                {"Key": o["Key"], "VersionId": o["VersionId"]}
                for key in ("Versions", "DeleteMarkers")
                for o in page.get(key, [])
            ]
            for i in range(0, len(batch), 1000):
                chunk = batch[i : i + 1000]
                s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})
                total += len(chunk)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchBucket", "404"}:
            print(f"  bucket {bucket} already gone")
            return 0
        raise

    print(f"  removed {total} object versions from {bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
