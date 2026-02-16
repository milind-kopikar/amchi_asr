#!/usr/bin/env python3
"""
Upload Amchi ASR checkpoint(s) to Cloudflare R2.
Run this on your RunPod pod (or any machine that has the .ckpt and network access).

Requires: pip install boto3

Environment variables (or pass as arguments):
  R2_ACCOUNT_ID       - Cloudflare account ID (from R2 dashboard)
  R2_ACCESS_KEY_ID    - R2 API token access key
  R2_SECRET_ACCESS_KEY - R2 API token secret key
  R2_BUCKET_NAME      - R2 bucket name (e.g. amchi-checkpoints)

R2 object keys mirror the repo layout so you know which checkpoint is where, e.g.:
  results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt

Usage:
  # Upload the default best checkpoint (key: results/2026-02-13_marathi_amchi_20epoch/checkpoints/...)
  export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET_NAME=amchi-checkpoints
  python scripts/upload_checkpoint_to_r2.py

  # Upload and print instructions for public URL (no expiry)
  python scripts/upload_checkpoint_to_r2.py --public-url

  # Upload a different checkpoint (key derived from path under repo)
  python scripts/upload_checkpoint_to_r2.py --file results/2026-03-01_marathi_v2/checkpoints/best.ckpt
"""
import argparse
import os
import sys

# R2 uses S3-compatible API
R2_ENDPOINT_TEMPLATE = "https://{account_id}.r2.cloudflarestorage.com"


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_key_for_path(local_path: str) -> str:
    """Use path relative to repo root as R2 key so layout is clear (e.g. results/.../checkpoints/file.ckpt)."""
    root = _repo_root()
    try:
        rel = os.path.relpath(local_path, root)
        return rel.replace("\\", "/")
    except ValueError:
        return "checkpoints/" + os.path.basename(local_path)


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Cloudflare R2")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to .ckpt file (default: best checkpoint from 20-epoch run)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Object key in R2 (default: path relative to repo, e.g. results/.../checkpoints/file.ckpt)",
    )
    parser.add_argument(
        "--presigned-expiry",
        type=int,
        default=0,
        help="If > 0, print a presigned GET URL valid for this many seconds (e.g. 604800 = 7 days)",
    )
    parser.add_argument(
        "--public-url",
        action="store_true",
        help="After upload, print the object key and how to get a public URL (no expiry)",
    )
    args = parser.parse_args()

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not all([account_id, access_key, secret_key, bucket]):
        print(
            "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME",
            file=sys.stderr,
        )
        sys.exit(1)

    default_ckpt = os.path.join(
        _repo_root(),
        "results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt",
    )
    local_path = args.file or default_ckpt
    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    object_key = args.key or _default_key_for_path(local_path)

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("Install boto3: pip install boto3", file=sys.stderr)
        sys.exit(1)

    endpoint = R2_ENDPOINT_TEMPLATE.format(account_id=account_id)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} ({file_size_mb:.1f} MB) to r2://{bucket}/{object_key} ...")
    s3.upload_file(local_path, bucket, object_key)
    print(f"Uploaded to r2://{bucket}/{object_key}")

    if args.presigned_expiry > 0:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=args.presigned_expiry,
        )
        print("\nPresigned URL (use as CHECKPOINT_URL in RunPod endpoint):")
        print(url)
        print(f"\nValid for {args.presigned_expiry} seconds ({args.presigned_expiry // 86400} days).")

    if args.public_url:
        print("\n--- Public URL (no expiry) ---")
        print("Object key in R2:", object_key)
        print(
            "1. In Cloudflare Dashboard → R2 → your bucket → Settings → enable 'Public access' (Allow Access)."
        )
        print(
            "2. Note the public bucket URL (e.g. https://pub-xxxx.r2.dev). Your checkpoint URL is:"
        )
        print("   <your-public-bucket-url>/" + object_key)
        print("   Use that full URL as CHECKPOINT_URL in your RunPod serverless endpoint.")


if __name__ == "__main__":
    main()
