"""Persist raw uploads locally, and to S3 when S3_BUCKET is set."""
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("SEAM_DATA_DIR", "./data/uploads"))


def save(name: str, raw: bytes):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / Path(name).name).write_bytes(raw)
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        try:
            import boto3
            boto3.client("s3").put_object(Bucket=bucket, Key=f"uploads/{Path(name).name}", Body=raw)
        except Exception as e:  # keep the upload usable even if S3 is unreachable
            print("S3 upload failed:", e)
