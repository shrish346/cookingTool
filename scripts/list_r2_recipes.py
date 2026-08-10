"""List every recipe sitting in the object-storage bucket.

Notice mode (frontend/src/config/site.ts) needs a hand-maintained list of video ids,
because nothing can enumerate the bucket from the browser: the clips Worker only serves
`env.CLIPS.get(key)` by path and there is no list endpoint on the API. This script is how
that list gets regenerated after new videos are processed.

Read-only - it lists and GETs, never writes. Reads S3_* from .env.

    python3 scripts/list_r2_recipes.py            # human-readable table
    python3 scripts/list_r2_recipes.py --ts       # paste-ready BROWSE_RECIPES entries

Its output is a starting point, not the answer: test videos and non-cooking clips end up
in the bucket too, and a recipe with 0 clips is worth dropping. Check the CLIPS column.
"""

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def s3_client():
    # Mirrors the endpoint handling in backend/api/deps.py: R2 rejects a bucket path on
    # the endpoint, and appending one silently doubles the bucket into every key.
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    if endpoint:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}"

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION", "auto"),
        endpoint_url=endpoint or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ts",
        action="store_true",
        help="emit BROWSE_RECIPES entries for frontend/src/config/site.ts",
    )
    args = parser.parse_args()

    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        print("S3_BUCKET_NAME is not set - copy env.example to .env first.", file=sys.stderr)
        return 1

    s3 = s3_client()

    # One pass over the whole bucket: `{video_id}/recipe.json` alongside
    # `{video_id}/clips/*.mp4`, so clip counts come for free.
    recipes: list[str] = []
    clip_counts: dict[str, int] = {}

    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            video_id, _, rest = key.partition("/")
            if rest == "recipe.json":
                recipes.append(video_id)
            elif rest.startswith("clips/"):
                clip_counts[video_id] = clip_counts.get(video_id, 0) + 1

    rows = []
    for video_id in sorted(recipes):
        try:
            body = s3.get_object(Bucket=bucket, Key=f"{video_id}/recipe.json")["Body"].read()
            recipe = json.loads(body)
        except (ClientError, ValueError) as exc:
            print(f"{video_id}: could not read recipe.json ({exc})", file=sys.stderr)
            continue
        rows.append((video_id, recipe.get("title", "<untitled>"), clip_counts.get(video_id, 0)))

    if args.ts:
        for video_id, title, _ in rows:
            print(f"  {{ id: '{video_id}', title: {json.dumps(title)} }},")
    else:
        print(f"{'VIDEO ID':16} {'CLIPS':>5}  TITLE")
        for video_id, title, clips in rows:
            print(f"{video_id:16} {clips:>5}  {title}")
        print(f"\n{len(rows)} recipes in {bucket}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
