import gzip

import boto3
from botocore.config import Config

# Credentials from user docs
ACCESS_KEY = "4266a84c-b782-4520-8899-08b377da310e"
SECRET_KEY = "8uetaniQorD_G_b_4FdpntljEkUPGO_z"
ENDPOINT = "https://files.massive.com"

session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

s3 = session.client(
    "s3",
    endpoint_url=ENDPOINT,
    config=Config(signature_version="s3v4"),
)

bucket_name = "flatfiles"
# This key exists according to probe_s3.py
key = "us_stocks_sip/day_aggs_v1/2003/09/2003-09-10.csv.gz"

print(f"Attempting to download s3://{bucket_name}/{key}...")

try:
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    print("Download stream opened successfully.")

    with gzip.open(obj["Body"], "rt") as f:
        content = f.read(100)  # Read first 100 chars
        print(f"Success! Content preview:\n{content}")

except Exception as e:
    print(f"FAILED: {e}")
