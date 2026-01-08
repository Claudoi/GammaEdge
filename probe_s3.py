import boto3
from botocore.config import Config

# Credentials from user docs + context
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
prefix = "us_stocks_sip"

print(f"Probing s3://{bucket_name}/{prefix}...")

try:
    paginator = s3.get_paginator("list_objects_v2")

    # Try to find what kind of aggregates exist
    # Look for day_aggs_v1 or similar for 2000
    print("\nListing first few objects:")
    count = 0
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=20):
        if "Contents" in page:
            for obj in page["Contents"]:
                print(f" - {obj['Key']} ({obj['Size']} bytes)")
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break

    # Check specifically for old dates
    print("\nChecking for year 2000...")
    prefix_2000 = "us_stocks_sip/day_aggs_v1/2000"
    count = 0
    found_2000 = False
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix_2000, MaxKeys=5):
        if "Contents" in page:
            for obj in page["Contents"]:
                print(f" - {obj['Key']}")
                found_2000 = True
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break

    if not found_2000:
        print("No daily aggs found for 2000.")

except Exception as e:
    print(f"Error: {e}")
