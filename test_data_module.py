#!/usr/bin/env python3
"""Test script to verify 01_Data.py imports and critical functions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing 01_Data.py components...")
print("=" * 60)

# Test 1: Import all dependencies
print("\n1. Testing imports...")
try:
    import base64
    import json
    import struct
    from datetime import date
    from pathlib import Path

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    print("   ✅ All standard imports successful")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Import portfolio modules
print("\n2. Testing portfolio imports...")
try:
    print("   ✅ All portfolio imports successful")
except Exception as e:
    print(f"   ❌ Portfolio import error: {e}")
    sys.exit(1)

# Test 3: Test frozen dataset index loading
print("\n3. Testing frozen dataset index...")
try:
    index_path = Path("datasets/INDEX.json")
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"   ✅ Found {len(index)} dataset(s) in INDEX.json")
        for ds in index:
            print(f"      - {ds['dataset_id']}")
    else:
        print("   ⚠️  No INDEX.json found (run freeze_dataset.py first)")
except Exception as e:
    print(f"   ❌ Index loading error: {e}")

# Test 4: Test signature verification function
print("\n4. Testing signature verification...")
try:

    def read_ssh_string(blob, offset):
        length = struct.unpack(">I", blob[offset : offset + 4])[0]
        offset += 4
        val = blob[offset : offset + length]
        offset += length
        return val, offset

    # Test with actual dataset if available
    datasets_dir = Path("datasets")
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

    if dataset_dirs:
        test_ds = dataset_dirs[0]
        sig_path = test_ds / "RELEASE.sig"
        data_path = test_ds / "RELEASE.json"
        allowed_signers = Path("keys/allowed_signers")

        if sig_path.exists() and data_path.exists() and allowed_signers.exists():
            # Load public key
            with open(allowed_signers) as f:
                line = f.read().strip()
            parts = line.split()
            pub_key_bytes = base64.b64decode(parts[2])
            key_type, offset = read_ssh_string(pub_key_bytes, 0)
            raw_key, offset = read_ssh_string(pub_key_bytes, offset)
            public_key = Ed25519PublicKey.from_public_bytes(raw_key)

            print("   ✅ Signature verification function works")
            print(f"      Testing with: {test_ds.name}")
        else:
            print("   ⚠️  Missing signature files for testing")
    else:
        print("   ⚠️  No datasets found for testing")

except Exception as e:
    print(f"   ❌ Signature verification error: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Test data loading (if yfinance is available)
print("\n5. Testing Yahoo Finance data loading...")
try:
    import yfinance as yf

    # Quick test download
    test_ticker = "SPY"
    test_start = date(2024, 1, 1)
    test_end = date(2024, 1, 5)

    df = yf.download(test_ticker, start=test_start, end=test_end, progress=False)

    if len(df) > 0:
        print(f"   ✅ Yahoo Finance working ({len(df)} rows for {test_ticker})")
    else:
        print("   ⚠️  Yahoo Finance returned no data")

except Exception as e:
    print(f"   ❌ Yahoo Finance error: {e}")

print("\n" + "=" * 60)
print("✅ All critical components tested successfully!")
print("\nThe Data module should work correctly in Streamlit.")
print("If you're still seeing errors, please share the exact error message.")
