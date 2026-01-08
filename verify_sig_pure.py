import base64
import struct
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def read_ssh_string(blob, offset):
    length = struct.unpack(">I", blob[offset : offset + 4])[0]
    offset += 4
    val = blob[offset : offset + length]
    offset += length
    return val, offset


def verify_openssh_signature(sig_path: Path, data_path: Path, allowed_signers_path: Path):
    # 1. Load Public Key from allowed_signers
    # Line format: email ssh-ed25519 BASE64... comment
    with open(allowed_signers_path) as f:
        line = f.read().strip()
    parts = line.split()
    # parts[0] = email/principal, parts[1] = algo, parts[2] = b64_key, parts[3] = comment
    pub_key_b64 = parts[2]
    pub_key_bytes = base64.b64decode(pub_key_b64)

    # Parse SSH Public Key Blob
    # [string:ssh-ed25519] [string:raw_key_bytes]
    key_type, offset = read_ssh_string(pub_key_bytes, 0)
    raw_key, offset = read_ssh_string(pub_key_bytes, offset)

    public_key = Ed25519PublicKey.from_public_bytes(raw_key)
    print(f"Loaded Public Key: {parts[0]}")

    # 2. Parse Signature File
    # Header: "-----BEGIN SSH SIGNATURE-----\n"
    # Body: Base64 blob
    with open(sig_path) as f:
        content = f.read()

    body = (
        content.replace("-----BEGIN SSH SIGNATURE-----", "")
        .replace("-----END SSH SIGNATURE-----", "")
        .strip()
    )
    sig_blob = base64.b64decode(body)

    # Structure of SIG blob:
    # [string: MAGIC] = "SSHSIG"
    # [uint32: version]
    # [string: public_key_blob] (must match ours)
    # [string: namespace]
    # [string: reserved]
    # [string: hash_algorithm]
    # [string: signature_blob]

    print(f"Sig blob preview (hex): {sig_blob[:20].hex()}")
    print(f"Sig blob preview (ascii): {sig_blob[:20]}")

    offset = 0
    magic = sig_blob[offset : offset + 6]
    offset += 6
    if magic != b"SSHSIG":
        raise ValueError(f"Invalid magic header: {magic}")

    version = struct.unpack(">I", sig_blob[offset : offset + 4])[0]
    offset += 4

    pk_blob, offset = read_ssh_string(sig_blob, offset)
    if pk_blob != pub_key_bytes:
        raise ValueError("Public key in signature doesn't match allowed_signers")

    namespace, offset = read_ssh_string(sig_blob, offset)
    print(f"Namespace: {namespace.decode()}")

    reserved, offset = read_ssh_string(sig_blob, offset)

    hash_algo, offset = read_ssh_string(sig_blob, offset)
    print(f"Hash Algo: {hash_algo.decode()}")

    # Nested signature blob: [string: algo] [string: raw_sig]
    nested_sig, offset = read_ssh_string(sig_blob, offset)
    ns_algo, ns_off = read_ssh_string(nested_sig, 0)
    raw_signature, ns_off = read_ssh_string(nested_sig, ns_off)

    # 3. Construct Signed Data
    # OpenSSH signs:
    # [string: MAGIC] "SSHSIG"
    # [string: namespace]
    # [string: reserved]
    # [string: hash_algo]
    # [string: H(raw_data)]

    with open(data_path, "rb") as f:
        raw_data = f.read()

    from cryptography.hazmat.primitives import hashes

    digest = hashes.Hash(hashes.SHA512())
    digest.update(raw_data)
    data_hash = digest.finalize()

    def ssh_string(data):
        return struct.pack(">I", len(data)) + data

    signed_payload = (
        b"SSHSIG"
        + ssh_string(namespace)
        + ssh_string(reserved)
        + ssh_string(hash_algo)
        + ssh_string(data_hash)
    )

    # 4. Verify
    try:
        public_key.verify(raw_signature, signed_payload)
        print("✅ SUCCESS: Signature Valid Verified (Pure Python)")
        return True
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False


# Run
ds_path = Path("datasets/qqq_voo_spy_common_window_yahoo_wide_v1.0.0")
verify_openssh_signature(
    ds_path / "RELEASE.sig", ds_path / "RELEASE.json", Path("keys/allowed_signers")
)
