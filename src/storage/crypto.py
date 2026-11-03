"""Encrypts the real values held in the vault (AES-256-GCM).

One master key lives outside the database. Two separate keys are derived from it: one encrypts,
one makes the keyed hash that finds "the same value" without decrypting anything. Each value is
sealed to its case and token, so a row copied to another case or changed in any way will not
decrypt. A stolen database without the master key is only scrambled bytes.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_VERSION = 1
KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12  # the size AES-GCM expects; must never repeat under one key, so it is random


class VaultKeyError(ValueError):
    """The master key is missing or malformed. The message never contains the key."""


class VaultDecryptError(RuntimeError):
    """A stored value would not decrypt: wrong key, moved row or tampering."""


def _derive(master: bytes, purpose: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=purpose).derive(master)


class VaultCipher:
    def __init__(self, master_key_b64: str) -> None:
        try:
            master = base64.urlsafe_b64decode(master_key_b64.encode())
        except (binascii.Error, ValueError):
            raise VaultKeyError("VAULT_MASTER_KEY is not valid base64") from None
        if len(master) != KEY_BYTES:
            raise VaultKeyError(f"VAULT_MASTER_KEY must be {KEY_BYTES} bytes of base64")
        self._aead = AESGCM(_derive(master, b"finguard vault encryption v1"))
        self._hash_key = _derive(master, b"finguard vault same-value hash v1")

    def value_hash(self, case_id: str, entity_type: str, value_key: str) -> bytes:
        """Keyed, and includes the case, so equal values in two cases cannot be matched."""
        message = "\0".join((case_id, entity_type, value_key)).encode()
        return hmac.new(self._hash_key, message, hashlib.sha256).digest()

    def encrypt(self, case_id: str, token: str, value: str) -> tuple[bytes, bytes]:
        """Returns (nonce, ciphertext)."""
        nonce = os.urandom(NONCE_BYTES)
        return nonce, self._aead.encrypt(nonce, value.encode(), self._sealed_to(case_id, token))

    def decrypt(self, case_id: str, token: str, nonce: bytes, ciphertext: bytes) -> str:
        try:
            return self._aead.decrypt(nonce, ciphertext, self._sealed_to(case_id, token)).decode()
        except InvalidTag:
            raise VaultDecryptError(f"{token} in case {case_id} would not decrypt") from None

    @staticmethod
    def _sealed_to(case_id: str, token: str) -> bytes:
        return f"{case_id}\0{token}".encode()
