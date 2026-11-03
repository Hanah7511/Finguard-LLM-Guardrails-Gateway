"""Tests for the vault's encryption. The keys and values below are made up."""
from __future__ import annotations

import base64

import pytest

from src.storage.crypto import VaultCipher, VaultDecryptError, VaultKeyError

KEY = base64.urlsafe_b64encode(b"k" * 32).decode()
OTHER_KEY = base64.urlsafe_b64encode(b"o" * 32).decode()
VALUE = "GB82 WEST 1234 5698 7654 32"


@pytest.fixture
def cipher() -> VaultCipher:
    return VaultCipher(KEY)


def test_round_trip(cipher):
    nonce, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    assert cipher.decrypt("CASE-1", "[IBAN_CODE_1]", nonce, ciphertext) == VALUE


def test_the_value_is_not_readable_in_the_ciphertext(cipher):
    _, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    assert VALUE.encode() not in ciphertext


def test_encrypting_twice_looks_different(cipher):
    first = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    second = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    assert first[0] != second[0] and first[1] != second[1]


@pytest.mark.parametrize("case_id, token", [("CASE-2", "[IBAN_CODE_1]"), ("CASE-1", "[IBAN_CODE_2]")])
def test_a_row_moved_to_another_case_or_token_will_not_decrypt(cipher, case_id, token):
    nonce, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    with pytest.raises(VaultDecryptError):
        cipher.decrypt(case_id, token, nonce, ciphertext)


def test_a_changed_byte_will_not_decrypt(cipher):
    nonce, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    changed = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
    with pytest.raises(VaultDecryptError):
        cipher.decrypt("CASE-1", "[IBAN_CODE_1]", nonce, changed)


def test_the_wrong_key_will_not_decrypt(cipher):
    nonce, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    with pytest.raises(VaultDecryptError):
        VaultCipher(OTHER_KEY).decrypt("CASE-1", "[IBAN_CODE_1]", nonce, ciphertext)


def test_decrypt_error_names_no_value(cipher):
    nonce, ciphertext = cipher.encrypt("CASE-1", "[IBAN_CODE_1]", VALUE)
    with pytest.raises(VaultDecryptError) as err:
        VaultCipher(OTHER_KEY).decrypt("CASE-1", "[IBAN_CODE_1]", nonce, ciphertext)
    assert VALUE not in str(err.value)


def test_value_hash_is_stable_but_differs_by_case_type_and_key(cipher):
    base = cipher.value_hash("CASE-1", "IBAN_CODE", "GB82")
    assert base == cipher.value_hash("CASE-1", "IBAN_CODE", "GB82")
    assert base != cipher.value_hash("CASE-2", "IBAN_CODE", "GB82")
    assert base != cipher.value_hash("CASE-1", "EMAIL_ADDRESS", "GB82")
    assert base != VaultCipher(OTHER_KEY).value_hash("CASE-1", "IBAN_CODE", "GB82")


@pytest.mark.parametrize("bad", ["", "short", base64.urlsafe_b64encode(b"x" * 16).decode(), "qqqqq"])
def test_a_bad_master_key_is_refused_without_quoting_it(bad):
    with pytest.raises(VaultKeyError) as err:
        VaultCipher(bad)
    assert not bad or bad not in str(err.value)
