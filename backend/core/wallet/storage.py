"""Encrypted wallet file storage."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from core.wallet.encryption import encrypt_private_key, decrypt_private_key


class WalletData(TypedDict):
    address: str
    encrypted_key: str
    salt: str
    created_at: str


class WalletStorage:
    """Manages encrypted wallet storage on disk."""

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        """Check if wallet file exists."""
        return self.path.exists()

    def load(self) -> WalletData | None:
        """Load wallet data (without decrypting)."""
        if not self.exists():
            return None
        return json.loads(self.path.read_text())

    def save(self, address: str, private_key: str, password: str) -> WalletData:
        """Encrypt and save wallet."""
        encrypted_key, salt = encrypt_private_key(private_key, password)

        data: WalletData = {
            "address": address,
            "encrypted_key": encrypted_key,
            "salt": salt,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        return data

    def decrypt(self, password: str) -> str:
        """Decrypt and return private key."""
        data = self.load()
        if not data:
            raise ValueError("No wallet found")
        return decrypt_private_key(data["encrypted_key"], data["salt"], password)

    def delete(self) -> bool:
        """Delete wallet file."""
        if self.exists():
            self.path.unlink()
            return True
        return False
