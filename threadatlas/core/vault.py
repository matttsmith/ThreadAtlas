"""Vault layout helpers.

The vault is the on-disk source of truth. SQLite is the index over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


VAULT_SUBDIRS = (
    "raw_imports",
    "normalized",
    "db",
    "exports",
    "cache",
    "logs",
)

VAULT_MARKER = ".threadatlas_vault.json"


@dataclass(frozen=True)
class Vault:
    root: Path

    @property
    def raw_imports(self) -> Path:
        return self.root / "raw_imports"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "threadatlas.sqlite3"

    @property
    def marker_path(self) -> Path:
        return self.root / VAULT_MARKER

    def normalized_path_for(self, conversation_id: str) -> Path:
        # Shard by the first 2 hex characters of the id suffix to avoid
        # huge flat directories.
        suffix = conversation_id.split("_", 1)[-1]
        shard = suffix[:2] if len(suffix) >= 2 else "00"
        return self.normalized / shard / f"{conversation_id}.json"

    def assert_initialized(self) -> None:
        if not self.marker_path.exists():
            raise FileNotFoundError(
                f"Not a ThreadAtlas vault: {self.root} (missing {VAULT_MARKER})"
            )


def init_vault(root: Path) -> Vault:
    """Create a new vault directory tree.

    Idempotent: re-running on an existing vault is safe.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sub in VAULT_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    marker = root / VAULT_MARKER
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "kind": "threadatlas_vault",
                    "schema_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return Vault(root=root)


def open_vault(root: Path) -> Vault:
    """Open an existing vault. Raises if not initialized."""
    v = Vault(root=Path(root).resolve())
    v.assert_initialized()
    return v
