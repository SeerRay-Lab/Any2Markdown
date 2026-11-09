"""Model profiles: named model endpoints the web UI can pick per request.

Profiles are stored as JSON at ``path`` and shared by everyone using the
server. API keys never leave the server: :meth:`ProfileStore.public` only
reports whether a key is set.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from any2markdown.model_client import ModelClient

logger = logging.getLogger(__name__)


class ProfileStore:
    def __init__(
        self,
        path: Path,
        seed: dict[str, Any],
        client_factory: Callable[[dict[str, Any]], ModelClient],
    ):
        """``seed`` is the profile used until the file has been saved once."""
        self.path = path
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._clients: dict[str, ModelClient] = {}
        self._data = self._load(seed)

    def _load(self, seed: dict[str, Any]) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            profiles = [p for p in data.get("profiles", []) if p.get("id")]
            if profiles:
                ids = {p["id"] for p in profiles}
                default = data.get("default")
                if default not in ids:
                    default = profiles[0]["id"]
                return {"default": default, "profiles": profiles}
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            logger.warning("cannot read model profiles %s: %s; using defaults", self.path, e)
        return {"default": seed["id"], "profiles": [dict(seed)]}

    def _find(self, profile_id: str) -> dict[str, Any] | None:
        return next((p for p in self._data["profiles"] if p["id"] == profile_id), None)

    def get(self, profile_id: str) -> dict[str, Any] | None:
        """Full profile (API key included), for server-side use only."""
        with self._lock:
            profile = self._find(profile_id)
            return dict(profile) if profile else None

    def public(self) -> dict[str, Any]:
        with self._lock:
            return {
                "default": self._data["default"],
                "profiles": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "model": p["model"],
                        "base_url": p["base_url"],
                        "has_api_key": bool(p.get("api_key")),
                    }
                    for p in self._data["profiles"]
                ],
            }

    def client(self, profile_id: str | None = None) -> ModelClient:
        """Model client for ``profile_id`` (default profile when ``None``).

        Raises ``KeyError`` for an unknown id.
        """
        with self._lock:
            pid = profile_id or self._data["default"]
            if pid not in self._clients:
                profile = self._find(pid)
                if profile is None:
                    raise KeyError(pid)
                self._clients[pid] = self._client_factory(profile)
            return self._clients[pid]

    def save(self, profile_id: str | None, fields: dict[str, Any]) -> str:
        """Create (``profile_id=None``) or update a profile; returns its id.

        ``fields["api_key"] is None`` keeps the stored key.
        """
        with self._lock:
            profiles = [dict(p) for p in self._data["profiles"]]
            if profile_id is None:
                profile_id = uuid.uuid4().hex[:8]
                profile: dict[str, Any] = {"id": profile_id, "api_key": None}
                profiles.append(profile)
            else:
                found = next((p for p in profiles if p["id"] == profile_id), None)
                if found is None:
                    raise KeyError(profile_id)
                profile = found
            for key in ("name", "model", "base_url"):
                profile[key] = fields[key]
            if fields.get("api_key") is not None:
                profile["api_key"] = fields["api_key"] or None
            self._write({"default": self._data["default"], "profiles": profiles})
            self._clients.pop(profile_id, None)
            return profile_id

    def delete(self, profile_id: str) -> None:
        with self._lock:
            profiles = [p for p in self._data["profiles"] if p["id"] != profile_id]
            if len(profiles) == len(self._data["profiles"]):
                raise KeyError(profile_id)
            if not profiles:
                raise ValueError("至少需要保留一个模型")
            default = self._data["default"]
            if default == profile_id:
                default = profiles[0]["id"]
            self._write({"default": default, "profiles": profiles})
            self._clients.pop(profile_id, None)

    def _write(self, data: dict[str, Any]) -> None:
        # Write-then-rename so a crash never leaves a half-written file; the
        # file holds API keys, so keep it private to the server user.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        self._data = data
