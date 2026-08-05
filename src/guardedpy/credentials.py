"""Keyring-only API credential storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from keyring.errors import KeyringError


_SERVICE_NAME = "guardedpy"
_USERNAME = "deepseek-api-key"


class KeyringBackend(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


class CredentialBackendUnavailableError(RuntimeError):
    """Raised when the operating-system keyring cannot be used."""


class CredentialNotConfiguredError(RuntimeError):
    """Raised when a provider requests a credential that has not been stored."""


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    configured: bool


class CredentialService:
    """Store a credential exclusively through an injected keyring backend."""

    def __init__(self, keyring: KeyringBackend) -> None:
        self._keyring = keyring

    def status(self) -> CredentialStatus:
        try:
            configured = self._keyring.get_password(_SERVICE_NAME, _USERNAME) is not None
        except KeyringError as error:
            raise CredentialBackendUnavailableError("keyring backend is unavailable") from error
        return CredentialStatus(configured=configured)

    def get_key(self) -> str:
        """Return the configured key to the provider adapter, never a fallback value."""
        try:
            key = self._keyring.get_password(_SERVICE_NAME, _USERNAME)
        except KeyringError as error:
            raise CredentialBackendUnavailableError("keyring backend is unavailable") from error
        if key is None:
            raise CredentialNotConfiguredError("credential is not configured")
        return key

    def set_key(self, key: str) -> None:
        try:
            self._keyring.set_password(_SERVICE_NAME, _USERNAME, key)
        except KeyringError as error:
            raise CredentialBackendUnavailableError("keyring backend is unavailable") from error

    def clear_key(self) -> None:
        try:
            self._keyring.delete_password(_SERVICE_NAME, _USERNAME)
        except KeyringError as error:
            raise CredentialBackendUnavailableError("keyring backend is unavailable") from error
