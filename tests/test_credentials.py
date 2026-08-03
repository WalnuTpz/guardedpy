from __future__ import annotations

from collections.abc import Callable

import pytest
from keyring.errors import NoKeyringError

from guardedpy.credentials import CredentialBackendUnavailableError, CredentialService


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self.values[(service_name, username)]


class UnavailableKeyring:
    def get_password(self, service_name: str, username: str) -> str | None:
        raise NoKeyringError("no backend")

    def set_password(self, service_name: str, username: str, password: str) -> None:
        raise NoKeyringError("no backend")

    def delete_password(self, service_name: str, username: str) -> None:
        raise NoKeyringError("no backend")


def test_status_is_boolean_only_and_key_can_be_set_and_cleared() -> None:
    service = CredentialService(FakeKeyring())

    assert service.status().configured is False
    service.set_key("not-a-real-key")
    assert service.status().configured is True
    assert "not-a-real-key" not in repr(service.status())

    service.clear_key()

    assert service.status().configured is False


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda service: service.status(), id="status"),
        pytest.param(lambda service: service.set_key("not-a-real-key"), id="set-key"),
        pytest.param(lambda service: service.clear_key(), id="clear-key"),
    ],
)
def test_unavailable_keyring_rejects_every_public_operation_without_plaintext_fallback(
    operation: Callable[[CredentialService], object],
) -> None:
    service = CredentialService(UnavailableKeyring())

    with pytest.raises(CredentialBackendUnavailableError, match="keyring backend is unavailable"):
        operation(service)
