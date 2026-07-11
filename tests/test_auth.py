import pytest

from services.auth import AuthService, AuthenticationError


def test_token_round_trip_and_role_scopes(tmp_path):
    service = AuthService(
        str(tmp_path / "auth.db"),
        jwt_secret="a-secure-test-secret-with-more-than-32-characters",
    )
    service.create_user("reader-id", "reader", "reader-password", "user")

    user = service.authenticate("reader", "reader-password")
    token, expires_in = service.issue_token(user)
    verified = service.verify_token(token)

    assert expires_in == 3600
    assert verified.user_id == "reader-id"
    assert verified.scopes == ["drive:read", "memory:read"]


def test_authentication_rejects_invalid_password(tmp_path):
    service = AuthService(
        str(tmp_path / "auth.db"),
        jwt_secret="a-secure-test-secret-with-more-than-32-characters",
    )
    service.create_user("admin-id", "admin", "admin-password", "admin")

    with pytest.raises(AuthenticationError, match="Invalid username or password"):
        service.authenticate("admin", "incorrect-password")


def test_short_jwt_secret_is_rejected(tmp_path):
    service = AuthService(str(tmp_path / "auth.db"), jwt_secret="short")
    service.create_user("admin-id", "admin", "admin-password", "admin")
    user = service.authenticate("admin", "admin-password")

    with pytest.raises(RuntimeError, match="at least 32 characters"):
        service.issue_token(user)


def test_expired_token_is_rejected(tmp_path):
    service = AuthService(
        str(tmp_path / "auth.db"),
        jwt_secret="a-secure-test-secret-with-more-than-32-characters",
        expire_minutes=-1,
    )
    service.create_user("admin-id", "admin", "admin-password", "admin")
    token, _ = service.issue_token(service.authenticate("admin", "admin-password"))

    with pytest.raises(AuthenticationError, match="has expired"):
        service.verify_token(token)
