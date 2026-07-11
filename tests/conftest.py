import pytest
from httpx import ASGITransport, AsyncClient

import server
import services.audit as audit_module
import services.auth as auth_module
from services.audit import AuditStore
from services.auth import AuthService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    database = tmp_path / "test.db"
    auth_service = AuthService(
        db_path=str(database),
        jwt_secret="test-secret-that-is-longer-than-32-characters",
        expire_minutes=60,
    )
    auth_service.create_user("admin-id", "admin", "admin-password", "admin")
    auth_service.create_user("user-id", "user", "user-password", "user")
    audit_store = AuditStore(str(database))
    auth_module._auth_service = auth_service
    audit_module._audit_store = audit_store
    server.sessions.clear()
    yield {"auth": auth_service, "audit": audit_store}
    server.sessions.clear()
    auth_module._auth_service = None
    audit_module._audit_store = None


@pytest.fixture
async def client():
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.fixture
def admin_token(isolated_services):
    user = isolated_services["auth"].authenticate("admin", "admin-password")
    return isolated_services["auth"].issue_token(user)[0]


@pytest.fixture
def user_token(isolated_services):
    user = isolated_services["auth"].authenticate("user", "user-password")
    return isolated_services["auth"].issue_token(user)[0]
