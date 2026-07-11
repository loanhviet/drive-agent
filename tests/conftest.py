import pytest
from httpx import ASGITransport, AsyncClient

import server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_sessions():
    server.sessions.clear()
    yield
    server.sessions.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
