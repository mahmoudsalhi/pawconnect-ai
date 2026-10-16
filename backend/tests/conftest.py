import os
import tempfile
from pathlib import Path

# Point the app at TEST_DATABASE_URL before anything reads the settings.
for _line in (Path(__file__).resolve().parent.parent / ".env").read_text(encoding="utf-8").splitlines():
    _key, _, _value = _line.partition("=")
    if _key.strip() == "TEST_DATABASE_URL" and _value.strip():
        os.environ["DATABASE_URL"] = _value.strip()
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="pawconnect-test-uploads-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.seed import DEMO_PASSWORD, load  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def database():
    assert "test" in os.environ["DATABASE_URL"], "Refusing to run tests against a non-test database"
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    load(db)
    db.close()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client():
    return TestClient(app)


def _login(client, email):
    r = client.post("/api/auth/login", json={"email": email, "password": DEMO_PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def demo(client):
    return _login(client, "yasmine@example.com")


@pytest.fixture
def staff(client):
    return _login(client, "carthage@example.com")


@pytest.fixture
def newbie(client):
    import uuid

    r = client.post("/api/auth/register", json={"name": "Sami Test", "email": f"{uuid.uuid4().hex[:8]}@example.com",
                                                "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
