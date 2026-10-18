"""AI face login rules. The model itself is measured on LFW (ml/face/calibrate.py); here the face reader
is replaced by fake readings so the login logic can be tested exactly: liveness, 1:1 match, replay, lockout."""

import numpy as np
import pytest

from app.ai import face
from app.ai.face_engine import FaceResult


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


ALICE, BOB = _unit(1), _unit(2)
ALICE_TURNED = ALICE + 0.25 * _unit(3)  # same person, other angle: still very similar
READINGS = {  # fake photo -> what the face reader "sees"
    "alice-front": (ALICE, 0.02),
    "alice-left": (ALICE_TURNED / np.linalg.norm(ALICE_TURNED), 0.45),  # turned to her own left
    "alice-right": (ALICE_TURNED / np.linalg.norm(ALICE_TURNED), -0.45),
    "bob-front": (BOB, 0.01),
    "bob-left": (BOB, 0.42),
    "bob-right": (BOB, -0.42),
}


def photo(name: str) -> str:
    return f"{name}:" + "x" * 120  # the API requires a realistic-length image string


@pytest.fixture(autouse=True)
def fake_face_reader(monkeypatch):
    def analyze(data: str) -> FaceResult:
        emb, yaw = READINGS[data.split(":", 1)[0]]
        return FaceResult(embedding=emb, yaw=yaw, score=0.99, width=200)
    monkeypatch.setattr(face, "analyze_image", analyze)


def email_of(client, headers) -> str:
    return client.get("/api/auth/me", headers=headers).json()["email"]


def enroll(client, headers, who="alice"):
    return client.post("/api/face/enroll", headers=headers,
                       json={"images": [photo(f"{who}-front")] * 3, "consent": True})


def face_login(client, email, who="alice", turn=None, front=None):
    ch = client.post("/api/face/login/challenge", json={"email": email}).json()
    turned = [photo(f"{who}-{turn or ch['direction']}")]
    return client.post("/api/face/login", json={"email": email, "state": ch["state"],
                                                "front": photo(front or f"{who}-front"), "turned": turned})


def test_enrolment_needs_consent(client, newbie):
    r = client.post("/api/face/enroll", headers=newbie, json={"images": [photo("alice-front")] * 3, "consent": False})
    assert r.status_code == 400


def test_enrol_then_sign_in_with_face(client, newbie):
    assert enroll(client, newbie).status_code == 201
    assert client.get("/api/face", headers=newbie).json()["enrolled"] is True
    r = face_login(client, email_of(client, newbie))
    assert r.status_code == 200, r.text
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert me["email"] == email_of(client, newbie)


def test_printed_photo_without_head_turn_is_refused(client, newbie):
    enroll(client, newbie)
    ch = client.post("/api/face/login/challenge", json={"email": email_of(client, newbie)}).json()
    r = client.post("/api/face/login", json={"email": email_of(client, newbie), "state": ch["state"],
                                             "front": photo("alice-front"), "turned": [photo("alice-front")]})
    assert r.status_code == 400 and "turn" in r.json()["detail"]


def test_turning_the_wrong_way_is_refused(client, newbie):
    enroll(client, newbie)
    email = email_of(client, newbie)
    ch = client.post("/api/face/login/challenge", json={"email": email}).json()
    wrong = "right" if ch["direction"] == "left" else "left"
    r = client.post("/api/face/login", json={"email": email, "state": ch["state"],
                                             "front": photo("alice-front"), "turned": [photo(f"alice-{wrong}")]})
    assert r.status_code == 400


def test_someone_elses_face_is_refused(client, newbie):
    enroll(client, newbie)
    r = face_login(client, email_of(client, newbie), who="bob")
    assert r.status_code == 401


def test_account_without_face_login_gets_the_same_answer(client, newbie):
    # Same message as a wrong face: an attacker can't find out who has face login.
    r = face_login(client, email_of(client, newbie))
    wrong = face_login(client, "nobody-here@example.com")
    assert r.status_code == wrong.status_code == 401 and r.json()["detail"] == wrong.json()["detail"]


def test_face_check_cannot_be_replayed_or_moved_to_another_account(client, newbie):
    enroll(client, newbie)
    email = email_of(client, newbie)
    ch = client.post("/api/face/login/challenge", json={"email": email}).json()
    body = {"email": email, "state": ch["state"], "front": photo("alice-front"), "turned": [photo(f"alice-{ch['direction']}")]}
    assert client.post("/api/face/login", json=body).status_code == 200
    assert client.post("/api/face/login", json=body).status_code == 400  # replay
    ch = client.post("/api/face/login/challenge", json={"email": "someone.else@example.com"}).json()
    assert client.post("/api/face/login", json={**body, "state": ch["state"]}).status_code == 400  # other account's check


def test_too_many_failures_lock_face_login(client, newbie):
    enroll(client, newbie)
    email = email_of(client, newbie)
    for _ in range(5):
        assert face_login(client, email, who="bob").status_code == 401
    assert face_login(client, email).status_code == 429  # even the right face waits; the password still works


def test_deleting_face_data_turns_face_login_off(client, newbie):
    enroll(client, newbie)
    assert client.delete("/api/face", headers=newbie).status_code == 204
    assert client.get("/api/face", headers=newbie).json()["enrolled"] is False
    assert face_login(client, email_of(client, newbie)).status_code == 401


def test_face_challenge_token_is_not_a_sign_in_token(client):
    state = client.post("/api/face/login/challenge", json={"email": "yasmine@example.com"}).json()["state"]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {state}"}).status_code == 401
