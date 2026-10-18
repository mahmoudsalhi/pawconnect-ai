import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
import jwt

from .config import get_settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def create_access_token(user_id: int, remember: bool = True) -> str:
    s = get_settings()
    minutes = s.jwt_expire_minutes if remember else 60 * 12
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
        if "purpose" in payload:  # a challenge token (face login) is never a sign-in token
            return None
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


# ---------- one-time challenge tokens (face login) ----------
# A signed, short-lived token carries the challenge between the "options" and "verify" calls, so the
# server keeps no session. Each token is accepted once: answered tokens are remembered until they
# expire (in memory, enough for one server process; several servers would need a shared store).

_used_challenges: dict[str, datetime] = {}
_used_lock = threading.Lock()


def make_challenge_token(purpose: str, data: dict, minutes: int = 5) -> str:
    s = get_settings()
    payload = {**data, "purpose": purpose, "jti": uuid4().hex,
               "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def consume_challenge_token(token: str, purpose: str) -> dict | None:
    """The token's data if it is valid, unexpired, for this purpose and not used before; None otherwise."""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") != purpose or "jti" not in payload:
        return None
    now = datetime.now(timezone.utc)
    with _used_lock:
        for jti in [j for j, until in _used_challenges.items() if until < now]:
            del _used_challenges[jti]
        if payload["jti"] in _used_challenges:
            return None
        _used_challenges[payload["jti"]] = datetime.fromtimestamp(payload["exp"], timezone.utc)
    return payload
