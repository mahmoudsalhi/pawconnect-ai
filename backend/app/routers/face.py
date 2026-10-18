"""AI face login: sign in by looking at the webcam (YuNet + SFace, calibrated on LFW in ml/face).

Opt-in and 1:1: the person types their email, then their face must match that account's saved
embedding (128 numbers, never a photo). A head-turn challenge stops printed photos and still images.
Photos are analysed in memory and never stored.
"""

import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import func, select

from ..ai import face
from ..ai.face_engine import FaceError
from ..deps import DB, CurrentUser
from ..models import FaceProfile, User
from ..schemas import FaceChallengeIn, FaceChallengeOut, FaceEnrollIn, FaceLoginIn, FaceStatusOut, TokenOut, UserOut
from ..security import consume_challenge_token, create_access_token, make_challenge_token
from ..services.notify import log_activity

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/face", tags=["face login"])

CHALLENGE_MINUTES = 2
MAX_FAILURES, WINDOW_SECONDS = 5, 15 * 60
NOT_RECOGNISED = "We couldn’t sign you in with your face. Try again, or use your password."
UNAVAILABLE = "Face login isn’t available on this server right now. Use your password."
# Camera problems say nothing about who is in front of it, so they don't count as failed attempts.
CAMERA_PROBLEMS = {"bad_image", "no_face", "too_far", "many_faces", "not_frontal"}

_failures: dict[str, deque] = defaultdict(deque)
_failures_lock = threading.Lock()


def _recent_failures(email: str) -> int:
    with _failures_lock:
        q, cutoff = _failures[email], time.monotonic() - WINDOW_SECONDS
        while q and q[0] < cutoff:
            q.popleft()
        return len(q)


def _fail(email: str, detail: str, code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    with _failures_lock:
        _failures[email].append(time.monotonic())
    return HTTPException(code, detail)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- set up (signed-in user) ----------


@router.get("", response_model=FaceStatusOut)
def face_status(user: CurrentUser, db: DB):
    profile = db.get(FaceProfile, user.id)
    return FaceStatusOut(available=face.get_engine() is not None, enrolled=profile is not None,
                         created_at=profile.created_at if profile else None,
                         last_used_at=profile.last_used_at if profile else None)


@router.post("/enroll", response_model=FaceStatusOut, status_code=status.HTTP_201_CREATED)
def enroll(data: FaceEnrollIn, user: CurrentUser, db: DB):
    if not data.consent:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please tick the box to agree before saving your face data.")
    try:
        embedding = face.enrolment_embedding(data.images)
    except face.FaceUnavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE)
    except FaceError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message)

    profile = db.get(FaceProfile, user.id)
    if profile is None:
        profile = FaceProfile(user_id=user.id)
        db.add(profile)
    profile.embedding, profile.model = embedding.tobytes(), face.MODEL_VERSION
    profile.consented_at = profile.created_at = _now()
    profile.last_used_at = None
    log_activity(db, user.id, "face", "You set up face login")
    db.commit()
    return FaceStatusOut(available=True, enrolled=True, created_at=profile.created_at)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_face_data(user: CurrentUser, db: DB):
    profile = db.get(FaceProfile, user.id)
    if profile:
        db.delete(profile)
        log_activity(db, user.id, "face", "You deleted your face login data")
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------- sign in ----------


@router.post("/login/challenge", response_model=FaceChallengeOut)
def login_challenge(data: FaceChallengeIn):
    """Which way to turn, chosen at random so a pre-recorded video turning one way fails half the time.

    Answered the same way for any email, so it doesn't reveal who has face login.
    """
    email = data.email.strip().lower()
    direction = secrets.choice(["left", "right"])
    state = make_challenge_token("face-login", {"email": email, "dir": direction}, CHALLENGE_MINUTES)
    return FaceChallengeOut(direction=direction, state=state)


@router.post("/login", response_model=TokenOut)
def login(data: FaceLoginIn, db: DB):
    email = data.email.strip().lower()
    if _recent_failures(email) >= MAX_FAILURES:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many face login attempts. Use your password, or try again in 15 minutes.")
    challenge = consume_challenge_token(data.state, "face-login")
    if challenge is None or challenge.get("email") != email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This face check expired. Please start again.")

    try:
        front = face.check_liveness(data.front, data.turned, challenge["dir"])
    except face.FaceUnavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE)
    except FaceError as exc:
        if exc.code in CAMERA_PROBLEMS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message)
        raise _fail(email, exc.message, status.HTTP_400_BAD_REQUEST)

    user = db.scalar(select(User).where(func.lower(User.email) == email))
    profile = db.get(FaceProfile, user.id) if user else None
    if profile is None or profile.model != face.MODEL_VERSION:
        raise _fail(email, NOT_RECOGNISED)
    ok, score = face.matches(front.embedding, profile.embedding)
    if not ok:
        log.info("Face login refused for user %s (similarity %.3f)", user.id, score)
        raise _fail(email, NOT_RECOGNISED)

    with _failures_lock:
        _failures.pop(email, None)
    profile.last_used_at = _now()
    db.commit()
    return TokenOut(access_token=create_access_token(user.id, data.remember), user=UserOut.model_validate(user))
