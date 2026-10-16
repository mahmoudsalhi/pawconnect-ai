from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from ..deps import DB, CurrentUser
from ..models import Activity, AdoptionApplication, Favorite, MatchResult, Message, Notification, Pet, Sighting, User
from ..schemas import (
    ActivityOut,
    DashboardOut,
    MatchOut,
    MessageOut,
    NotificationOut,
    PetOut,
    ReportSummary,
    SettingsIn,
    UserOut,
)
from ..services.serialize import application_out, favorite_ids, pet_out

router = APIRouter(prefix="/api/me", tags=["me"])

PET_LOAD = (selectinload(Pet.photos), selectinload(Pet.shelter))


@router.patch("", response_model=UserOut)
def update_settings(data: SettingsIn, user: CurrentUser, db: DB):
    changes = data.model_dump(exclude_unset=True)
    if "email" in changes:
        changes["email"] = changes["email"].lower()
        taken = db.scalar(select(User).where(func.lower(User.email) == changes["email"], User.id != user.id))
        if taken:
            raise HTTPException(status.HTTP_409_CONFLICT, "That email is already used by another account.")
    for k, v in changes.items():
        setattr(user, k, v.strip() if isinstance(v, str) else v)
    db.commit()
    return user


# ---------- favourites ----------


@router.get("/favorites", response_model=list[PetOut])
def list_favorites(user: CurrentUser, db: DB):
    pets = db.scalars(
        select(Pet).join(Favorite, Favorite.pet_id == Pet.id).where(Favorite.user_id == user.id)
        .options(*PET_LOAD).order_by(Favorite.created_at.desc())
    ).all()
    favs = {p.id for p in pets}
    return [pet_out(p, user, favs) for p in pets]


@router.put("/favorites/{pet_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_favorite(pet_id: int, user: CurrentUser, db: DB):
    if not db.get(Pet, pet_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pet not found.")
    if not db.get(Favorite, (user.id, pet_id)):
        db.add(Favorite(user_id=user.id, pet_id=pet_id))
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/favorites/{pet_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_favorite(pet_id: int, user: CurrentUser, db: DB):
    fav = db.get(Favorite, (user.id, pet_id))
    if fav:
        db.delete(fav)
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------- notifications ----------


@router.get("/notifications", response_model=list[NotificationOut])
def list_notifications(user: CurrentUser, db: DB):
    return db.scalars(
        select(Notification).where(Notification.user_id == user.id).order_by(Notification.created_at.desc()).limit(50)
    ).all()


@router.post("/notifications/{notif_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def read_notification(notif_id: int, user: CurrentUser, db: DB):
    db.execute(update(Notification).where(Notification.id == notif_id, Notification.user_id == user.id).values(unread=False))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/notifications/read-all", status_code=status.HTTP_204_NO_CONTENT)
def read_all_notifications(user: CurrentUser, db: DB):
    db.execute(update(Notification).where(Notification.user_id == user.id).values(unread=False))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------- messages ----------


@router.get("/messages", response_model=list[MessageOut])
def inbox(user: CurrentUser, db: DB):
    msgs = db.scalars(
        select(Message).where(Message.recipient_id == user.id)
        .options(selectinload(Message.sender), selectinload(Message.pet)).order_by(Message.created_at.desc()).limit(100)
    ).all()
    return [
        MessageOut(id=m.id, pet_id=m.pet_id, body=m.body, created_at=m.created_at, read=m.read,
                   sender_name=m.sender.name, pet_name=m.pet.name)
        for m in msgs
    ]


# ---------- dashboard ----------


@router.get("/dashboard", response_model=DashboardOut)
def dashboard(user: CurrentUser, db: DB):
    favs = favorite_ids(db, user)

    my_pets = db.scalars(
        select(Pet).where(Pet.owner_id == user.id).options(*PET_LOAD).order_by(Pet.resolved, Pet.created_at.desc())
    ).all()
    my_ids = [p.id for p in my_pets]
    sight_counts = dict(db.execute(
        select(Sighting.pet_id, func.count()).where(Sighting.pet_id.in_(my_ids)).group_by(Sighting.pet_id)
    ).all()) if my_ids else {}

    match_rows = db.scalars(
        select(MatchResult)
        .where(MatchResult.query_pet_id.in_(my_ids), MatchResult.dismissed.is_(False), MatchResult.score >= 30)
        .options(selectinload(MatchResult.candidate).selectinload(Pet.photos),
                 selectinload(MatchResult.query_pet).selectinload(Pet.photos))
        .order_by(MatchResult.score.desc())
    ).all() if my_ids else []
    match_rows = [m for m in match_rows if not m.candidate.resolved and not m.query_pet.resolved]
    match_counts: dict[int, int] = {}
    for m in match_rows:
        match_counts[m.query_pet_id] = match_counts.get(m.query_pet_id, 0) + 1

    apps = db.scalars(
        select(AdoptionApplication).where(AdoptionApplication.user_id == user.id)
        .options(selectinload(AdoptionApplication.pet).selectinload(Pet.photos),
                 selectinload(AdoptionApplication.pet).selectinload(Pet.shelter))
        .order_by(AdoptionApplication.created_at.desc())
    ).all()

    saved = db.scalars(
        select(Pet).join(Favorite, Favorite.pet_id == Pet.id).where(Favorite.user_id == user.id)
        .options(*PET_LOAD).order_by(Favorite.created_at.desc())
    ).all()

    notifs = db.scalars(
        select(Notification).where(Notification.user_id == user.id).order_by(Notification.created_at.desc()).limit(30)
    ).all()
    activity = db.scalars(
        select(Activity).where(Activity.user_id == user.id).order_by(Activity.created_at.desc()).limit(8)
    ).all()

    return DashboardOut(
        reports=[
            ReportSummary(pet=pet_out(p, user, favs), sightings=sight_counts.get(p.id, 0), matches=match_counts.get(p.id, 0))
            for p in my_pets
        ],
        matches=[
            MatchOut(id=m.id, query=pet_out(m.query_pet, user, favs), candidate=pet_out(m.candidate, user, favs),
                     score=m.score, is_new=not m.seen, created_at=m.created_at)
            for m in match_rows[:10]
        ],
        applications=[application_out(a) for a in apps],
        saved=[pet_out(p, user, favs) for p in saved],
        notifications=[NotificationOut.model_validate(n) for n in notifs],
        activity=[ActivityOut.model_validate(a) for a in activity],
        unread=sum(1 for n in notifs if n.unread),
    )
