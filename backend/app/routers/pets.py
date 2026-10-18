from datetime import date, timedelta
from math import ceil
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import selectinload

from ..ai import get_provider
from ..deps import DB, CurrentUser, OptionalUser
from ..geo import km
from ..models import Message, Pet, PetPhoto, PetStatus, Sighting
from ..schemas import (
    MessageIn,
    PetCreate,
    PetCreated,
    PetDetailOut,
    PetOut,
    PetPage,
    SightingIn,
    SightingOut,
)
from ..services.matching import image_input, match_new_report
from ..services.notify import log_activity, notify
from ..services.serialize import favorite_ids, pet_detail_out, pet_out

router = APIRouter(prefix="/api/pets", tags=["pets"])

PET_LOAD = (selectinload(Pet.photos), selectinload(Pet.shelter))
AGE_GROUPS = {"Puppy / kitten (under 1 year)": "young", "Adult (1–7 years)": "adult", "Senior (8+ years)": "senior"}


def _get_pet(db, pet_id: int) -> Pet:
    pet = db.scalar(
        select(Pet).where(Pet.id == pet_id)
        .options(*PET_LOAD, selectinload(Pet.sightings), selectinload(Pet.owner))
    )
    if not pet:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This listing doesn’t exist or was removed.")
    return pet


@router.get("", response_model=PetPage)
def list_pets(
    db: DB,
    user: OptionalUser,
    status_: Literal["all", "lost", "found", "adopt"] = Query("all", alias="status"),
    q: str = "",
    gov: str = "all",
    species: str = "any",
    age: str = "any",
    size: str = "any",
    breed: str = "",
    shelter: str = "any",
    days: int | None = Query(None, ge=1, le=365),
    distance: int = Query(300, ge=1, le=300),
    sort: Literal["recent", "distance", "name"] = "recent",
    page: int = Query(1, ge=1),
    page_size: int = Query(8, ge=1, le=100),
):
    base = select(Pet).where(Pet.resolved.is_(False))
    if species != "any":
        base = base.where(Pet.species == species)
    if age != "any":
        base = base.where(Pet.age_group == age)
    if size != "any":
        base = base.where(Pet.size == size)
    if gov != "all":
        base = base.where(Pet.gov == gov)
    if breed.strip():
        base = base.where(Pet.breed.ilike(f"%{breed.strip()}%"))
    if shelter != "any":
        base = base.where(Pet.shelter.has(slug=shelter))
    if days:
        base = base.where(Pet.event_date >= date.today() - timedelta(days=days))
    if q.strip():
        like = f"%{q.strip()}%"
        base = base.where(or_(Pet.name.ilike(like), Pet.breed.ilike(like), Pet.city.ilike(like), Pet.gov.ilike(like),
                              Pet.color.ilike(like), cast(Pet.traits, String).ilike(like)))

    # Tab counts reflect every filter except the tab itself.
    counted = db.scalars(base.options(*PET_LOAD)).all()
    favs = favorite_ids(db, user)
    rows = [pet_out(p, user, favs) for p in counted]
    if distance < 300:
        rows = [r for r in rows if r.distance_km <= distance]
    counts = {"all": len(rows)} | {s.value: sum(1 for r in rows if r.status == s.value) for s in PetStatus}
    if status_ != "all":
        rows = [r for r in rows if r.status == status_]

    key = {"distance": lambda r: r.distance_km, "name": lambda r: r.name.lower()}.get(sort, lambda r: (r.days_ago, -r.id))
    rows.sort(key=key)
    pages = max(1, ceil(len(rows) / page_size))
    page = min(page, pages)
    return PetPage(items=rows[(page - 1) * page_size: page * page_size], total=len(rows), page=page, pages=pages, counts=counts)


@router.get("/{pet_id}", response_model=PetDetailOut)
def get_pet(pet_id: int, db: DB, user: OptionalUser):
    pet = _get_pet(db, pet_id)
    if not (user and pet.owner_id == user.id):
        pet.views += 1
        db.commit()
    return pet_detail_out(pet, user, favorite_ids(db, user))


@router.get("/{pet_id}/similar", response_model=list[PetOut])
def similar_pets(pet_id: int, db: DB, user: OptionalUser):
    pet = _get_pet(db, pet_id)
    others = db.scalars(
        select(Pet).where(Pet.id != pet.id, Pet.species == pet.species, Pet.resolved.is_(False)).options(*PET_LOAD)
    ).all()
    favs = favorite_ids(db, user)
    others = sorted(others, key=lambda p: (p.status != pet.status, km(pet.lat, pet.lng, p.lat, p.lng)))
    return [pet_out(p, user, favs) for p in others[:4]]


@router.post("", response_model=PetCreated, status_code=status.HTTP_201_CREATED)
def create_report(data: PetCreate, user: CurrentUser, db: DB):
    if data.status == "lost" and not data.name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Enter your pet’s name.")
    if data.event_date > date.today():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The date can’t be in the future.")
    provider = get_provider()
    images = [image_input(u) for u in data.photos]
    breeds = provider.predict_breed(images, data.species)
    city = data.address.split(",")[-1].strip() if data.address else ""
    pet = Pet(
        ref=f"tmp-{uuid4().hex[:12]}",
        status=PetStatus(data.status),
        name=data.name or ("Found " + (data.color.lower() + " " if data.color else "") + data.species.lower()),
        species=data.species,
        breed=data.breed or ("Unknown · " + breeds[0].label.lower() + " type" if breeds else ""),
        age_label=data.age_label,
        age_group=AGE_GROUPS.get(data.age_label, "adult"),
        sex=data.sex,
        size=data.size,
        color=data.color,
        chip=data.chip,
        description=data.description,
        tags=data.tags,
        ai_breeds=[[b.label, b.confidence] for b in breeds],
        background_removed=data.background_removed,
        city=city or "Greater Tunis",
        gov=_gov_for(data.lat, data.lng),
        lat=data.lat,
        lng=data.lng,
        address=data.address,
        event_date=data.event_date,
        event_time=data.event_time,
        contact_name=data.contact_name,
        contact_role="Owner" if data.status == "lost" else "Finder",
        contact_phone=data.contact_phone,
        contact_email=data.contact_email,
        contact_pref=data.contact_pref,
        show_phone=data.show_phone,
        owner_id=user.id,
    )
    pet.photos = [
        PetPhoto(url=u, position=i, quality=provider.score_photo(img).score) for i, (u, img) in enumerate(zip(data.photos, images))
    ]
    db.add(pet)
    db.flush()
    pet.ref = f"PC-TN-{1000 + pet.id:05d}"
    pet.owner = user
    pool, strong = match_new_report(db, pet)
    log_activity(db, user.id, "campaign" if data.status == "lost" else "volunteer_activism",
                 f"You published a {data.status} report for {pet.name}")
    db.commit()
    db.refresh(pet)
    return PetCreated(pet=pet_out(pet, user, favorite_ids(db, user)), pool=pool, match_count=strong)


def _gov_for(lat: float, lng: float) -> str:
    centres = {"Tunis": (36.81, 10.18), "Ariana": (36.87, 10.17), "Ben Arous": (36.75, 10.23), "Manouba": (36.81, 10.09),
               "Nabeul": (36.45, 10.73), "Sousse": (35.83, 10.6), "Sfax": (34.74, 10.76)}
    return min(centres, key=lambda g: km(lat, lng, *centres[g]))


@router.post("/{pet_id}/resolve", response_model=PetOut)
def resolve_report(pet_id: int, user: CurrentUser, db: DB):
    """Mark a report as reunited (lost/found) or adopted (adoption listing)."""
    pet = _get_pet(db, pet_id)
    if pet.owner_id != user.id and not (pet.shelter_id and pet.shelter_id == user.shelter_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the person who published this listing can close it.")
    pet.resolved = True
    pet.resolved_at = func.now()
    log_activity(db, user.id, "favorite", f"{pet.name} was marked as {'adopted' if pet.status == PetStatus.adopt else 'reunited'}")
    db.commit()
    db.refresh(pet)
    return pet_out(pet, user, favorite_ids(db, user))


@router.delete("/{pet_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(pet_id: int, user: CurrentUser, db: DB):
    pet = _get_pet(db, pet_id)
    if pet.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the person who published this report can delete it.")
    db.delete(pet)
    db.commit()


@router.post("/{pet_id}/sightings", response_model=SightingOut, status_code=status.HTTP_201_CREATED)
def add_sighting(pet_id: int, data: SightingIn, db: DB, user: OptionalUser):
    pet = _get_pet(db, pet_id)
    if pet.status != PetStatus.lost:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Sightings can only be added to lost-pet reports.")
    by = "Anonymous"
    if user:
        parts = user.name.split()
        by = parts[0] + (f" {parts[-1][0]}." if len(parts) > 1 else "")
    s = Sighting(pet_id=pet.id, user_id=user.id if user else None, place=data.place.strip(), seen_at=data.seen_at,
                 note=data.note.strip(), by_name=by)
    db.add(s)
    notify(db, pet.owner_id, "visibility", f"New sighting of {pet.name}", data.place.strip(), f"/pets/{pet.id}")
    log_activity(db, pet.owner_id, "visibility", f"Someone reported a sighting of {pet.name} in {data.place.strip()}")
    if user and user.id != pet.owner_id:
        log_activity(db, user.id, "visibility", f"You reported a sighting of {pet.name}")
    db.commit()
    db.refresh(s)
    return s


@router.post("/{pet_id}/messages", status_code=status.HTTP_201_CREATED)
def send_message(pet_id: int, data: MessageIn, user: CurrentUser, db: DB):
    pet = _get_pet(db, pet_id)
    if pet.owner_id == user.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This is your own listing.")
    body = data.body.strip()
    db.add(Message(pet_id=pet.id, sender_id=user.id, recipient_id=pet.owner_id, body=body))
    first = user.name.split()[0]
    # M6: the message is always delivered; a suspicious one only changes how the owner is notified.
    check = get_provider().detect_scam(body)
    if check.level == "none":
        icon, title, note = "chat", f"Message from {first} about {pet.name}", f"“{body[:140]}”"
    else:
        prefix = "Likely scam" if check.level == "high" else "Be careful"
        icon, title, note = "warning", f"{prefix}: message from {first} about {pet.name}", check.reasons[0]
    notify(db, pet.owner_id, icon, title, note, "/dashboard/messages")
    log_activity(db, user.id, "chat", f"You messaged {pet.contact_name} about {pet.name}")
    db.commit()
    return {"ok": True, "recipient": pet.contact_name}
