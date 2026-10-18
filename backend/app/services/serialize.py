from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..geo import km
from ..models import AdoptionApplication, Favorite, Pet, User
from ..schemas import ApplicationOut, PetDetailOut, PetOut, ShelterBrief, SightingOut


def favorite_ids(db: Session, user: User | None) -> set[int]:
    if not user:
        return set()
    return set(db.scalars(select(Favorite.pet_id).where(Favorite.user_id == user.id)))


def days_ago(d: date) -> int:
    return max(0, (date.today() - d).days)


def pet_out(pet: Pet, user: User | None = None, favs: set[int] | None = None, origin: tuple[float, float] | None = None) -> PetOut:
    s = get_settings()
    lat, lng = origin or (s.home_lat, s.home_lng)
    is_mine = bool(user and pet.owner_id == user.id)
    return PetOut(
        id=pet.id,
        ref=pet.ref,
        status=pet.status.value,
        resolved=pet.resolved,
        name=pet.name,
        species=pet.species,
        breed=pet.breed,
        age_label=pet.age_label,
        age_group=pet.age_group,
        sex=pet.sex,
        size=pet.size,
        color=pet.color,
        chip=pet.chip,
        description=pet.description,
        tags=pet.tags or [],
        traits=pet.traits or [],
        reward=pet.reward,
        fee=pet.fee,
        vaccinated=pet.vaccinated,
        dewormed=pet.dewormed,
        sterilized=pet.sterilized,
        fur_length=pet.fur_length,
        ai_breeds=pet.ai_breeds or [],
        adopt_speed=pet.adopt_speed,
        # Public listings only show an approximate area.
        city=pet.city,
        gov=pet.gov,
        lat=round(pet.lat, 3),
        lng=round(pet.lng, 3),
        event_date=pet.event_date,
        event_time=pet.event_time,
        days_ago=days_ago(pet.event_date),
        contact_name=pet.contact_name,
        contact_role=pet.contact_role,
        contact_phone=pet.contact_phone if (pet.show_phone or is_mine) else None,
        contact_pref=pet.contact_pref,
        photos=[p.url for p in pet.photos],
        shelter=ShelterBrief.model_validate(pet.shelter) if pet.shelter else None,
        distance_km=round(km(lat, lng, pet.lat, pet.lng), 2),
        is_favorite=pet.id in (favs or set()),
        is_mine=is_mine,
        views=pet.views,
        background_removed=pet.background_removed,
        created_at=pet.created_at,
    )


def contact_note(pet: Pet) -> str:
    if pet.contact_role == "Shelter" and pet.shelter:
        return pet.shelter.reply_note
    if pet.contact_role == "Owner":
        year = pet.owner.created_at.year if pet.owner else 2025
        return f"Phone verified · Member since {year}"
    if pet.contact_role == "Finder":
        d = days_ago(pet.event_date)
        when = "today" if d == 0 else "yesterday" if d == 1 else f"{d} days ago"
        return f"Phone verified · Reported {when}"
    return "Registered rescue group"


def pet_detail_out(pet: Pet, user: User | None, favs: set[int]) -> PetDetailOut:
    base = pet_out(pet, user, favs)
    return PetDetailOut(
        **base.model_dump(),
        sightings=[SightingOut.model_validate(s) for s in pet.sightings],
        contact_note=contact_note(pet),
    )


NEXT_STEPS = [
    "The shelter will review your application within 2 days.",
    "A volunteer is reading your application.",
    "The shelter will propose a home visit.",
    "The shelter will contact you with their decision.",
]


def application_out(a: AdoptionApplication) -> ApplicationOut:
    return ApplicationOut(
        id=a.id,
        code=a.code,
        pet_id=a.pet_id,
        pet_name=a.pet.name,
        pet_photo=a.pet.photos[0].url if a.pet.photos else "",
        shelter_name=a.pet.shelter.name if a.pet.shelter else a.pet.contact_name,
        stage=a.stage,
        stage_label=AdoptionApplication.STAGES[a.stage],
        next_step=a.next_step or NEXT_STEPS[a.stage],
        created_at=a.created_at,
    )
