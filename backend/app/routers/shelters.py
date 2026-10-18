from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..deps import DB
from ..models import Pet, PetStatus, Shelter
from ..schemas import InsightRow, ShelterOut, StatsOut
from ..services.adoption import predict_speed
from ..services.serialize import days_ago

router = APIRouter(prefix="/api", tags=["shelters"])


@router.get("/shelters", response_model=list[ShelterOut])
def list_shelters(db: DB):
    listed = dict(db.execute(
        select(Pet.shelter_id, func.count()).where(Pet.resolved.is_(False), Pet.shelter_id.is_not(None)).group_by(Pet.shelter_id)
    ).all())
    return [
        ShelterOut.model_validate(s).model_copy(update={"listed": listed.get(s.id, 0)})
        for s in db.scalars(select(Shelter).order_by(Shelter.id))
    ]


@router.get("/shelters/{slug}/insights", response_model=list[InsightRow])
def shelter_insights(slug: str, db: DB):
    """Predicted adoption speed (M3) for each open listing of a shelter, slowest first."""
    shelter = db.scalar(select(Shelter).where(Shelter.slug == slug))
    if not shelter:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Shelter not found.")
    pets = db.scalars(
        select(Pet).where(Pet.shelter_id == shelter.id, Pet.status == PetStatus.adopt, Pet.resolved.is_(False))
        .options(selectinload(Pet.photos))
    ).all()
    rows = []
    for p in pets:
        speed = predict_speed(p)
        rows.append(InsightRow(pet_id=p.id, name=p.name, photo=p.photos[0].url if p.photos else "", species=p.species,
                               age_label=p.age_label, days_listed=days_ago(p.event_date), speed_class=speed.speed_class,
                               tips=speed.tips))
    return sorted(rows, key=lambda r: -r.speed_class)


@router.get("/stats", response_model=StatsOut)
def stats(db: DB):
    def n(*where):
        return db.scalar(select(func.count()).select_from(Pet).where(*where)) or 0

    return StatsOut(
        lost=n(Pet.status == PetStatus.lost, Pet.resolved.is_(False)),
        found=n(Pet.status == PetStatus.found, Pet.resolved.is_(False)),
        adopt=n(Pet.status == PetStatus.adopt, Pet.resolved.is_(False)),
        reunited=n(Pet.status != PetStatus.adopt, Pet.resolved.is_(True)),
        shelters=db.scalar(select(func.count()).select_from(Shelter)) or 0,
    )
