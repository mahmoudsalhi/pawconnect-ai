from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..deps import DB, CurrentUser
from ..models import AdoptionApplication, Pet, PetStatus
from ..schemas import ApplicationIn, ApplicationOut, StageIn
from ..services.notify import log_activity, notify
from ..services.serialize import application_out

router = APIRouter(prefix="/api/adoptions", tags=["adoption"])

APP_LOAD = (
    selectinload(AdoptionApplication.pet).selectinload(Pet.photos),
    selectinload(AdoptionApplication.pet).selectinload(Pet.shelter),
)


@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
def apply(data: ApplicationIn, user: CurrentUser, db: DB):
    pet = db.get(Pet, data.pet_id)
    if not pet or pet.status != PetStatus.adopt or pet.resolved:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This animal is no longer available for adoption.")
    existing = db.scalar(select(AdoptionApplication).where(AdoptionApplication.user_id == user.id,
                                                           AdoptionApplication.pet_id == pet.id))
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, f"You already applied to adopt {pet.name}.")
    app = AdoptionApplication(code="tmp", pet_id=pet.id, user_id=user.id, **data.model_dump(exclude={"pet_id"}))
    db.add(app)
    db.flush()
    app.code = f"APP-{2040 + app.id}"
    shelter_name = pet.shelter.name if pet.shelter else pet.contact_name
    notify(db, user.id, "assignment", f"Application sent for {pet.name}",
           f"{shelter_name} usually replies within 2 days.", "/dashboard/applications")
    log_activity(db, user.id, "assignment", f"You applied to adopt {pet.name}")
    db.commit()
    app = db.scalar(select(AdoptionApplication).where(AdoptionApplication.id == app.id).options(*APP_LOAD))
    return application_out(app)


@router.get("/mine", response_model=list[ApplicationOut])
def my_applications(user: CurrentUser, db: DB):
    apps = db.scalars(
        select(AdoptionApplication).where(AdoptionApplication.user_id == user.id).options(*APP_LOAD)
        .order_by(AdoptionApplication.created_at.desc())
    ).all()
    return [application_out(a) for a in apps]


@router.get("/incoming", response_model=list[ApplicationOut])
def shelter_applications(user: CurrentUser, db: DB):
    """Applications for the signed-in shelter account's animals."""
    if not user.shelter_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only shelter accounts can see incoming applications.")
    apps = db.scalars(
        select(AdoptionApplication).join(Pet).where(Pet.shelter_id == user.shelter_id).options(*APP_LOAD)
        .order_by(AdoptionApplication.created_at.desc())
    ).all()
    return [application_out(a) for a in apps]


@router.patch("/{app_id}", response_model=ApplicationOut)
def update_stage(app_id: int, data: StageIn, user: CurrentUser, db: DB):
    app = db.scalar(select(AdoptionApplication).where(AdoptionApplication.id == app_id).options(*APP_LOAD))
    if not app:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Application not found.")
    if not user.shelter_id or app.pet.shelter_id != user.shelter_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the shelter caring for this animal can update the application.")
    app.stage = data.stage
    app.next_step = data.next_step
    notify(db, app.user_id, "event_available", f"Update on your application for {app.pet.name}",
           data.next_step or AdoptionApplication.STAGES[data.stage], "/dashboard/applications")
    db.commit()
    return application_out(app)
