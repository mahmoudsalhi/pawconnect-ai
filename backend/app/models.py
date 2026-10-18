from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PetStatus(StrEnum):
    lost = "lost"
    found = "found"
    adopt = "adopt"


class AccountType(StrEnum):
    owner = "owner"
    adopter = "adopter"
    shelter = "shelter"


def _enum(e: type[StrEnum], name: str) -> Enum:
    return Enum(e, name=name, values_callable=lambda x: [m.value for m in x])


class Shelter(Base):
    __tablename__ = "shelters"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    city: Mapped[str] = mapped_column(String(80))
    gov: Mapped[str] = mapped_column(String(40))
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    photo_url: Mapped[str] = mapped_column(String(500))
    animal_count: Mapped[int] = mapped_column(Integer, default=0)
    since: Mapped[int] = mapped_column(Integer)
    reply_note: Mapped[str] = mapped_column(String(120))
    focus: Mapped[str] = mapped_column(String(120))

    pets: Mapped[list[Pet]] = relationship(back_populates="shelter")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    account_type: Mapped[AccountType] = mapped_column(_enum(AccountType, "account_type"), default=AccountType.owner)
    phone: Mapped[str] = mapped_column(String(40), default="")
    gov: Mapped[str] = mapped_column(String(40), default="Tunis")
    lang: Mapped[str] = mapped_column(String(20), default="English")
    notify_email: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_sms: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_push: Mapped[bool] = mapped_column(Boolean, default=True)
    show_phone: Mapped[bool] = mapped_column(Boolean, default=False)
    min_match_score: Mapped[int] = mapped_column(Integer, default=60)
    shelter_id: Mapped[int | None] = mapped_column(ForeignKey("shelters.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    shelter: Mapped[Shelter | None] = relationship()
    pets: Mapped[list[Pet]] = relationship(back_populates="owner")


class Pet(Base):
    """A lost report, a found report or an adoption listing."""

    __tablename__ = "pets"

    id: Mapped[int] = mapped_column(primary_key=True)
    ref: Mapped[str] = mapped_column(String(20), unique=True)
    status: Mapped[PetStatus] = mapped_column(_enum(PetStatus, "pet_status"), index=True)
    # Lost/found: reunited with family. Adoption: adopted. Resolved listings leave search results.
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    name: Mapped[str] = mapped_column(String(80))
    species: Mapped[str] = mapped_column(String(20), index=True)
    breed: Mapped[str] = mapped_column(String(80), default="")
    age_label: Mapped[str] = mapped_column(String(60), default="")
    age_group: Mapped[str] = mapped_column(String(20), default="adult")
    sex: Mapped[str] = mapped_column(String(20), default="Unknown")
    size: Mapped[str] = mapped_column(String(20), default="Medium")
    color: Mapped[str] = mapped_column(String(80), default="")
    chip: Mapped[str] = mapped_column(String(40), default="Unknown")
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    traits: Mapped[list[str]] = mapped_column(JSON, default=list)
    reward: Mapped[str | None] = mapped_column(String(60))
    fee: Mapped[str | None] = mapped_column(String(40))
    # Adoption listings: health and coat details (None = not stated). Inputs to the M3 model.
    vaccinated: Mapped[bool | None] = mapped_column(Boolean)
    dewormed: Mapped[bool | None] = mapped_column(Boolean)
    sterilized: Mapped[bool | None] = mapped_column(Boolean)
    fur_length: Mapped[str | None] = mapped_column(String(10))  # Short / Medium / Long
    ai_breeds: Mapped[list[list]] = mapped_column(JSON, default=list)  # [[label, pct], ...]
    adopt_speed: Mapped[int | None] = mapped_column(Integer)  # PetFinder class 0-4
    listing_tips: Mapped[list[str]] = mapped_column(JSON, default=list)
    background_removed: Mapped[bool] = mapped_column(Boolean, default=False)

    city: Mapped[str] = mapped_column(String(80))
    gov: Mapped[str] = mapped_column(String(40), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    address: Mapped[str] = mapped_column(String(200), default="")
    event_date: Mapped[date] = mapped_column(Date, index=True)
    event_time: Mapped[str] = mapped_column(String(10), default="")

    contact_name: Mapped[str] = mapped_column(String(120))
    contact_role: Mapped[str] = mapped_column(String(40))
    contact_phone: Mapped[str] = mapped_column(String(40), default="")
    contact_email: Mapped[str] = mapped_column(String(254), default="")
    contact_pref: Mapped[str] = mapped_column(String(40), default="In-app messages")
    show_phone: Mapped[bool] = mapped_column(Boolean, default=False)

    views: Mapped[int] = mapped_column(Integer, default=0)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    shelter_id: Mapped[int | None] = mapped_column(ForeignKey("shelters.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped[User | None] = relationship(back_populates="pets")
    shelter: Mapped[Shelter | None] = relationship(back_populates="pets")
    photos: Mapped[list[PetPhoto]] = relationship(
        back_populates="pet", order_by="PetPhoto.position", cascade="all, delete-orphan"
    )
    sightings: Mapped[list[Sighting]] = relationship(
        back_populates="pet", order_by="Sighting.created_at.desc()", cascade="all, delete-orphan"
    )


class PetPhoto(Base):
    __tablename__ = "pet_photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(String(500))
    position: Mapped[int] = mapped_column(Integer, default=0)
    quality: Mapped[int | None] = mapped_column(Integer)

    pet: Mapped[Pet] = relationship(back_populates="photos")


class Sighting(Base):
    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(primary_key=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    place: Mapped[str] = mapped_column(String(200))
    seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str] = mapped_column(Text, default="")
    by_name: Mapped[str] = mapped_column(String(120), default="Anonymous")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    pet: Mapped[Pet] = relationship(back_populates="sightings")


class Favorite(Base):
    __tablename__ = "favorites"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    recipient_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    body: Mapped[str] = mapped_column(Text)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    pet: Mapped[Pet] = relationship()
    sender: Mapped[User] = relationship(foreign_keys=[sender_id])


class AdoptionApplication(Base):
    __tablename__ = "adoption_applications"
    __table_args__ = (UniqueConstraint("user_id", "pet_id", name="uq_application_user_pet"),)

    STAGES = ("Submitted", "Under review", "Home visit", "Decision")

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(40))
    gov: Mapped[str] = mapped_column(String(40))
    housing: Mapped[str] = mapped_column(String(60))
    other_pets: Mapped[str] = mapped_column(String(60))
    hours_alone: Mapped[str] = mapped_column(String(60))
    motivation: Mapped[str] = mapped_column(Text)
    home_visit_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    stage: Mapped[int] = mapped_column(Integer, default=0)
    next_step: Mapped[str] = mapped_column(String(240), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    pet: Mapped[Pet] = relationship()
    user: Mapped[User] = relationship()


class MatchResult(Base):
    """Cached similarity between a query report and a candidate report."""

    __tablename__ = "match_results"
    __table_args__ = (UniqueConstraint("query_pet_id", "candidate_pet_id", name="uq_match_pair"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    query_pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    candidate_pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    score: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(60), default="demo")
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    seen: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_pet: Mapped[Pet] = relationship(foreign_keys=[query_pet_id])
    candidate: Mapped[Pet] = relationship(foreign_keys=[candidate_pet_id])


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    icon: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text, default="")
    link: Mapped[str] = mapped_column(String(200), default="")  # client route, e.g. /pets/12
    unread: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Activity(Base):
    __tablename__ = "activity"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    icon: Mapped[str] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Upload(Base):
    """An uploaded photo, stored in the database so every machine sharing it can serve the file."""

    __tablename__ = "uploads"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "<uuid>.jpg", served at /uploads/<name>
    content_type: Mapped[str] = mapped_column(String(40), default="image/jpeg")
    data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
