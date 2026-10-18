from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

PHONE_HELP = "Enter an 8-digit Tunisian number, e.g. 22 418 903."


def normalise_tn_phone(v: str) -> str:
    digits = "".join(c for c in v if c.isdigit())
    if digits.startswith("216"):
        digits = digits[3:]
    if len(digits) != 8:
        raise ValueError(PHONE_HELP)
    return f"+216 {digits[:2]} {digits[2:5]} {digits[5:]}"


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- auth & users ----------


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    account_type: Literal["owner", "adopter", "shelter"] = "owner"


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str
    remember: bool = True


class UserOut(ORM):
    id: int
    name: str
    email: str
    account_type: str
    phone: str
    gov: str
    lang: str
    notify_email: bool
    notify_sms: bool
    notify_push: bool
    show_phone: bool
    min_match_score: int
    shelter_id: int | None
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SettingsIn(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=120)
    email: EmailStr | None = None
    phone: str | None = None
    gov: str | None = None
    lang: Literal["English", "Français", "العربية"] | None = None
    notify_email: bool | None = None
    notify_sms: bool | None = None
    notify_push: bool | None = None
    show_phone: bool | None = None
    min_match_score: int | None = Field(None, ge=30, le=95)

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str | None) -> str | None:
        return normalise_tn_phone(v) if v else v


# ---------- shelters ----------


class ShelterOut(ORM):
    id: int
    slug: str
    name: str
    city: str
    gov: str
    lat: float
    lng: float
    photo_url: str
    animal_count: int
    since: int
    reply_note: str
    focus: str
    listed: int = 0


class InsightRow(BaseModel):
    pet_id: int
    name: str
    photo: str
    species: str
    age_label: str
    days_listed: int
    speed_class: int
    tips: list[str]


# ---------- pets ----------


class ShelterBrief(ORM):
    id: int
    slug: str
    name: str
    reply_note: str
    animal_count: int


class PetOut(BaseModel):
    id: int
    ref: str
    status: str
    resolved: bool
    name: str
    species: str
    breed: str
    age_label: str
    age_group: str
    sex: str
    size: str
    color: str
    chip: str
    description: str
    tags: list[str]
    traits: list[str]
    reward: str | None
    fee: str | None
    ai_breeds: list[list]
    adopt_speed: int | None
    city: str
    gov: str
    lat: float
    lng: float
    event_date: date
    event_time: str
    days_ago: int
    contact_name: str
    contact_role: str
    contact_phone: str | None
    contact_pref: str
    photos: list[str]
    shelter: ShelterBrief | None
    distance_km: float
    is_favorite: bool
    is_mine: bool
    views: int
    background_removed: bool
    created_at: datetime


class SightingOut(ORM):
    id: int
    place: str
    seen_at: datetime | None
    note: str
    by_name: str
    created_at: datetime


class PetDetailOut(PetOut):
    sightings: list[SightingOut]
    contact_note: str


class PetPage(BaseModel):
    items: list[PetOut]
    total: int
    page: int
    pages: int
    counts: dict[str, int]


class PetCreate(BaseModel):
    status: Literal["lost", "found"]
    name: str = Field("", max_length=80)
    species: Literal["Dog", "Cat", "Other"]
    breed: str = Field("", max_length=80)
    age_label: str = Field("Not sure", max_length=60)
    sex: Literal["Male", "Female", "Unknown"] = "Unknown"
    size: Literal["Small", "Medium", "Large"] = "Medium"
    color: str = Field(min_length=1, max_length=80)
    chip: Literal["Yes", "No", "Unknown"] = "Unknown"
    description: str = Field(min_length=20, max_length=600)
    tags: list[str] = Field(default_factory=list, max_length=12)
    photos: list[str] = Field(min_length=1, max_length=6)
    background_removed: bool = False
    lat: float = Field(ge=30, le=38)
    lng: float = Field(ge=7, le=12)
    address: str = Field("", max_length=200)
    event_date: date
    event_time: str = Field("", max_length=10)
    contact_name: str = Field(min_length=2, max_length=120)
    contact_phone: str
    contact_email: EmailStr
    contact_pref: Literal["Phone call", "WhatsApp", "In-app messages"] = "Phone call"
    show_phone: bool = False

    @field_validator("contact_phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalise_tn_phone(v)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class PetCreated(BaseModel):
    pet: PetOut
    pool: int
    match_count: int


class SightingIn(BaseModel):
    place: str = Field(min_length=2, max_length=200)
    seen_at: datetime | None = None
    note: str = Field("", max_length=1000)


class MessageIn(BaseModel):
    body: str = Field(min_length=5, max_length=2000)


class MessageOut(ORM):
    id: int
    pet_id: int
    body: str
    created_at: datetime
    read: bool
    sender_name: str
    pet_name: str


# ---------- adoption ----------


class ApplicationIn(BaseModel):
    pet_id: int
    full_name: str = Field(min_length=3, max_length=120)
    phone: str
    gov: str = Field(max_length=40)
    housing: Literal["Apartment", "House with garden", "House without garden"]
    other_pets: Literal["None", "Dogs", "Cats", "Dogs and cats"]
    hours_alone: Literal["Less than 2 hours", "2–4 hours", "4–6 hours", "More than 6 hours"]
    motivation: str = Field(min_length=30, max_length=2000)
    home_visit_ok: bool

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalise_tn_phone(v)

    @field_validator("home_visit_ok")
    @classmethod
    def _visit(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Shelters need to agree a home visit before an adoption.")
        return v


class ApplicationOut(BaseModel):
    id: int
    code: str
    pet_id: int
    pet_name: str
    pet_photo: str
    shelter_name: str
    stage: int
    stage_label: str
    next_step: str
    created_at: datetime


class StageIn(BaseModel):
    stage: int = Field(ge=0, le=3)
    next_step: str = Field("", max_length=240)


# ---------- notifications & dashboard ----------


class NotificationOut(ORM):
    id: int
    icon: str
    title: str
    body: str
    link: str
    unread: bool
    created_at: datetime


class ActivityOut(ORM):
    id: int
    icon: str
    text: str
    created_at: datetime


class MatchOut(BaseModel):
    id: int
    query: PetOut
    candidate: PetOut
    score: int
    is_new: bool
    created_at: datetime


class ReportSummary(BaseModel):
    pet: PetOut
    sightings: int
    matches: int


class DashboardOut(BaseModel):
    reports: list[ReportSummary]
    matches: list[MatchOut]
    applications: list[ApplicationOut]
    saved: list[PetOut]
    notifications: list[NotificationOut]
    activity: list[ActivityOut]
    unread: int


# ---------- AI ----------


class AnalyzeIn(BaseModel):
    photos: list[str] = Field(min_length=1, max_length=6)
    species: str = "Dog"


class AnalyzeOut(BaseModel):
    model: str
    breeds: list[list]
    quality: list[dict]
    cutout: str | None


class TagsIn(BaseModel):
    text: str = Field("", max_length=2000)


class TagOut(BaseModel):
    label: str
    confidence: float


class MatchIn(BaseModel):
    pet_id: int | None = None
    photo: str | None = None
    species: Literal["Dog", "Cat"]
    status: Literal["found", "lost", "both"] = "found"
    area: str = "ennasr"
    radius: int = Field(20, ge=1, le=300)
    threshold: int = Field(50, ge=0, le=95)


class MatchHit(BaseModel):
    pet: PetOut
    score: int
    distance_from_area: float


class MatchRunOut(BaseModel):
    model: str
    pool: int
    area_label: str
    area_lat: float
    area_lng: float
    signature: list[float]  # first 48 values, for the UI bars
    signature_dim: int  # full length of the embedding the signature was cut from
    results: list[MatchHit]


class AIInfoOut(BaseModel):
    provider: str  # AIProvider.name: "demo" for placeholder outputs, otherwise the trained model's id


class StatsOut(BaseModel):
    lost: int
    found: int
    adopt: int
    reunited: int
    shelters: int
