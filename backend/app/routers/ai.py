import dataclasses
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import update

from ..ai import assistant, location
from ..ai import get_provider
from ..ai.base import LocationPrediction, LocationQuery
from ..ai.location import LocationModelUnavailable
from ..deps import DB, CurrentUser, OptionalUser
from ..geo import AREAS, km
from ..models import MatchResult, Pet, PetStatus
from ..schemas import AnalyzeIn, AnalyzeOut, AssistantOut, LocationOut, MatchHit, MatchIn, MatchRunOut, TagOut, TagsIn
from ..services.matching import image_input, open_reports, scores_for_pet, scores_for_photo
from ..services.serialize import favorite_ids, pet_out

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.post("/analyze", response_model=AnalyzeOut)
def analyze_photos(data: AnalyzeIn):
    """Breed suggestion (M1), photo quality (M5) and background removal (M6) for report photos."""
    provider = get_provider()
    images = [image_input(u) for u in data.photos]
    breeds = provider.predict_breed(images, data.species)
    quality = [{"url": i.url, **provider.score_photo(i).__dict__} for i in images]
    cutout = provider.remove_background(images[0])
    return AnalyzeOut(model=provider.name, breeds=[[b.label, b.confidence] for b in breeds], quality=quality,
                      cutout=cutout.url if cutout else None)


@router.post("/tags", response_model=list[TagOut])
def suggest_tags(data: TagsIn):
    """Structured tags from a description (M4)."""
    return [TagOut(label=t.label, confidence=t.confidence) for t in get_provider().suggest_tags(data.text)]


# Tunisia is UTC+1 all year (no daylight saving since 2009); report dates and times are local.
TUNIS = timezone(timedelta(hours=1))


def _open_lost(db, pet_id: int) -> Pet:
    pet = db.get(Pet, pet_id)
    if not pet:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found.")
    if pet.status != PetStatus.lost or pet.resolved:
        raise HTTPException(status.HTTP_409_CONFLICT, "Search zones are only available for open lost reports.")
    return pet


def _location_query(pet: Pet) -> tuple[LocationQuery, float]:
    """M7 query from the report; uses the same rounded coordinates as the public listing."""
    hh, _, mm = (pet.event_time or "").partition(":")
    hour = int(hh) if hh.isdigit() and int(hh) < 24 else None
    lost_at = datetime.combine(pet.event_date, datetime.min.time(), TUNIS) + timedelta(
        hours=hour if hour is not None else 12, minutes=int(mm) if mm.isdigit() and hour is not None else 0)
    hours = max(1.0, (datetime.now(TUNIS) - lost_at).total_seconds() / 3600)
    query = LocationQuery(species=pet.species.lower(), lat=round(pet.lat, 3), lng=round(pet.lng, 3), hours_missing=hours,
                          hour_of_day=hour, month=pet.event_date.month, age_group=pet.age_group, size=pet.size,
                          breed=pet.breed, sex=pet.sex,
                          microchipped={"yes": True, "no": False}.get((pet.chip or "").strip().lower()))
    return query, hours


def _predict(query: LocationQuery) -> LocationPrediction:
    try:
        pred = get_provider().predict_location(query)
    except LocationModelUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    if pred is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Search zones aren't available for this area or species yet.")
    return pred


def _location_out(pred: LocationPrediction, hours: float) -> LocationOut:
    return LocationOut(model=pred.model, hours_missing=round(hours, 1), cell_m=pred.cell_m,
                       zones=[{"share": z.share, "cells": z.cells} for z in pred.zones], peak=pred.peak,
                       self_return_share=pred.self_return_share, missing_inputs=pred.missing_inputs,
                       recovery=pred.recovery.__dict__ if pred.recovery else None)


@router.get("/location/{pet_id}", response_model=LocationOut)
def predict_location(pet_id: int, db: DB):
    """Search zones and recovery outlook for an open lost report (M7), from the form fields only."""
    query, hours = _location_query(_open_lost(db, pet_id))
    return _location_out(_predict(query), hours)


@router.get("/assistant/{pet_id}", response_model=AssistantOut)
def search_assistant(pet_id: int, db: DB):
    """AI search assistant: Claude reads the description, M7 runs with what it found, Claude writes the plan.

    Without an API key (or if the Claude call fails) this still answers with M7 alone, status
    "unavailable" / "error", so the page never loses its zones.
    """
    pet = _open_lost(db, pet_id)
    query, hours = _location_query(pet)
    try:
        if not location.supported(query.species, query.lat, query.lng):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Search zones aren't available for this area or species yet.")
    except LocationModelUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e

    state, message, facts = "ok", None, None
    try:
        facts = assistant.read_report(query.species, pet.name, pet.description)
    except assistant.AssistantUnavailable as e:
        state, message = "unavailable", str(e)
    except assistant.AssistantError as e:
        state, message = "error", str(e)
    if facts:
        query = dataclasses.replace(
            query,
            indoor_only={"no": True, "yes": False}.get(facts.goes_outside) if query.species == "cat" else None,
            temperament=None if facts.temperament == "unknown" else facts.temperament,
            circumstance=None if facts.circumstance == "unknown" else facts.circumstance,
            collar_tag={"yes": True, "no": False}.get(facts.collar_or_tag))
    pred = _predict(query)

    plan = None
    if state == "ok":
        pet_info = {"id": pet.id, "name": pet.name, "species": query.species, "breed": pet.breed, "color": pet.color,
                    "sex": pet.sex, "age": pet.age_label, "size": pet.size, "city": pet.city, "ref": pet.ref, "contact_name": pet.contact_name,
                    "reward": pet.reward, "hours_missing": hours, "description": pet.description,
                    "last_seen_when": f"{pet.event_date:%d %b %Y} {pet.event_time or ''}".strip()}
        places = assistant.zone_places(location.places_near(query.lat, query.lng), pred.zones, pred.cell_m,
                                       query.lat, query.lng)
        try:
            plan = assistant.write_plan(pet_info, facts, pred, places)
        except assistant.AssistantError as e:
            state, message = "error", str(e)
    return AssistantOut(status=state, message=message,
                        model=assistant.model_name() if state == "ok" else None,
                        facts=facts.model_dump() if facts else None,
                        location=_location_out(pred, hours),
                        plan=plan.model_dump() if plan else None)


@router.post("/match", response_model=MatchRunOut)
def run_match(data: MatchIn, db: DB, user: OptionalUser):
    """Rank open reports by visual similarity (M2) to a report or an uploaded photo."""
    area = AREAS.get(data.area)
    if not area:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown search area.")
    statuses = [PetStatus.found, PetStatus.lost] if data.status == "both" else [PetStatus(data.status)]
    provider = get_provider()

    if data.pet_id:
        query = db.get(Pet, data.pet_id)
        if not query:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found.")
        pool = open_reports(db, data.species, statuses, exclude_id=query.id)
        scores = scores_for_pet(db, query, pool)
        signature_src = query.photos[0].url if query.photos else query.ref
        if user and query.owner_id == user.id:
            db.execute(update(MatchResult).where(MatchResult.query_pet_id == query.id).values(seen=True))
        db.commit()
    elif data.photo:
        pool = open_reports(db, data.species, statuses)
        scores = scores_for_photo(data.photo, data.species, pool)
        signature_src = data.photo
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Choose a report or upload a photo to match.")

    favs = favorite_ids(db, user)
    hits = []
    for p in pool:
        s = scores.get(p.id)
        if s is None or s < data.threshold:
            continue
        d = km(area["lat"], area["lng"], p.lat, p.lng)
        if d <= data.radius:
            hits.append(MatchHit(pet=pet_out(p, user, favs), score=s, distance_from_area=round(d, 2)))
    hits.sort(key=lambda h: -h.score)
    return MatchRunOut(model=provider.name, pool=len(pool), area_label=area["label"], area_lat=area["lat"],
                       area_lng=area["lng"], signature=provider.embed(image_input(signature_src))[:48], results=hits)


@router.post("/match/{query_id}/dismiss/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
def dismiss_match(query_id: int, candidate_id: int, user: CurrentUser, db: DB):
    """Owner of a report marks a suggested match as "not a match"; it stops appearing."""
    query = db.get(Pet, query_id)
    if not query or query.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner of this report can dismiss its matches.")
    db.execute(update(MatchResult).where(MatchResult.query_pet_id == query_id, MatchResult.candidate_pet_id == candidate_id)
               .values(dismissed=True))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
