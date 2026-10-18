import json

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from ..ai import get_provider
from ..ai.base import CompatPet, ModelUnavailable, Questionnaire
from ..ai.breed import META_PATH as BREED_META
from ..ai.compatibility import pet_text
from ..deps import DB, CurrentUser, OptionalUser
from ..geo import AREAS, km
from ..models import MatchResult, Pet, PetStatus
from ..schemas import (AgeOut, AIInfoOut, AnalyzeIn, AnalyzeOut, BreedGuess, BreedIn, BreedInfoOut, BreedOut,
                       CompatHit, CompatIn, CompatOut, CompatReasonOut, MatchHit, MatchIn, MatchRunOut, TagOut, TagsIn)
from ..services.matching import image_input, open_reports, scores_for_pet, scores_for_photo
from ..services.serialize import favorite_ids, pet_out

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/info", response_model=AIInfoOut)
def ai_info():
    """Which AI provider is active, so the UI only labels results as demo outputs when they are."""
    return AIInfoOut(provider=get_provider().name)


@router.post("/analyze", response_model=AnalyzeOut)
def analyze_photos(data: AnalyzeIn):
    """Breed suggestion (M1), photo quality (M5), background removal (M6) and age group for report photos."""
    provider = get_provider()
    images = [image_input(u) for u in data.photos]
    breeds = provider.predict_breed(images, data.species)
    quality = [{"url": i.url, **provider.score_photo(i).__dict__} for i in images]
    cutout = provider.remove_background(images[0])
    age = provider.predict_age(images, data.species)
    return AnalyzeOut(model=provider.name, breeds=[[b.label, b.confidence] for b in breeds], quality=quality,
                      cutout=cutout.url if cutout else None,
                      age=AgeOut(group=age.group, confidence=age.confidence) if age else None)


@router.post("/breed", response_model=BreedOut)
def identify_breed(data: BreedIn):
    """Breed identification (M1): top five guesses for one to three photos of the same animal."""
    provider = get_provider()
    guesses = provider.predict_breed([image_input(u) for u in data.photos], data.species, top=5)
    species = _breed_species()
    return BreedOut(model=provider.name, predictions=[
        BreedGuess(label=g.label, confidence=g.confidence, species=species.get(g.label, data.species)) for g in guesses
    ])


@router.get("/breed/info", response_model=BreedInfoOut)
def breed_info():
    """The breeds M1 knows and its test results, for the breed identification page."""
    if not BREED_META.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The breed model has not been trained yet.")
    meta = json.loads(BREED_META.read_text(encoding="utf-8"))
    return BreedInfoOut(model=meta["model"], dataset=meta["dataset"], metrics=meta["metrics"],
                        most_confused=meta["most_confused"],
                        breeds=[BreedGuess(label=l, confidence=0, species=s) for l, s in zip(meta["labels"], meta["species"])])


def _breed_species() -> dict[str, str]:
    if not BREED_META.exists():
        return {}
    meta = json.loads(BREED_META.read_text(encoding="utf-8"))
    return dict(zip(meta["labels"], meta["species"]))


@router.post("/compatibility", response_model=CompatOut)
def compatibility(data: CompatIn, db: DB, user: OptionalUser):
    """Rank open adoption listings by compatibility with the adopter's questionnaire (M9)."""
    pets = list(db.scalars(
        select(Pet).options(selectinload(Pet.photos), selectinload(Pet.shelter))
        .where(Pet.status == PetStatus.adopt, Pet.resolved.is_(False))
    ))
    provider = get_provider()
    matches = provider.compatibility(Questionnaire(**data.model_dump()), [
        CompatPet(pet_id=p.id, species=p.species, size=p.size, age_group=p.age_group, fur_length=p.fur_length or "",
                  text=pet_text(p.breed, p.description, p.traits or []))
        for p in pets
    ])
    by_id, favs = {p.id: p for p in pets}, favorite_ids(db, user)
    return CompatOut(model=provider.name, results=[
        CompatHit(pet=pet_out(by_id[m.pet_id], user, favs), score=m.score,
                  reasons=[CompatReasonOut(text=r.text, good=r.good) for r in m.reasons])
        for m in matches
    ])


@router.post("/tags", response_model=list[TagOut])
def suggest_tags(data: TagsIn):
    """Structured tags from a description (M4)."""
    return [TagOut(label=t.label, confidence=t.confidence) for t in get_provider().suggest_tags(data.text)]


@router.post("/match", response_model=MatchRunOut)
def run_match(data: MatchIn, db: DB, user: OptionalUser):
    """Rank open reports by visual similarity (M2) to a report or an uploaded photo."""
    area = AREAS.get(data.area)
    if not area:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown search area.")
    statuses = [PetStatus.found, PetStatus.lost] if data.status == "both" else [PetStatus(data.status)]
    provider = get_provider()

    try:
        return _run_match(data, db, user, area, statuses, provider)
    except ModelUnavailable:
        db.rollback()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The photo-matching model is not available right now.")


def _run_match(data, db, user, area, statuses, provider):
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
    signature = provider.embed(image_input(signature_src))
    return MatchRunOut(model=provider.name, pool=len(pool), area_label=area["label"], area_lat=area["lat"],
                       area_lng=area["lng"], signature=signature[:48], signature_dim=len(signature), results=hits)


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
