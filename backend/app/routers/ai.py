from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import update

from ..ai import get_provider
from ..deps import DB, CurrentUser, OptionalUser
from ..geo import AREAS, km
from ..models import MatchResult, Pet, PetStatus
from ..schemas import AnalyzeIn, AnalyzeOut, MatchHit, MatchIn, MatchRunOut, TagOut, TagsIn
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
