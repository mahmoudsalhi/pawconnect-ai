from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..ai import get_provider
from ..ai.base import ImageInput, MatchSubject
from ..database import SessionLocal
from ..models import MatchResult, Pet, PetStatus, Upload
from .notify import log_activity, notify


def image_input(url: str) -> ImageInput:
    """Resolve a photo URL to something a model can read; uploads can also be read as bytes."""
    if url.startswith("/uploads/"):
        return ImageInput(url=url, loader=lambda: _upload_bytes(url.removeprefix("/uploads/")))
    return ImageInput(url=url)


def _upload_bytes(name: str) -> bytes | None:
    with SessionLocal() as db:
        upload = db.get(Upload, name)
        return upload.data if upload else None


def subject_for(pet: Pet) -> MatchSubject:
    return MatchSubject(
        images=[image_input(p.url) for p in pet.photos],
        species=pet.species,
        breed=pet.breed,
        color=pet.color,
        size=pet.size,
        description=pet.description,
        pet_id=pet.id,
    )


def open_reports(db: Session, species: str, statuses: list[PetStatus], exclude_id: int | None = None) -> list[Pet]:
    q = (
        select(Pet)
        .options(selectinload(Pet.photos), selectinload(Pet.shelter))
        .where(Pet.species == species, Pet.status.in_(statuses), Pet.resolved.is_(False))
    )
    if exclude_id:
        q = q.where(Pet.id != exclude_id)
    return list(db.scalars(q))


def scores_for_pet(db: Session, query: Pet, candidates: list[Pet]) -> dict[int, int]:
    """Similarity for each candidate, reusing cached results and caching new ones."""
    provider = get_provider()
    cached = {
        m.candidate_pet_id: m
        for m in db.scalars(select(MatchResult).where(MatchResult.query_pet_id == query.id))
    }
    q_subject = subject_for(query)
    out: dict[int, int] = {}
    for cand in candidates:
        hit = cached.get(cand.id)
        if hit is None:
            hit = MatchResult(
                query_pet_id=query.id,
                candidate_pet_id=cand.id,
                score=provider.similarity(q_subject, subject_for(cand)),
                model=provider.name,
            )
            db.add(hit)
        if not hit.dismissed:
            out[cand.id] = hit.score
    db.flush()
    return out


def scores_for_photo(photo: str, species: str, candidates: list[Pet]) -> dict[int, int]:
    provider = get_provider()
    subject = MatchSubject(images=[image_input(photo)], species=species)
    return {c.id: provider.similarity(subject, subject_for(c)) for c in candidates}


def match_new_report(db: Session, pet: Pet) -> tuple[int, int]:
    """Compare a freshly published lost/found report with open reports of the other kind.

    Returns (pool size, possible matches at 50%+). Notifies both sides above their alert threshold.
    """
    other = PetStatus.found if pet.status == PetStatus.lost else PetStatus.lost
    pool = open_reports(db, pet.species, [other], exclude_id=pet.id)
    scores = scores_for_pet(db, pet, pool)
    by_id = {p.id: p for p in pool}
    strong = 0
    for cand_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if score < 50:
            continue
        strong += 1
        cand = by_id[cand_id]
        # Cache the reverse direction so the other owner sees the same score.
        if not db.scalar(select(MatchResult).where(MatchResult.query_pet_id == cand.id, MatchResult.candidate_pet_id == pet.id)):
            db.add(MatchResult(query_pet_id=cand.id, candidate_pet_id=pet.id, score=score, model=get_provider().name))
        if pet.owner and score >= pet.owner.min_match_score:
            notify(
                db, pet.owner_id, "auto_awesome", f"New possible match for {pet.name or 'your report'}",
                f"A {cand.species.lower()} {cand.status.value} in {cand.city} scored {score}% visual similarity.",
                f"/match?pet={pet.id}&selected={cand.id}",
            )
        if cand.owner and score >= cand.owner.min_match_score:
            notify(
                db, cand.owner_id, "auto_awesome", f"New possible match for {cand.name}",
                f"A {pet.species.lower()} {pet.status.value} in {pet.city} scored {score}% visual similarity.",
                f"/match?pet={cand.id}&selected={pet.id}",
            )
            log_activity(db, cand.owner_id, "auto_awesome", f"AI matching found a new possible match for {cand.name}")
    return len(pool), strong
