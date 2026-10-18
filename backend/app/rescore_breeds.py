"""Recompute the stored AI breed estimate (M1) for every pet from its photos.

Run after retraining M1 or switching AI_PROVIDER:  .venv\\Scripts\\python -m app.rescore_breeds
Seed photos are remote, so this needs an internet connection.
"""

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .ai import get_provider
from .database import SessionLocal
from .models import Pet
from .services.matching import image_input


def main() -> None:
    provider = get_provider()
    print(f"Provider: {provider.name}")
    db = SessionLocal()
    for pet in db.scalars(select(Pet).options(selectinload(Pet.photos)).order_by(Pet.id)):
        breeds = provider.predict_breed([image_input(p.url) for p in pet.photos[:3]], pet.species)
        if not breeds:
            print(f"  {pet.name:22} skipped (photos could not be loaded)")
            continue
        pet.ai_breeds = [[b.label, b.confidence] for b in breeds]
        print(f"  {pet.name:22} {pet.species:4} listed as {pet.breed[:24]:24} -> "
              + ", ".join(f"{b.label} {b.confidence}%" for b in breeds))
    db.commit()
    db.close()


if __name__ == "__main__":
    main()
