"""Recompute the stored adoption speed (M3) and tips for every open adoption listing.

Run after retraining or switching M3_WEIGHTS:  .venv\\Scripts\\python -m app.rescore_adoptions
"""

from sqlalchemy import select

from .config import get_settings
from .database import SessionLocal
from .models import Pet, PetStatus
from .services.adoption import predict_speed

SPEED = ["Same day", "1-7 days", "8-30 days", "31-90 days", "100+ days"]


def main() -> None:
    print(f"M3 weights: {get_settings().m3_weights}")
    db = SessionLocal()
    for pet in db.scalars(select(Pet).where(Pet.status == PetStatus.adopt).order_by(Pet.id)):
        speed = predict_speed(pet)
        pet.adopt_speed, pet.listing_tips = speed.speed_class, speed.tips
        print(f"  {pet.name:10} {SPEED[speed.speed_class]:11} {speed.tips}")
    db.commit()
    db.close()


if __name__ == "__main__":
    main()
