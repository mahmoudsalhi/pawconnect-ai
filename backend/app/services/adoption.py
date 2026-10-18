from ..ai import get_provider
from ..ai.base import AdoptionSpeed, ListingFeatures
from ..models import Pet


def predict_speed(pet: Pet) -> AdoptionSpeed:
    """Adoption speed class (M3) and listing tips for an adoption listing, from its current details."""
    return get_provider().adoption_speed(ListingFeatures(
        species=pet.species, age_group=pet.age_group, breed=pet.breed, photo_count=len(pet.photos),
        description=pet.description, fee=pet.fee, size=pet.size, age_label=pet.age_label, sex=pet.sex,
        vaccinated=pet.vaccinated, dewormed=pet.dewormed, sterilized=pet.sterilized, fur_length=pet.fur_length or "",
    ))
