"""M9 compatibility scoring: an adopter's questionnaire against each pet's traits.

Trait levels (energy, children, apartment, novice owner, other dogs, grooming) come from the provider: the trained
trait model in ``pet_traits`` for PawModels, keyword placeholders for DemoProvider. Hours alone and cats are not in
the training data, so they use the written rules below. Each answer becomes a weighted 0-1 fit; the score is the
weighted average, and strong fits / poor fits become the reasons shown to the adopter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import CompatMatch, CompatPet, CompatReason, Questionnaire

ACTIVITY = {"relaxed": 0.0, "moderate": 0.5, "active": 1.0}
SIZE_ORDER = ["Small", "Medium", "Large"]
AGE_LABEL = {"young": "Young", "adult": "Adult", "senior": "Senior"}
CATS_OK = re.compile(r"good with cats|with (?:other |another )?(?:young )?cats?\b|other cats", re.I)
CATS_NO = re.compile(r"only (?:cat|pet)|not good with cats|chases cats", re.I)
INDEPENDENT = re.compile(r"independent|quiet when left alone|fine alone|on (?:her|his|their) own terms", re.I)
CLINGY = re.compile(r"follows you|wherever people are|needs company|does not like being alone", re.I)


@dataclass
class Fit:
    value: float  # 0 (poor) to 1 (ideal)
    weight: float
    good: str = ""  # shown when value >= 0.7
    bad: str = ""  # shown when value <= 0.4


def _alone_level(pet: CompatPet) -> float:
    """Written rule: how well the pet copes alone, 0-1. Cats and adults cope better; puppies and kittens worse."""
    level = 0.8 if pet.species == "Cat" else 0.5
    level -= 0.3 if pet.age_group == "young" else 0
    level += 0.2 if INDEPENDENT.search(pet.text) else 0
    level -= 0.2 if CLINGY.search(pet.text) else 0
    return min(1.0, max(0.0, level))


def _cats_level(pet: CompatPet) -> float:
    """Written rule from the description: good with cats or prefers to be the only pet."""
    if CATS_NO.search(pet.text):
        return 0.15
    if CATS_OK.search(pet.text):
        return 0.9
    return 0.7 if pet.species == "Cat" else 0.5


def fits(a: Questionnaire, pet: CompatPet, t: dict[str, float], ideal: tuple[float, str] | None) -> list[Fit]:
    """t holds trait levels from 0 (low) to 1 (high) relative to real breeds."""
    out: list[Fit] = []
    kids, dogs, apt, novice, energy, groom = (t["children_friendly"], t["good_with_other_dogs"], t["apartment_friendly"],
                                              t["novice_owner_friendly"], t["energy"], t["grooming"])

    if a.children == "young":
        out.append(Fit(kids, 3, "Good with young children", "May not suit a home with young children"))
    elif a.children == "older":
        out.append(Fit(0.4 + 0.6 * kids, 2, "Should be fine with older children", "Better suited to an adult-only home"))

    pets_fits = []
    if a.other_pets in ("dogs", "both"):
        pets_fits.append(Fit(dogs, 2, "Gets along with other dogs", "May not get along with your dog"))
    if a.other_pets in ("cats", "both"):
        cats = _cats_level(pet)
        pets_fits.append(Fit(cats, 2, "Gets along with cats",
                             "Prefers to be the only pet" if cats < 0.2 else "Not known to live with cats"))
    out += pets_fits

    if a.home == "apartment":
        out.append(Fit(apt, 2, "Suited to apartment living", "Needs more space than an apartment offers"))
    elif a.home == "house":
        out.append(Fit(0.6 + 0.4 * apt, 1, "Comfortable in a house"))
    else:
        out.append(Fit(1.0, 1, "Will enjoy your garden" if apt < 0.4 else ""))

    want = ACTIVITY.get(a.activity, 0.5)
    good = {"active": "Energetic, a match for your active lifestyle", "relaxed": "Calm, suits a relaxed home"}.get(
        a.activity, "Energy level matches yours")
    out.append(Fit(1 - abs(want - energy), 2, good,
                   "Needs more exercise than you described" if energy > want else "Calmer than you would like"))

    if a.experience == "first":
        out.append(Fit(novice, 1.5, "Easy for a first-time owner", "May be challenging for a first-time owner"))
    elif a.experience == "some":
        out.append(Fit(0.4 + 0.6 * novice, 1, "Easy to care for"))

    if a.grooming == "minimal":
        out.append(Fit(1 - groom, 1, "Low grooming needs", "Needs regular grooming"))
    elif a.grooming == "some":
        out.append(Fit(1 - max(0.0, groom - 0.6) * 1.5, 0.5, "", "Needs more grooming than you planned"))
    elif groom > 0.6:
        out.append(Fit(1.0, 0.5, "Needs regular grooming, which you are happy to do"))

    alone = _alone_level(pet)
    if a.hours_alone == "4to8":
        out.append(Fit(0.5 + 0.5 * alone, 1, "Copes with a working day alone", "May find long days alone hard"))
    elif a.hours_alone == "over8":
        out.append(Fit(alone, 1.5, "Copes well with time alone", "May struggle alone for long hours"))

    if a.age != "any":
        label = AGE_LABEL.get(pet.age_group, pet.age_group.capitalize())
        out.append(Fit(1.0 if pet.age_group == a.age else 0.3, 1, f"{label}, the age you prefer",
                       "Not the age you are looking for"))
    if a.size != "any" and pet.size in SIZE_ORDER and a.size in SIZE_ORDER:
        gap = abs(SIZE_ORDER.index(pet.size) - SIZE_ORDER.index(a.size))
        out.append(Fit([1.0, 0.5, 0.0][gap], 1, f"{pet.size}, the size you prefer", f"{pet.size}, not {a.size.lower()}"))

    if ideal:
        score, sentence = ideal
        out.append(Fit(score, 2, f"Matches what you described: “{sentence}”" if sentence else "Matches what you described"))
    return out


def score(a: Questionnaire, pets: list[CompatPet], traits: list[dict[str, float]],
          ideal: list[tuple[float, str]] | None = None) -> list[CompatMatch]:
    matches = []
    for k, (pet, t) in enumerate(zip(pets, traits)):
        if a.species in ("Dog", "Cat") and pet.species != a.species:
            continue
        fs = fits(a, pet, t, ideal[k] if ideal else None)
        total = sum(f.value * f.weight for f in fs) / sum(f.weight for f in fs)
        good = sorted((f for f in fs if f.value >= 0.7 and f.good), key=lambda f: -f.value * f.weight)[:3]
        bad = sorted((f for f in fs if f.value <= 0.4 and f.bad), key=lambda f: f.value * f.weight)[:3]
        reasons = [CompatReason(f.good, True) for f in good] + [CompatReason(f.bad, False) for f in bad]
        matches.append(CompatMatch(pet.pet_id, round(100 * total), reasons))
    return sorted(matches, key=lambda m: -m.score)


def pet_text(breed: str, description: str, traits: list[str]) -> str:
    return f"{breed}. {description} {'. '.join(traits)}.".strip()
