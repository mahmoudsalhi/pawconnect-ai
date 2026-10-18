"""Deterministic placeholder models. Same input, same output, no ML dependencies."""

from __future__ import annotations

import hashlib
import re

from . import compatibility
from .base import (
    AdoptionSpeed,
    AgePrediction,
    AIProvider,
    BreedPrediction,
    CompatMatch,
    CompatPet,
    ImageInput,
    ListingFeatures,
    MatchSubject,
    PhotoQuality,
    Questionnaire,
    ScamAssessment,
    TagSuggestion,
)
from .scam import assess as assess_scam

TAG_RULES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"collar", re.I), "Wears a collar", 0.91),
    (re.compile(r"friendly|gentle|sweet|calm|affection", re.I), "Friendly", 0.86),
    (re.compile(r"shy|nervous|scared|afraid", re.I), "Shy or nervous", 0.82),
    (re.compile(r"chip", re.I), "Microchipped", 0.94),
    (re.compile(r"neuter|spay|steril", re.I), "Neutered / spayed", 0.88),
    (re.compile(r"injur|limp|wound|blood", re.I), "May need vet care", 0.79),
    (re.compile(r"puppy|kitten", re.I), "Young animal", 0.84),
    (re.compile(r"white (patch|spot|chest|paw|mark)", re.I), "White markings", 0.77),
    (re.compile(r"indoor|apartment|balcony|flat", re.I), "Indoor pet", 0.73),
    (re.compile(r"tag|answers|name", re.I), "Answers to name", 0.70),
    (re.compile(r"harness|lead|leash", re.I), "On a harness or lead", 0.75),
]

BREEDS = {
    "Dog": [
        [("Beagle", 58), ("Basset hound", 17), ("English cocker spaniel", 9)],
        [("American pit bull terrier", 31), ("Staffordshire bull terrier", 24), ("Boxer", 11)],
        [("Leonberger", 41), ("Great Pyrenees", 22), ("Newfoundland", 13)],
        [("Miniature pinscher", 39), ("Chihuahua", 21), ("Basset hound", 14)],
    ],
    "Cat": [
        [("Egyptian Mau", 46), ("Bengal", 21), ("Abyssinian", 12)],
        [("Persian", 44), ("Ragdoll", 29), ("Birman", 14)],
        [("Abyssinian", 38), ("Bengal", 24), ("Maine Coon", 17)],
    ],
}

QUALITY = [84, 66, 73, 58, 79, 70]

# Placeholder trait reading for M9: (trait, raises it, lowers it)
TRAIT_WORDS = [
    ("energy", r"energetic|endless energy|puppy energy|active|playful|loves to run", r"calm|couch|lap cat|relaxed|naps"),
    ("children_friendly", r"good with (?:kids|children)|great with children|gentle with children", r"no (?:young )?children|adult household"),
    ("apartment_friendly", r"apartment|small[- ]space|small apartment", r"garden|space to run"),
    ("novice_owner_friendly", r"house-trained|easy|first-time", r"experienced owner"),
    ("good_with_other_dogs", r"good with (?:other )?dogs|other dogs", r"only pet|not good with dogs"),
]


def _h(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", s.lower()) if len(w) > 2 and w not in {"unknown", "type", "mix", "and"}}


def quality_tip(score: int) -> str:
    if score >= 80:
        return "Sharp, well lit, face clearly visible."
    if score >= 70:
        return "Good. Crop a little closer to the face."
    if score >= 60:
        return "Slightly dark. Try near a window."
    return "Blurry. Hold still and tap to focus."


class DemoProvider(AIProvider):
    name = "demo"

    def predict_breed(self, images: list[ImageInput], species: str, top: int = 3) -> list[BreedPrediction]:
        options = BREEDS.get(species, BREEDS["Dog"])
        pick = options[_h(*(i.url for i in images), species) % len(options)] if images else options[0]
        return [BreedPrediction(label, pct) for label, pct in pick]

    def score_photo(self, image: ImageInput) -> PhotoQuality:
        score = QUALITY[_h(image.url) % len(QUALITY)]
        return PhotoQuality(score, quality_tip(score))

    def suggest_tags(self, text: str) -> list[TagSuggestion]:
        return [TagSuggestion(label, conf) for rx, label, conf in TAG_RULES if rx.search(text or "")]

    def remove_background(self, image: ImageInput) -> ImageInput | None:
        return None

    def embed(self, image: ImageInput) -> list[float]:
        seed = _h(image.url)
        out = []
        for _ in range(128):
            seed = (seed * 1103515245 + 12345) % 2**32
            out.append(((seed >> 16) % 1000) / 1000)
        return out

    def similarity(self, query: MatchSubject, candidate: MatchSubject) -> int:
        if query.species != candidate.species:
            return 0
        score = 22
        if _words(query.color) & _words(candidate.color):
            score += 30
        if _words(query.breed) & _words(candidate.breed):
            score += 18
        if query.size and query.size == candidate.size:
            score += 8
        if _words(query.description) & _words(candidate.description) & {"collar", "white", "patch", "tabby", "tricolour"}:
            score += 6
        jitter = _h(*(i.url for i in query.images), *(i.url for i in candidate.images)) % 17 - 8
        return max(5, min(95, score + jitter))

    def adoption_speed(self, f: ListingFeatures) -> AdoptionSpeed:
        cls = 1
        tips: list[str] = []
        if f.age_group == "senior":
            cls += 2
        elif f.age_group == "adult":
            cls += 1
        if f.photo_count < 3:
            cls += 1
            tips.append(f"Add {3 - f.photo_count} more photo{'s' if 3 - f.photo_count > 1 else ''}")
        if len(f.description) < 120:
            tips.append("Describe their personality in the first line")
        if f.fee and f.fee[:1].isdigit() and int(re.sub(r"\D", "", f.fee) or 0) >= 50 and f.age_group != "young":
            tips.append("Waive or lower the fee")
        if f.size == "Large":
            cls += 0 if f.age_group == "young" else 1
            tips.append("Add photos with people")
        cls = max(0, min(4, cls - (1 if f.age_group == "young" else 0)))
        return AdoptionSpeed(cls, tips or ["Listing is in good shape"])

    def compatibility(self, answers: Questionnaire, pets: list[CompatPet]) -> list[CompatMatch]:
        traits = []
        for p in pets:
            t = {name: 0.5 + 0.35 * bool(re.search(up, p.text, re.I)) - 0.35 * bool(re.search(down, p.text, re.I))
                 for name, up, down in TRAIT_WORDS}
            t["grooming"] = {"Long": 0.9, "Medium": 0.5}.get(p.fur_length, 0.2)
            traits.append(t)
        wish = _words(answers.ideal)
        ideal = [(min(1.0, len(wish & _words(p.text)) / 3), "") for p in pets] if wish else None
        return compatibility.score(answers, pets, traits, ideal)

    def predict_age(self, images: list[ImageInput], species: str) -> AgePrediction | None:
        # No real guess: "not sure" with 0% so the report form never pre-selects a made-up age.
        return AgePrediction(group=None, confidence=0)

    def detect_scam(self, text: str) -> ScamAssessment:
        # Not a placeholder: the real M6 model (ml/m6). Providers that subclass DemoProvider inherit it.
        return assess_scam(text)
