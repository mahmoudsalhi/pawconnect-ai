"""The contract between PawConnect and its AI models.

Every AI feature in the app goes through an ``AIProvider``. The built-in ``DemoProvider``
returns deterministic placeholder results so the product works end to end. To plug in
your own models, subclass ``AIProvider``, implement the methods you have models for
(inherit from ``DemoProvider`` to keep placeholders for the rest), and set
``AI_PROVIDER=your.module:YourProvider`` in ``backend/.env``.

Model map (matches the labels shown in the UI):
    M1  predict_breed       breed identification (CNN, Oxford-IIIT Pet)
    M2  embed / similarity  lost & found photo matching (ArcFace embedding CNN, DogFaceNet)
    M3  adoption_speed      adoption speed class 0-4 (XGBoost, PetFinder adoption)
    M4  suggest_tags        listing text tagging (TF-IDF + SVM, PetFinder text)
    M5  score_photo         photo engagement score (CNN, PetFinder Pawpularity)
    M6  remove_background   background removal (U-Net, Oxford-IIIT Pet)
    M9  compatibility       pet-owner compatibility from a questionnaire (sentence embeddings + scoring)
        detect_scam         scam warnings on messages (TF-IDF + logistic regression, ml/m6);
                            "M6" in the team's new numbering. Real model, not a placeholder.
    M7  predict_location    where a lost pet is likely to be now (LightGBM cell scorer, ml/m7_location)

M7 has a default implementation (the bundled model), so providers only override it to swap models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


class ModelUnavailable(RuntimeError):
    """A provider raises this when its model backend can't be reached (e.g. the M2 service is down).

    Callers skip AI matching or answer 503 instead of storing a made-up score.
    """


@dataclass(frozen=True)
class ImageInput:
    """A photo the model should look at. ``url`` is always set (remote seed photos only have a URL);
    photos uploaded to this server can also be read as bytes with ``read()``, loaded on first use."""

    url: str
    loader: Callable[[], bytes | None] | None = field(default=None, repr=False, compare=False)

    def read(self) -> bytes | None:
        """The image bytes for photos stored by this server, or None for remote URLs."""
        return self.loader() if self.loader else None


@dataclass(frozen=True)
class BreedPrediction:
    label: str
    confidence: int  # percent, 0-100


@dataclass(frozen=True)
class PhotoQuality:
    score: int  # 0-100
    tip: str


@dataclass(frozen=True)
class TagSuggestion:
    label: str
    confidence: float  # 0-1


@dataclass(frozen=True)
class AdoptionSpeed:
    speed_class: int  # 0 same day, 1 within a week, 2 within a month, 3 within 1-3 months, 4 100+ days
    tips: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgePrediction:
    group: str | None  # "young" | "adult" | "senior", or None when the model is not sure
    confidence: int  # percent, 0-100 (0 = no real prediction)


@dataclass(frozen=True)
class MatchSubject:
    """What the matcher knows about one side of a comparison."""

    images: list[ImageInput]
    species: str
    breed: str = ""
    color: str = ""
    size: str = ""
    description: str = ""
    pet_id: int | None = None


@dataclass(frozen=True)
class ListingFeatures:
    species: str
    age_group: str
    breed: str
    photo_count: int
    description: str
    fee: str | None
    size: str
    age_label: str = ""  # free text such as "4 months" or "2 years"
    sex: str = "Unknown"
    vaccinated: bool | None = None  # None = not stated
    dewormed: bool | None = None
    sterilized: bool | None = None
    fur_length: str = ""  # "Short", "Medium", "Long" or "" if not stated


@dataclass(frozen=True)
class Questionnaire:
    """An adopter's answers for compatibility matching (M9)."""

    species: str = "Any"  # Dog, Cat, Any
    home: str = "apartment"  # apartment, house, house_garden
    children: str = "none"  # none, older (6+), young (under 6)
    other_pets: str = "none"  # none, dogs, cats, both
    activity: str = "moderate"  # relaxed, moderate, active
    hours_alone: str = "4to8"  # under4, 4to8, over8
    experience: str = "some"  # first, some, experienced
    grooming: str = "some"  # minimal, some, happy
    age: str = "any"  # young, adult, senior, any
    size: str = "any"  # Small, Medium, Large, any
    ideal: str = ""  # free text: "describe your ideal pet"


@dataclass(frozen=True)
class CompatPet:
    """What the matcher knows about one adoptable pet."""

    pet_id: int
    species: str
    size: str
    age_group: str
    fur_length: str
    text: str  # description and traits


@dataclass(frozen=True)
class CompatReason:
    text: str
    good: bool


@dataclass(frozen=True)
class CompatMatch:
    pet_id: int
    score: int  # 0-100
    reasons: list[CompatReason] = field(default_factory=list)


@dataclass(frozen=True)
class ScamAssessment:
    """How likely a message is a scam. The app only ever warns; it never blocks a message."""

    score: float  # 0-1
    level: str  # "none", "caution" (soft "be careful" banner) or "high" (likely scam)
    labels: list[str] = field(default_factory=list)  # e.g. ["upfront_payment"], most likely first
    reasons: list[str] = field(default_factory=list)  # one plain sentence per label, for the UI
    model: str = ""


@dataclass(frozen=True)
class LocationQuery:
    """What the platform knows about a lost pet. Unknown fields stay None."""

    species: str                    # "cat" / "dog"
    lat: float                      # last-seen point (use the public, rounded coordinates)
    lng: float
    hours_missing: float
    hour_of_day: int | None = None  # local hour it went missing
    month: int | None = None
    age_group: str | None = None    # kitten / puppy / adult / senior
    size: str | None = None         # small / medium / large
    breed: str = ""
    indoor_only: bool | None = None # cats: True = never goes outside
    temperament: str | None = None
    circumstance: str | None = None
    collar_tag: bool | None = None
    sex: str | None = None          # male / female
    microchipped: bool | None = None
    home_lat: float | None = None
    home_lng: float | None = None


@dataclass(frozen=True)
class SearchZone:
    """One band of the search map. The 50 band holds the most likely cells adding up to 50%;
    the 80 band the next cells up to 80% (so 50 + 80 bands = the 80% zone); then 95."""

    share: int
    cells: list[tuple[float, float]] = field(default_factory=list)  # cell centres (lat, lng), this band only


@dataclass(frozen=True)
class RecoveryOutlook:
    """Chances from now on, given the pet has not been found yet."""

    found_alive: float              # from now on: found alive within the horizon (60 days after it went missing)
    found_alive_at_start: float     # the same chance at the moment it went missing
    within_24h: float
    within_7d: float
    median_hours_if_found: float | None  # typical extra time until found, among pets that are found
    beyond_horizon: bool            # missing longer than the model was trained for
    horizon_days: int
    # Cats whose report doesn't say if they go outside: the chance under each answer.
    by_lifestyle: dict[str, float] | None = None


@dataclass(frozen=True)
class LocationPrediction:
    model: str
    cell_m: float                   # cell size of the zones
    zones: list[SearchZone]         # bands 50, 80, 95
    peak: tuple[float, float]       # most likely cell
    self_return_share: float        # how often similar pets came home by themselves
    missing_inputs: list[str]       # report fields that would sharpen the estimate
    recovery: RecoveryOutlook | None = None


class AIProvider(ABC):
    #: Short identifier stored next to cached results, e.g. "demo" or "siamese-v2".
    name: str = "custom"

    @abstractmethod
    def predict_breed(self, images: list[ImageInput], species: str, top: int = 3) -> list[BreedPrediction]:
        """Top breed guesses, best first. species "Dog"/"Cat" limits guesses to that species, anything else allows both."""

    @abstractmethod
    def score_photo(self, image: ImageInput) -> PhotoQuality:
        """Engagement / quality score for one listing photo, with a practical tip."""

    @abstractmethod
    def suggest_tags(self, text: str) -> list[TagSuggestion]:
        """Structured tags extracted from a free-text description."""

    @abstractmethod
    def remove_background(self, image: ImageInput) -> ImageInput | None:
        """A cut-out version of the photo, or None if not supported (the UI then shows a preview mask)."""

    @abstractmethod
    def embed(self, image: ImageInput) -> list[float]:
        """Visual signature vector for an image (the UI draws the first 48 values)."""

    @abstractmethod
    def similarity(self, query: MatchSubject, candidate: MatchSubject) -> int:
        """Visual similarity between two animals, 0-100."""

    @abstractmethod
    def adoption_speed(self, features: ListingFeatures) -> AdoptionSpeed:
        """Predicted adoption speed class and the listing improvements most likely to help."""

    @abstractmethod
    def compatibility(self, answers: Questionnaire, pets: list[CompatPet]) -> list[CompatMatch]:
        """M9: compatibility of each pet with an adopter's questionnaire, best first. Pets of an excluded
        species are left out."""

    def predict_age(self, images: list[ImageInput], species: str) -> AgePrediction | None:
        """Age group from a report's photos. Not abstract, so existing providers keep working;
        None means the provider has no age model (the API then returns no suggestion)."""
        return None

    def look_signature(self, images: list[ImageInput]) -> list[float] | None:
        """How the animal looks, as an L2-normalised vector averaged over its photos (M1 look-alikes).
        Not abstract; None means the provider has no image model (the API then returns no look-alikes)."""
        return None

    def detect_scam(self, text: str) -> ScamAssessment:
        """Scam warning level and reasons for a message or listing text."""

    def predict_location(self, query: LocationQuery) -> LocationPrediction | None:
        """Search zones for a lost pet, or None when the place is outside the mapped regions.
        Raises LocationModelUnavailable when the model or its libraries are not installed."""
        from .location import predict

        return predict(query)
