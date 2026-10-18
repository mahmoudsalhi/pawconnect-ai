"""The contract between PawConnect and its AI models.

Every AI feature in the app goes through an ``AIProvider``. The built-in ``DemoProvider``
returns deterministic placeholder results so the product works end to end. To plug in
your own models, subclass ``AIProvider``, implement the methods you have models for
(inherit from ``DemoProvider`` to keep placeholders for the rest), and set
``AI_PROVIDER=your.module:YourProvider`` in ``backend/.env``.

Model map (matches the labels shown in the UI):
    M1  predict_breed       breed identification (CNN, Oxford-IIIT Pet)
    M2  embed / similarity  lost & found photo matching (Siamese CNN, PetFinder pairs)
    M3  adoption_speed      adoption speed class 0-4 (XGBoost, PetFinder adoption)
    M4  suggest_tags        listing text tagging (TF-IDF + SVM, PetFinder text)
    M5  score_photo         photo engagement score (CNN, PetFinder Pawpularity)
    M6  remove_background   background removal (U-Net, Oxford-IIIT Pet)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


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
class HealthPrediction:
    status: str  # "healthy" | "skin_problem" | "unsure"
    condition: str | None  # most likely skin problem when status == "skin_problem"
    confidence: int  # percent, 0-100


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


class AIProvider(ABC):
    #: Short identifier stored next to cached results, e.g. "demo" or "siamese-v2".
    name: str = "custom"

    @abstractmethod
    def predict_breed(self, images: list[ImageInput], species: str) -> list[BreedPrediction]:
        """Top breed guesses, best first (the UI shows the first three)."""

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

    def predict_age(self, images: list[ImageInput], species: str) -> AgePrediction | None:
        """Age group from a report's photos. Not abstract, so existing providers keep working;
        None means the provider has no age model (the API then returns no suggestion)."""
        return None

    def predict_health(self, images: list[ImageInput], species: str) -> HealthPrediction | None:
        """Visible skin problems on a dog's photos. An indication only, never a diagnosis.
        None means not available (no model, not a dog, or no readable photo)."""
        return None
