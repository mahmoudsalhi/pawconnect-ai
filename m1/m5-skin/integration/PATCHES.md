# Plugging M5 into PawConnect

Same pattern as M4. Adjust names to what is really in the repo, and use your own provider
class so you do not overwrite teammates' code.

## 1. `backend/app/ai/health.py`
Copy `integration/health.py` as is. `onnxruntime` is already in requirements from M4.

## 2. `backend/app/ai/base.py`
```python
@dataclass
class HealthPrediction:
    status: str             # "healthy" | "skin_problem" | "unsure"
    condition: str | None   # e.g. "ringworm" when status == "skin_problem"
    confidence: int         # 0-100

class AIProvider:
    ...
    def predict_health(self, images: list["ImageInput"], species: str | None) -> HealthPrediction | None:
        return None   # default: feature not available
```

## 3. Your provider class (next to `predict_age`)
```python
from .health import get_health_model

def predict_health(self, images, species):
    model = get_health_model()
    if model is None:
        return None
    r = model.predict([img.read() for img in images], species)
    return HealthPrediction(r.status, r.condition, r.confidence) if r else None
```

## 4. `backend/app/schemas.py` and `routers/ai.py`
```python
class HealthOut(BaseModel):
    status: str
    condition: str | None
    confidence: int

class AnalyzeOut(BaseModel):
    ...
    health: HealthOut | None = None

# routers/ai.py, in /api/ai/analyze
h = provider.predict_health(images, species)
... health=HealthOut(status=h.status, condition=h.condition, confidence=h.confidence) if h else None
```

## 5. `.env`
```
M5_SKIN_MODEL_PATH=/absolute/path/to/m5_skin.onnx   # m5_skin.json in the same folder
```

## 6. Frontend (Report.tsx, under the photo upload or the description)
```tsx
// analysis.health = { status, condition, confidence } | null
const CONDITION_LABEL: Record<string, string> = {
  dermatitis: "Dermatitis", fungal_infections: "Fungal infection",
  hypersensitivity: "Allergic reaction", demodicosis: "Mange (mites)", ringworm: "Ringworm",
};
{analysis?.health?.status === "skin_problem" && (
  <p className="hint warn">
    Possible skin problem: {CONDITION_LABEL[analysis.health.condition ?? ""] ?? "skin issue"} ({analysis.health.confidence}%).
    This is only an indication, not a diagnosis. Please see a vet.
  </p>
)}
{analysis?.health?.status === "healthy" && (
  <p className="hint">No visible skin problem detected ({analysis.health.confidence}%).</p>
)}
{analysis?.health?.status === "unsure" && (
  <p className="hint">The AI cannot judge the skin from this photo. Try a closer, well-lit photo.</p>
)}
```
Show it only for dogs (the model returns `null` for cats). Keep the "not a diagnosis" sentence.
