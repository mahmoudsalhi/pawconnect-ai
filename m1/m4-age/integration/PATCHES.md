# Plugging M4 into PawConnect

These snippets follow the file names and line hints in your plan (`base.py`, `demo.py`,
`routers/ai.py`, `schemas.py`, `Report.tsx`). I have not seen the repo, so adjust names and
imports to what is really there. Agree on the method name and return type with your
teammates first, since everyone edits `base.py`.

## 1. `backend/app/ai/age.py`
Copy `integration/age.py` as is. Add `onnxruntime` to `backend/requirements.txt`.

## 2. `backend/app/ai/base.py`
```python
@dataclass
class AgePrediction:
    group: str | None   # "young" | "adult" | "senior", or None = "Not sure"
    confidence: int     # 0-100

class AIProvider:
    ...
    def predict_age(self, images: list["ImageInput"], species: str | None) -> AgePrediction | None:
        raise NotImplementedError
```

## 3. `backend/app/ai/demo.py` (placeholder so the app works without the model)
```python
def predict_age(self, images, species):
    return AgePrediction(group="adult", confidence=0)  # confidence 0 = no real prediction
```

## 4. Real provider (your own class, so you do not overwrite teammates' code)
```python
from .age import get_age_model

class PawModels(DemoProvider):
    def predict_age(self, images, species):
        model = get_age_model()
        if model is None:                      # model files missing
            return super().predict_age(images, species)
        result = model.predict([img.read() for img in images], species)
        if result is None:                     # only remote seed photos, nothing readable
            return super().predict_age(images, species)
        return AgePrediction(group=result.group, confidence=result.confidence)
```
`ImageInput.read()` returns `None` for remote seed photos; `AgeModel.predict` skips them.

## 5. `backend/app/schemas.py`
```python
class AgeOut(BaseModel):
    group: str | None
    confidence: int

class AnalyzeOut(BaseModel):
    ...
    age: AgeOut | None = None
```

## 6. `backend/app/routers/ai.py` (`/api/ai/analyze`, next to the M1 breed call)
```python
age = provider.predict_age(images, species)
return AnalyzeOut(..., age=AgeOut(group=age.group, confidence=age.confidence) if age else None)
```

## 7. `.env`
```
M4_AGE_MODEL_PATH=/absolute/path/to/m4_age.onnx   # m4_age.json must be in the same folder
```

## 8. Frontend `Report.tsx` (near the age dropdown, around line 407)
```tsx
// analysis.age = { group: "young" | "adult" | "senior" | null, confidence: number } | null
useEffect(() => {
  const age = analysis?.age;
  if (age?.group && !ageTouched) setAge(age.group);   // pre-select, user can still change it
}, [analysis]);

{analysis?.age?.group ? (
  <p className="hint">
    AI suggests: {AGE_LABEL[analysis.age.group]} ({analysis.age.confidence}%)
  </p>
) : analysis?.age ? (
  <p className="hint">AI is not sure about the age. Please choose.</p>
) : null}
```
`ageTouched` should become true when the user changes the dropdown by hand, so the AI never
overwrites a choice the user made.

## 9. Numbering clash
In the current code M4 is the text tagger and M5/M6 are photo quality and background
removal. Your new list renames them. Update the `base.py` docstring and the UI labels
together, or people will mix up the numbers.
