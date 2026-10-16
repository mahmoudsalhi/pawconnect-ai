from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routers import adoptions, ai, auth, me, pets, shelters, uploads

settings = get_settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PawConnect AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    """Return one readable message per field so forms can show errors inline."""
    errors: dict[str, str] = {}
    for err in exc.errors():
        field = str(err["loc"][-1]) if err["loc"] else "body"
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "string_too_short" and err.get("ctx", {}).get("min_length"):
            msg = f"Use at least {err['ctx']['min_length']} characters."
        elif err["type"] == "value_error" and field == "email":
            msg = "Enter a valid email address, like name@example.com."
        errors.setdefault(field, msg)
    first = next(iter(errors.values()), "Some fields need attention.")
    return JSONResponse(status_code=422, content={"detail": first, "errors": errors})


for r in (auth.router, me.router, pets.router, shelters.router, adoptions.router, ai.router, uploads.router):
    app.include_router(r)

app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")


@app.get("/api/health")
def health():
    return {"ok": True}
