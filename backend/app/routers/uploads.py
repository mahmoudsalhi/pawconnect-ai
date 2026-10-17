from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import get_settings
from ..deps import DB
from ..models import Upload

router = APIRouter(prefix="/api/uploads", tags=["uploads"])
# Photos are served at /uploads/<name> (outside /api) so stored URLs stay short and stable.
files_router = APIRouter(prefix="/uploads", tags=["uploads"])

MAX_SIDE = 1600


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_photos(files: list[UploadFile], db: DB) -> dict[str, list[str]]:
    """Store pet photos. Images are re-encoded as JPEG (strips metadata such as GPS) and downscaled."""
    s = get_settings()
    if not files or len(files) > 6:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Upload between 1 and 6 photos.")
    urls = []
    for f in files:
        raw = await f.read(s.max_upload_mb * 1024 * 1024 + 1)
        if len(raw) > s.max_upload_mb * 1024 * 1024:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{f.filename} is larger than {s.max_upload_mb} MB.")
        try:
            img = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{f.filename} isn’t a supported image.")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        out = BytesIO()
        img.save(out, "JPEG", quality=86)
        name = f"{uuid4().hex}.jpg"
        db.add(Upload(name=name, content_type="image/jpeg", data=out.getvalue()))
        urls.append(f"/uploads/{name}")
    db.commit()
    return {"urls": urls}


@files_router.get("/{name}")
def get_photo(name: str, db: DB) -> Response:
    upload = db.get(Upload, name)
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found.")
    # Names are random and never reused, so browsers can cache them forever.
    return Response(upload.data, media_type=upload.content_type,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})
