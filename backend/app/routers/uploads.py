from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import get_settings

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

MAX_SIDE = 1600


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_photos(files: list[UploadFile]) -> dict[str, list[str]]:
    """Store pet photos. Images are re-encoded as JPEG (strips metadata such as GPS) and downscaled."""
    s = get_settings()
    if not files or len(files) > 6:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Upload between 1 and 6 photos.")
    s.upload_dir.mkdir(parents=True, exist_ok=True)
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
        name = f"{uuid4().hex}.jpg"
        img.save(s.upload_dir / name, "JPEG", quality=86)
        urls.append(f"/uploads/{name}")
    return {"urls": urls}
