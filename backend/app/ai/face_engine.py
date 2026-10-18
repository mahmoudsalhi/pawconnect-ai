"""Face engine for AI face login: detect a face, turn it into a 128-number embedding, measure head turn.

Two pretrained neural networks from the OpenCV model zoo:
    YuNet  (face detection + 5 landmarks)   face_detection_yunet_2023mar.onnx    ~0.2 MB
    SFace  (face recognition, 128-d)        face_recognition_sface_2021dec.onnx  ~39 MB

Only numpy and OpenCV: ml/face loads this file directly (no app imports) so the threshold is calibrated
with exactly the code the app runs. Download the models with:
    python backend/app/ai/face_engine.py --download
"""

from __future__ import annotations

import base64
import hashlib
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "face"
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}
DETECTOR, RECOGNIZER = MODELS

MAX_SIDE = 640  # images are downscaled to this before detection
MIN_FACE_PX = 70  # smaller faces are too far from the camera to trust
MIN_SCORE = 0.85  # detector confidence
MAX_IMAGE_BYTES = 3 * 1024 * 1024


class FaceError(Exception):
    """A photo that can't be used. `message` is written for the person in front of the camera."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FaceResult:
    embedding: np.ndarray  # 128 float32, L2-normalised
    yaw: float  # head turn: about 0 facing the camera; > 0 turned to the person's left, < 0 to their right
    score: float  # detector confidence
    width: int  # face width in pixels (after downscaling)


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        path = MODEL_DIR / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            print(f"ok          {name}")
            continue
        print(f"downloading {name} ...")
        data = urllib.request.urlopen(url, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"{name}: checksum mismatch, not saved")
        path.write_bytes(data)
        print(f"saved       {name} ({len(data) / 1e6:.1f} MB)")


def decode_image(data: str) -> np.ndarray:
    """A base64 JPEG/PNG (optionally a data: URL) -> BGR image."""
    if "," in data[:100]:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (ValueError, TypeError):
        raise FaceError("bad_image", "That photo couldn’t be read. Please try again.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise FaceError("bad_image", "That photo is too large.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("bad_image", "That photo couldn’t be read. Please try again.")
    return img


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceEngine:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self._detector = cv2.FaceDetectorYN.create(str(model_dir / DETECTOR), "", (320, 320), MIN_SCORE, 0.3, 50)
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_dir / RECOGNIZER), "")
        self._lock = threading.Lock()  # OpenCV DNN objects are not thread-safe

    def detect(self, img: np.ndarray) -> np.ndarray:
        """All faces found, one row each: x, y, w, h, right eye, left eye, nose, mouth corners (x, y), score."""
        h, w = img.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(img)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def embed(self, img: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned = self._recognizer.alignCrop(img, face)
            feat = self._recognizer.feature(aligned).astype(np.float32).ravel()
        return feat / (np.linalg.norm(feat) + 1e-9)

    def analyze(self, img: np.ndarray, pick_central: bool = False) -> FaceResult:
        """Exactly one usable face -> embedding and head turn. Raises FaceError otherwise.

        pick_central: for datasets with background faces, use the face nearest the centre instead of refusing.
        """
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = self.detect(img)
        if len(faces) == 0:
            raise FaceError("no_face", "No face found. Look straight at the camera in good light.")
        if pick_central:
            cx, cy = img.shape[1] / 2, img.shape[0] / 2
            face = min(faces, key=lambda f: (f[0] + f[2] / 2 - cx) ** 2 + (f[1] + f[3] / 2 - cy) ** 2)
        else:
            faces = faces[np.argsort(-faces[:, 2])]
            if len(faces) > 1 and faces[1, 2] > 0.5 * faces[0, 2]:
                raise FaceError("many_faces", "More than one face is in view. Make sure only you are in the picture.")
            face = faces[0]
        if face[2] < MIN_FACE_PX:
            raise FaceError("too_far", "Your face is too small in the picture. Move closer to the camera.")

        right_eye, left_eye, nose = face[4:6], face[6:8], face[8:10]
        eye_dist = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
        yaw = float((nose[0] - (right_eye[0] + left_eye[0]) / 2) / eye_dist)
        return FaceResult(embedding=self.embed(img, face), yaw=yaw, score=float(face[14]), width=int(face[2]))


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_models()
    else:
        print(__doc__)
