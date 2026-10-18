"""Measure the face-login model on LFW and choose the login threshold.

    ml/.venv/Scripts/python ml/face/calibrate.py

LFW (Labeled Faces in the Wild): 13,233 photos of 5,749 people; the official protocol is 6,000 pairs
(3,000 same person, 3,000 different people) in 10 folds. For every pair the app's own pipeline
(backend/app/ai/face_engine.py: YuNet detection -> alignment -> SFace embedding) gives a cosine
similarity. From those scores:
  - 10-fold accuracy, the standard LFW number (threshold picked on 9 folds, tested on the 10th);
  - the false accept rate (FAR: a stranger accepted) and false reject rate (FRR: the right person refused);
  - the login threshold: the lowest score where FAR <= 0.1%, i.e. at most 1 stranger in 1,000 gets in.
  - head-turn (yaw) statistics, used to decide what counts as "looking straight at the camera".
Writes backend/app/ai/models/face/face_config.json (read by the app) and ml/face/reports/face_calibration.md.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
LFW = ROOT / "ml" / "face" / "data" / "lfw_home"
REPORT = ROOT / "ml" / "face" / "reports" / "face_calibration.md"
FAR_TARGET = 0.001
SFACE_PAPER_THRESHOLD = 0.363  # cosine threshold recommended by the SFace authors / OpenCV

spec = importlib.util.spec_from_file_location("face_engine", ROOT / "backend" / "app" / "ai" / "face_engine.py")
fe = importlib.util.module_from_spec(spec)
sys.modules["face_engine"] = fe  # dataclasses need their module registered
spec.loader.exec_module(fe)
CONFIG = fe.MODEL_DIR / "face_config.json"


def read_pairs() -> tuple[list[tuple[str, str, int, int]], int]:
    """(image_a, image_b, same, fold) for every official pair."""
    lines = (LFW / "pairs.txt").read_text().split("\n")
    folds, n = map(int, lines[0].split())
    path = lambda name, i: f"{name}/{name}_{int(i):04d}.jpg"  # noqa: E731
    out, row = [], 1
    for fold in range(folds):
        for _ in range(n):
            name, a, b = lines[row].split("\t"); row += 1
            out.append((path(name, a), path(name, b), 1, fold))
        for _ in range(n):
            n1, a, n2, b = lines[row].split("\t"); row += 1
            out.append((path(n1, a), path(n2, b), 0, fold))
    return out, folds


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(scores)
    acc = [((scores >= t) == labels).mean() for t in candidates]
    return float(candidates[int(np.argmax(acc))])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    engine = fe.FaceEngine()
    pairs, folds = read_pairs()
    images = sorted({p for a, b, *_ in pairs for p in (a, b)})

    start, emb, yaw, failed = time.time(), {}, {}, []
    for i, rel in enumerate(images):
        img = cv2.imread(str(LFW / "lfw_funneled" / rel))
        try:
            r = engine.analyze(img, pick_central=True)
            emb[rel], yaw[rel] = r.embedding, r.yaw
        except fe.FaceError as exc:
            failed.append((rel, exc.code))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(images)} images ({time.time() - start:.0f}s)")
    print(f"{len(images)} images in {time.time() - start:.0f}s · no usable face in {len(failed)}")

    # A pair with an unusable photo counts as "not the same person" (score -1), as a login would.
    scores = np.array([fe.cosine(emb[a], emb[b]) if a in emb and b in emb else -1.0 for a, b, *_ in pairs])
    labels = np.array([s for _, _, s, _ in pairs])
    fold_of = np.array([f for *_, f in pairs])

    # Standard LFW protocol: threshold from 9 folds, accuracy on the held-out fold.
    fold_acc = []
    for k in range(folds):
        t = best_threshold(scores[fold_of != k], labels[fold_of != k])
        fold_acc.append(((scores[fold_of == k] >= t) == labels[fold_of == k]).mean())
    acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc))

    far, tar, thr = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    def at(t: float) -> dict:
        accept = scores >= t
        return {"threshold": round(float(t), 4),
                "far": round(float(accept[labels == 0].mean()), 5),  # strangers let in
                "frr": round(float((~accept)[labels == 1].mean()), 5),  # right person refused
                "accuracy": round(float((accept == labels).mean()), 4)}

    ok = far <= FAR_TARGET
    login_t = float(thr[ok][np.argmax(tar[ok])])  # best acceptance while FAR stays within the target
    login = at(login_t)
    paper = at(SFACE_PAPER_THRESHOLD)
    tar_at = {f"far_{x:g}": round(float(np.interp(x, far, tar)), 4) for x in (0.01, 0.001)}

    yaws = np.abs(np.array(list(yaw.values())))
    yaw_stats = {f"p{p}": round(float(np.percentile(yaws, p)), 3) for p in (50, 90, 95, 99)}
    # Sanity check of the head-turn measure: a mirrored photo must give the opposite sign.
    probe = images[0]
    flipped = engine.analyze(cv2.flip(cv2.imread(str(LFW / "lfw_funneled" / probe)), 1), pick_central=True).yaw
    mirror_ok = bool(np.sign(flipped) == -np.sign(yaw[probe]) or abs(yaw[probe]) < 0.02)

    config = {
        "detector": fe.DETECTOR, "recognizer": fe.RECOGNIZER,
        "match_threshold": login["threshold"], "far_target": FAR_TARGET,
        # Liveness: the first photo must face the camera, the second must be clearly turned the asked way.
        "frontal_max_yaw": 0.15, "turn_min_yaw": 0.25, "same_session_min_similarity": 0.25,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lfw": {"pairs": len(pairs), "images": len(images), "no_face": len(failed), "auc": round(auc, 4),
                "accuracy_10fold": round(acc_mean, 4), "accuracy_10fold_std": round(acc_std, 4),
                "at_login_threshold": login, "at_sface_paper_threshold": paper, "tar": tar_at,
                "abs_yaw_percentiles": yaw_stats, "yaw_mirror_check": mirror_ok},
    }
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nLFW 10-fold accuracy: {acc_mean:.2%} ± {acc_std:.2%} · ROC AUC {auc:.4f}")
    print(f"TAR (right person accepted) at FAR 1%: {tar_at['far_0.01']:.2%} · at FAR 0.1%: {tar_at['far_0.001']:.2%}")
    print(f"login threshold {login['threshold']}: FAR {login['far']:.2%} (strangers in) · FRR {login['frr']:.2%} (right person refused)")
    print(f"SFace paper threshold {SFACE_PAPER_THRESHOLD}: FAR {paper['far']:.2%} · FRR {paper['frr']:.2%}")
    print(f"|yaw| on LFW: {yaw_stats} · mirror check {'OK' if mirror_ok else 'FAILED'}")
    print(f"wrote {CONFIG.relative_to(ROOT)}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# Face login: calibration on LFW", "",
        f"Pipeline: YuNet face detection → 5-landmark alignment → SFace 128-d embedding → cosine similarity "
        f"(`backend/app/ai/face_engine.py`). Benchmark: LFW, official {len(pairs):,} pairs in {folds} folds "
        f"({len(images):,} images; no usable face found in {len(failed)}, counted as non-matches).", "",
        "## Results", "",
        "| Measure | Value |", "|---|---|",
        f"| LFW 10-fold accuracy | **{acc_mean:.2%}** ± {acc_std:.2%} |",
        f"| ROC AUC | {auc:.4f} |",
        f"| Right person accepted at FAR 1% | {tar_at['far_0.01']:.2%} |",
        f"| Right person accepted at FAR 0.1% | {tar_at['far_0.001']:.2%} |", "",
        "## Login threshold", "",
        "| Threshold | Strangers let in (FAR) | Right person refused (FRR) | Accuracy |", "|---|---|---|---|",
        f"| **{login['threshold']}** (chosen: FAR ≤ {FAR_TARGET:.1%}) | {login['far']:.2%} | {login['frr']:.2%} | {login['accuracy']:.2%} |",
        f"| {SFACE_PAPER_THRESHOLD} (SFace paper) | {paper['far']:.2%} | {paper['frr']:.2%} | {paper['accuracy']:.2%} |", "",
        "Login is a 1:1 check (the person types their email, the face must match that account), limited to "
        "5 failed attempts per 15 minutes, so the per-attempt FAR above is the relevant number.", "",
        "## Head turn (liveness)", "",
        f"|yaw| (nose offset from the eye midpoint ÷ eye distance) on LFW photos: {yaw_stats}. "
        f"Mirrored-photo sign check: {'passed' if mirror_ok else 'FAILED'}. The app asks for one photo facing the "
        f"camera (|yaw| ≤ {config['frontal_max_yaw']}) and one turned the requested way (|yaw| ≥ {config['turn_min_yaw']}).", "",
        "## Limits", "",
        "- LFW photos are celebrities in good conditions; webcam photos (lighting, angle, low resolution) will do worse.",
        "- The head-turn check stops a printed photo or a still image, not a video replay or a 3D mask.",
        "- Face login is a convenience, not stronger security than a password. Recommended next step: ask for the password before changing the account email (PawConnect's Account settings don't ask for it yet).", "",
    ]), encoding="utf-8")
    print(f"report -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
