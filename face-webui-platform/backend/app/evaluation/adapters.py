"""
Concrete DatasetAdapter implementations.

SimpleFaceFolderAdapter
───────────────────────
Expects the classic two-folder layout used by training pipelines:

    data/val/
        face/         ← positive samples (one face per image)
        non_face/     ← negative samples (no faces)

Ground truth for face images is a centred heuristic box unless an explicit
annotation CSV is provided via ``bbox_csv``.  Non-face images always have
``bbox=None``.

CelebAAdapter
─────────────
Uses the official ``list_bbox_celeba.csv`` annotation file alongside a
directory of images.  Supports the full 202,599-image dataset or a subset.

CSV format:  image_id, x_1, y_1, width, height   (top-left + dimensions)

    adapter = CelebAAdapter(
        images_dir=Path("data/CelebA/img_celeba"),
        bbox_csv=Path("data/CelebA/list_bbox_celeba.csv"),
        max_samples=1000,
    )
    samples = adapter.load_samples()

FutureCOCOAdapter (stub)
────────────────────────
Placeholder showing how to extend to COCO.  Not yet implemented.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.evaluation.base import DatasetAdapter, GroundTruth, Sample

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _heuristic_bbox(w: int, h: int) -> Tuple[float, float, float, float]:
    """Return a centred 60 % face box when no annotation is available."""
    pad_x = w * 0.20
    pad_y = h * 0.20
    return (pad_x, pad_y, w - pad_x, h - pad_y)


# ──────────────────────────────────────────────────────────────────────────────
#  SimpleFaceFolderAdapter
# ──────────────────────────────────────────────────────────────────────────────

class SimpleFaceFolderAdapter(DatasetAdapter):
    """
    Adapter for datasets stored in ``{root}/face/`` + ``{root}/non_face/``.

    Parameters
    ──────────
    val_dir
        Root directory containing ``face/`` and ``non_face/`` sub-folders.
    bbox_csv
        Optional path to a CSV file with columns
        ``image_id, x_1, y_1, width, height`` for the face images.
        If ``None`` a heuristic centred box is used instead.
    seed
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        val_dir: Path,
        bbox_csv: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self._val_dir  = val_dir
        self._seed     = seed
        self._bboxes:  Optional[Dict[str, Tuple[float, float, float, float]]] = None

        if bbox_csv and bbox_csv.is_file():
            self._bboxes = self._load_bbox_csv(bbox_csv)
            logger.info(
                "SimpleFaceFolderAdapter: loaded %d CelebA bboxes from %s",
                len(self._bboxes), bbox_csv.name,
            )

    @property
    def name(self) -> str:
        return "SimpleFaceFolder"

    # ------------------------------------------------------------------

    @staticmethod
    def _load_bbox_csv(
        csv_path: Path,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Parse ``image_id, x_1, y_1, width, height`` CSV into x1y1x2y2 dict."""
        result: Dict[str, Tuple[float, float, float, float]] = {}
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                x1 = float(row["x_1"])
                y1 = float(row["y_1"])
                x2 = x1 + float(row["width"])
                y2 = y1 + float(row["height"])
                result[row["image_id"]] = (x1, y1, x2, y2)
        return result

    def _gt_for_face(self, path: Path) -> GroundTruth:
        """Look up bbox in CSV or fall back to heuristic."""
        if self._bboxes and path.name in self._bboxes:
            return GroundTruth(label="face", bbox=self._bboxes[path.name])
        # No annotation — use heuristic (reads image dims lazily)
        return GroundTruth(label="face", bbox=None)  # caller handles None→heuristic

    def load_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        face_dir     = self._val_dir / "face"
        non_face_dir = self._val_dir / "non_face"

        face_paths  = sorted(p for p in face_dir.glob("*")     if _is_image(p)) \
            if face_dir.is_dir() else []
        nface_paths = sorted(p for p in non_face_dir.glob("*") if _is_image(p)) \
            if non_face_dir.is_dir() else []

        rng = random.Random(self._seed)
        if max_samples and max_samples < len(face_paths) + len(nface_paths):
            half = max_samples // 2
            face_paths  = rng.sample(face_paths,  min(half, len(face_paths)))
            nface_paths = rng.sample(nface_paths, min(max_samples - len(face_paths), len(nface_paths)))

        samples: List[Sample] = []
        for p in face_paths:
            samples.append(Sample(path=p, ground_truth=self._gt_for_face(p)))
        for p in nface_paths:
            samples.append(Sample(path=p, ground_truth=GroundTruth(label="non_face")))

        logger.info(
            "SimpleFaceFolderAdapter: %d face + %d non_face samples",
            len(face_paths), len(nface_paths),
        )
        return samples

    def ground_truth_for(self, sample: Sample) -> GroundTruth:
        return sample.ground_truth

    def validate(self, samples: List[Sample]) -> List[str]:
        """Run base checks plus adapter-specific duplicate-path detection."""
        errors = super().validate(samples)
        # No duplicate image paths
        seen: set = set()
        for sample in samples:
            if sample.path in seen:
                errors.append(f"Duplicate sample path: {sample.path}")
            seen.add(sample.path)
        return errors


# ──────────────────────────────────────────────────────────────────────────────
#  CelebAAdapter
# ──────────────────────────────────────────────────────────────────────────────

class CelebAAdapter(DatasetAdapter):
    """
    Adapter for the CelebA dataset.

    Parses ``list_bbox_celeba.csv`` for ground-truth bounding boxes and
    serially yields ``Sample`` objects with real CelebA annotations.

    Parameters
    ──────────
    images_dir
        Directory containing the CelebA JPEG images
        (e.g. ``data/CelebA/img_celeba/``).
    bbox_csv
        Path to ``list_bbox_celeba.csv``.
    split_csv
        Optional path to ``list_eval_partition.csv``.  If provided, only
        images from *partition* are loaded (0=train, 1=val, 2=test).
    partition
        Which split to use.  Ignored if ``split_csv`` is ``None``.
    seed
        Random seed for reproducible down-sampling.
    """

    def __init__(
        self,
        images_dir: Path,
        bbox_csv: Path,
        split_csv: Optional[Path] = None,
        partition: int = 1,          # 1 = val
        seed: int = 42,
    ) -> None:
        self._images_dir = images_dir
        self._bbox_csv   = bbox_csv
        self._split_csv  = split_csv
        self._partition  = partition
        self._seed       = seed
        self._bboxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None

    @property
    def name(self) -> str:
        return "CelebA"

    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> Dict[str, Tuple[float, float, float, float]]:
        if self._bboxes is not None:
            return self._bboxes
        logger.info("CelebAAdapter: loading bbox CSV from %s …", self._bbox_csv)
        result: Dict[str, Tuple[float, float, float, float]] = {}
        with open(self._bbox_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                x1 = float(row["x_1"])
                y1 = float(row["y_1"])
                x2 = x1 + float(row["width"])
                y2 = y1 + float(row["height"])
                result[row["image_id"]] = (x1, y1, x2, y2)
        self._bboxes = result
        logger.info("CelebAAdapter: loaded %d bboxes", len(result))
        return result

    def _split_ids(self) -> Optional[set]:
        if self._split_csv is None or not self._split_csv.is_file():
            return None
        ids: set = set()
        with open(self._split_csv, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) >= 2 and int(row[1]) == self._partition:
                    ids.add(row[0].strip())
        return ids

    def load_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        bboxes    = self._ensure_loaded()
        split_ids = self._split_ids()

        candidates = sorted(
            p for p in self._images_dir.glob("*") if _is_image(p)
        )
        if split_ids:
            candidates = [p for p in candidates if p.name in split_ids]

        rng = random.Random(self._seed)
        if max_samples and len(candidates) > max_samples:
            candidates = rng.sample(candidates, max_samples)

        samples: List[Sample] = []
        hit = miss = 0
        for p in candidates:
            if p.name in bboxes:
                gt = GroundTruth(label="face", bbox=bboxes[p.name])
                hit += 1
            else:
                gt = GroundTruth(label="face", bbox=None)
                miss += 1
            samples.append(Sample(path=p, ground_truth=gt))

        logger.info(
            "CelebAAdapter: %d samples (%d with bbox, %d without)",
            len(samples), hit, miss,
        )
        return samples

    def ground_truth_for(self, sample: Sample) -> GroundTruth:
        # All CelebA images should have a bbox — if missing, return as-is
        if sample.ground_truth.bbox is not None:
            return sample.ground_truth
        # Lazy fallback: look up the CSV
        bboxes = self._ensure_loaded()
        if sample.path.name in bboxes:
            return GroundTruth(label="face", bbox=bboxes[sample.path.name])
        return sample.ground_truth

    def validate(self, samples: List[Sample]) -> List[str]:
        """Run base checks plus CelebA-specific duplicate image_id detection."""
        errors = super().validate(samples)
        # No duplicate image filenames (CelebA uses unique filenames as IDs)
        seen_names: set = set()
        for sample in samples:
            if sample.path.name in seen_names:
                errors.append(f"Duplicate CelebA image_id: {sample.path.name!r}")
            seen_names.add(sample.path.name)
        # Warn about missing bboxes
        no_bbox = [s for s in samples if s.ground_truth.bbox is None]
        if no_bbox:
            errors.append(
                f"{len(no_bbox)} sample(s) have no bbox annotation "
                f"(first: {no_bbox[0].path.name!r})"
            )
        return errors


# ──────────────────────────────────────────────────────────────────────────────
#  Stub for future COCO adapter
# ──────────────────────────────────────────────────────────────────────────────

class FutureCOCOAdapter(DatasetAdapter):
    """
    Placeholder for a COCO-format adapter (person/face category).

    Raises ``NotImplementedError`` — implement when needed.
    """

    @property
    def name(self) -> str:
        return "COCO"

    def load_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        raise NotImplementedError("COCO adapter not yet implemented")

    def ground_truth_for(self, sample: Sample) -> GroundTruth:
        raise NotImplementedError("COCO adapter not yet implemented")
