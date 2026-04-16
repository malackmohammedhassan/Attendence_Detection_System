"""
Evaluation abstractions — dataset-agnostic interfaces.

The abstract ``DatasetAdapter`` decouples ``BenchmarkEngine`` from any
particular dataset layout or annotation format.  To add a new dataset:

1. Subclass ``DatasetAdapter``.
2. Implement ``load_samples()`` and ``ground_truth_for()``.
3. Pass an instance into ``BenchmarkEngine`` (or its factory).

Example usage
─────────────
    from app.evaluation import CelebAAdapter

    adapter = CelebAAdapter(
        images_dir=Path("data/CelebA/img_celeba"),
        bbox_csv=Path("data/CelebA/list_bbox_celeba.csv"),
    )
    samples   = adapter.load_samples(max_samples=400)
    gt        = adapter.ground_truth_for(samples[0])
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
#  Value types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroundTruth:
    """
    Ground truth for a single image.

    bbox  — (x1, y1, x2, y2) in *pixel* coordinates of the source image,
            or ``None`` for images that contain no faces (negative samples).
    label — human-readable class label, e.g. ``"face"`` or ``"non_face"``.
    """
    label: str
    bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def has_face(self) -> bool:
        return self.bbox is not None

    def scaled(self, sx: float, sy: float) -> "GroundTruth":
        """Return a new GroundTruth with bbox coordinates scaled by (sx, sy)."""
        if self.bbox is None:
            return self
        x1, y1, x2, y2 = self.bbox
        return GroundTruth(
            label=self.label,
            bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
        )


@dataclass
class Sample:
    """
    A single evaluation sample produced by a DatasetAdapter.

    path          — absolute path to the image file.
    ground_truth  — associated ground truth (label + optional bbox).
    original_size — (width, height) of the image before any resizing;
                    used to compute scale factors for bbox coordinates.
                    May be ``None`` if the adapter does not pre-read dims.
    metadata      — arbitrary adapter-specific key/value pairs.
    """
    path:          Path
    ground_truth:  GroundTruth
    original_size: Optional[Tuple[int, int]] = None   # (w, h)
    metadata:      dict                      = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
#  Abstract adapter interface
# ──────────────────────────────────────────────────────────────────────────────

class DatasetAdapter(abc.ABC):
    """
    Abstract base class for evaluation dataset adapters.

    All adapters must implement:

    ``load_samples(max_samples)``
        Return a list of ``Sample`` objects drawn from the dataset.
        The adapter is free to apply shuffling, stratification, or
        any other selection strategy.

    ``ground_truth_for(sample)``
        Return the ``GroundTruth`` for a given sample.  This may be
        a no-op that merely returns ``sample.ground_truth``, or it
        may perform lazy loading of annotations.

    ``name``
        Human-readable identifier for the dataset (e.g. ``"CelebA"``).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable dataset identifier."""

    @abc.abstractmethod
    def load_samples(self, max_samples: Optional[int] = None) -> List[Sample]:
        """
        Load and return evaluation samples.

        Parameters
        ──────────
        max_samples
            Upper bound on the number of samples to return.
            ``None`` means return all available samples.
        """

    @abc.abstractmethod
    def ground_truth_for(self, sample: Sample) -> GroundTruth:
        """
        Return (possibly lazy-loaded) ground truth for *sample*.

        The default implementation simply returns ``sample.ground_truth``.
        Override to support lazy / on-demand annotation loading.
        """

    def validate(self, samples: List[Sample]) -> List[str]:
        """
        Validate a list of samples produced by ``load_samples()``.

        Returns a list of human-readable error strings.  An empty list means
        the dataset passed all checks.

        Default checks (run by this base implementation):
          1. Image paths exist on disk.
          2. Bounding boxes have non-negative dimensions (x2 > x1, y2 > y1).
          3. If ``original_size`` is provided, boxes lie within image bounds.

        Subclasses may override or extend this method.
        """
        errors: List[str] = []
        for i, sample in enumerate(samples):
            tag = f"sample[{i}] {sample.path.name!r}"

            # — path exists
            if not sample.path.exists():
                errors.append(f"{tag}: image path does not exist: {sample.path}")
                continue  # skip further checks if file is missing

            # — bbox validity
            bbox = sample.ground_truth.bbox
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                if x2 <= x1:
                    errors.append(f"{tag}: non-positive width (x1={x1}, x2={x2})")
                if y2 <= y1:
                    errors.append(f"{tag}: non-positive height (y1={y1}, y2={y2})")
                if x1 < 0 or y1 < 0:
                    errors.append(f"{tag}: negative bbox coordinate (x1={x1}, y1={y1})")

                # — bounds check when image size is known
                if sample.original_size is not None:
                    iw, ih = sample.original_size
                    if x2 > iw or y2 > ih:
                        errors.append(
                            f"{tag}: bbox ({x1},{y1},{x2},{y2}) extends outside "
                            f"image bounds ({iw}x{ih})"
                        )
        return errors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
