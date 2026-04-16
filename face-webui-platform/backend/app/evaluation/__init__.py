"""
app.evaluation — Dataset-agnostic evaluation abstractions.

Public exports
──────────────
  DatasetAdapter      abstract base class
  Sample              typed sample descriptor
  GroundTruth         bounding-box ground truth helper
  SimpleFaceFolderAdapter  data/val/{face,non_face}/ layout
  CelebAAdapter            CelebA bbox-CSV + image folder layout
"""

from app.evaluation.base import DatasetAdapter, GroundTruth, Sample
from app.evaluation.adapters import CelebAAdapter, SimpleFaceFolderAdapter

__all__ = [
    "DatasetAdapter",
    "GroundTruth",
    "Sample",
    "SimpleFaceFolderAdapter",
    "CelebAAdapter",
]
