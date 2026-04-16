"""
Model Registry — metadata versioning for .pth model files.

Each model gets a sidecar JSON file alongside the .pth:
    models/scratch_cnn.pth  →  models/scratch_cnn_metadata.json

Metadata schema
───────────────
{
    "name":               "scratch_cnn",
    "version":            "1.0.0",
    "file_path":          "/absolute/path/to/scratch_cnn.pth",
    "file_size_mb":       12.4,
    "registered_at":      "2024-01-01T00:00:00",
    "training_date":      null,          # set after training
    "dataset":            null,
    "architecture":       null,
    "hyperparameters":    {},
    "val_accuracy":       null,
    "benchmark_results":  [],            # list of run_id references
    "description":        "",
    "tags":               []
}

Usage
─────
    from app.services.model_registry import model_registry

    meta = model_registry.get_metadata("scratch_cnn")
    model_registry.update_metadata("scratch_cnn", {"val_accuracy": 0.92})
    all_models = model_registry.list_all()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


_DEFAULT_META: Dict[str, Any] = {
    "name":             "",
    "version":          "1.0.0",
    "file_path":        "",
    "file_size_mb":     0.0,
    "sha256":           None,
    "registered_at":    "",
    "training_date":    None,
    "dataset":          None,
    "architecture":     None,
    "hyperparameters":  {},
    "val_accuracy":     None,
    "benchmark_results": [],
    "description":      "",
    "tags":             [],
}


class ModelRegistry:
    """
    Reads and writes JSON metadata sidecar files alongside model weights.
    Auto-creates stub metadata for any .pth file found in models_dir.
    """

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_stubs()
        self._startup_integrity_check()

    def _startup_integrity_check(self) -> None:
        """Log a warning for any model whose file hash has changed since registration."""
        for pth in self._dir.glob("*.pth"):
            status = self.verify_integrity(pth.stem)
            if status == "tampered":
                logger.warning(
                    "[INTEGRITY] Model '%s' file has been replaced since registration! "
                    "Re-register or retrain to update the hash.",
                    pth.stem,
                )
            elif status == "no_hash":
                logger.info(
                    "[INTEGRITY] Model '%s' has no stored hash — updating now.",
                    pth.stem,
                )
                # Refresh the hash so future checks work.
                self.update_metadata(pth.stem, {
                    "sha256": self._file_sha256(pth),
                })

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _meta_path(self, name: str) -> Path:
        return self._dir / f"{name}_metadata.json"

    def _pth_path(self, name: str) -> Optional[Path]:
        p = self._dir / f"{name}.pth"
        return p if p.is_file() else None

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Return the hex sha256 digest of *path* (reads in 64 KB chunks)."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _ensure_stubs(self) -> None:
        """Auto-create stub metadata for every .pth that lacks one."""
        for pth in self._dir.glob("*.pth"):
            name = pth.stem
            if not self._meta_path(name).exists():
                self._create_stub(name, pth)

    def verify_integrity(self, name: str) -> str:
        """
        Compare the stored sha256 against the current file hash.

        Returns one of:
          ``"ok"``          — hashes match
          ``"tampered"``    — file changed since last registration
          ``"no_hash"``     — metadata has no stored hash (legacy stub)
          ``"missing_file"``— .pth file not found
        """
        pth = self._pth_path(name)
        if pth is None:
            return "missing_file"
        meta = self._read_meta(name)
        stored = (meta or {}).get("sha256")
        if not stored:
            return "no_hash"
        current = self._file_sha256(pth)
        if current == stored:
            return "ok"
        logger.warning(
            "Integrity check FAILED for '%s': stored=%s… current=%s…",
            name, stored[:12], current[:12],
        )
        return "tampered"

    def _create_stub(self, name: str, pth: Path) -> Dict[str, Any]:
        size_mb = round(pth.stat().st_size / 1_048_576, 2)
        sha256  = self._file_sha256(pth)
        meta: Dict[str, Any] = {
            **_DEFAULT_META,
            "name":          name,
            "file_path":     str(pth),
            "file_size_mb":  size_mb,
            "sha256":        sha256,
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._write_meta(name, meta)
        logger.info("Created metadata stub for model '%s' (%.1f MB, sha256=%s…)", name, size_mb, sha256[:12])
        return meta

    def _read_meta(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(name)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to read metadata for '%s': %s", name, exc)
            return None

    def _write_meta(self, name: str, meta: Dict[str, Any]) -> None:
        path = self._meta_path(name)
        try:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)
            os.replace(tmp, path)   # atomic rename
        except Exception as exc:
            logger.error("Failed to write metadata for '%s': %s", name, exc)
            raise

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return metadata dict for the given model name, or None if not found.
        Auto-creates stub if the .pth exists but metadata file is missing.
        """
        meta = self._read_meta(name)
        if meta is None:
            pth = self._pth_path(name)
            if pth:
                meta = self._create_stub(name, pth)
        return meta

    def update_metadata(self, name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge `updates` into existing metadata and persist.
        Creates a stub first if needed.
        """
        meta = self.get_metadata(name) or {**_DEFAULT_META, "name": name}
        meta.update(updates)
        self._write_meta(name, meta)
        return meta

    def set_training_result(
        self,
        name: str,
        *,
        training_date: Optional[str] = None,
        dataset: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        val_accuracy: Optional[float] = None,
        description: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method called after a training run completes.
        Only non-None values are written.
        """
        updates: Dict[str, Any] = {}
        if training_date  is not None: updates["training_date"]    = training_date
        if dataset        is not None: updates["dataset"]          = dataset
        if hyperparameters is not None: updates["hyperparameters"] = hyperparameters
        if val_accuracy   is not None: updates["val_accuracy"]     = val_accuracy
        if description    is not None: updates["description"]      = description
        if architecture   is not None: updates["architecture"]     = architecture
        # refresh file size + sha256 in case weights were re-saved
        pth = self._pth_path(name)
        if pth:
            updates["file_size_mb"] = round(pth.stat().st_size / 1_048_576, 2)
            updates["sha256"]       = self._file_sha256(pth)
        return self.update_metadata(name, updates)

    def add_benchmark_ref(self, name: str, run_id: str) -> None:
        """Append a benchmark run_id to the model's benchmark_results list."""
        meta = self.get_metadata(name)
        if meta is None:
            return
        refs: List[str] = meta.get("benchmark_results", [])
        if run_id not in refs:
            refs.append(run_id)
        meta["benchmark_results"] = refs[-20:]   # keep last 20 refs
        self._write_meta(name, meta)

    def list_all(self) -> List[Dict[str, Any]]:
        """Return metadata for every known model, annotated with live integrity status."""
        names: set = set()
        for p in self._dir.glob("*.pth"):
            names.add(p.stem)
        for p in self._dir.glob("*_metadata.json"):
            name = p.stem.removesuffix("_metadata")
            names.add(name)

        result = []
        for name in sorted(names):
            meta = self.get_metadata(name)
            if meta:
                meta["integrity"] = self.verify_integrity(name)
                result.append(meta)
        return result

    def delete_metadata(self, name: str) -> bool:
        """Delete metadata JSON (does not delete the .pth file)."""
        path = self._meta_path(name)
        if path.is_file():
            path.unlink()
            return True
        return False


# Module-level singleton
model_registry = ModelRegistry(settings.models_dir)
