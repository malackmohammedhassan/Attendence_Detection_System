"""
Model Manager — central registry for ML models.

Responsibilities:
  - Dynamic discovery of available models
  - Lazy loading with LRU-style memory cache
  - Thread-safe active-model switching
  - Metadata & health tracking per model
  - Designed to be adapter-extended (ScratchCNN, MTCNN, YOLO, ...)

Models are NOT imported here; this manager uses an abstract ModelAdapter
interface so concrete adapters can be registered at runtime.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Abstract adapter that concrete model wrappers must implement
# ────────────────────────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """
    Thin wrapper around any model backend.
    Concrete implementations live in ../adapters/ (created in a later phase).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def framework(self) -> str: ...

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @abstractmethod
    def predict(self, frame: Any) -> Tuple[List[Dict], float]:
        """
        Run inference on a single frame.

        Args:
            frame: numpy array (H, W, C) in BGR or RGB.

        Returns:
            (detections, latency_ms)
            detections: list of {"x1", "y1", "x2", "y2", "confidence", "class"}
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool: ...


# ────────────────────────────────────────────────────────────────────────────
#  Model registry entry
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    adapter: ModelAdapter
    loaded: bool = False
    load_time_sec: float = 0.0
    last_used: Optional[datetime] = None
    total_inferences: int = 0
    total_errors: int = 0
    registered_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def name(self) -> str:
        return self.adapter.name

    def as_info(self) -> Dict[str, Any]:
        return {
            "name": self.adapter.name,
            "framework": self.adapter.framework,
            "loaded": self.loaded,
            "load_time_sec": round(self.load_time_sec, 3),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "registered_at": self.registered_at.isoformat(),
        }


# ────────────────────────────────────────────────────────────────────────────
#  Manager
# ────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Thread-safe registry and lifecycle manager for ML model adapters.

    Example usage:
        manager = ModelManager(cache_max=3)
        manager.register(ScratchCNNAdapter(weights_path=...))
        manager.set_active("scratch_cnn")
        detections, latency = manager.run_active(frame)
    """

    def __init__(self, cache_max: int = 3) -> None:
        self._cache_max = cache_max
        self._registry: Dict[str, ModelEntry] = {}
        self._active_name: Optional[str] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #

    def register(self, adapter: ModelAdapter) -> None:
        """Register an adapter. Replaces any existing entry with the same name."""
        with self._lock:
            if adapter.name in self._registry and self._registry[adapter.name].loaded:
                logger.info("Re-registering loaded model '%s' — unloading first", adapter.name)
                self._unload_entry(self._registry[adapter.name])
            self._registry[adapter.name] = ModelEntry(adapter=adapter)
            logger.info("Registered model adapter: %s (%s)", adapter.name, adapter.framework)

    def unregister(self, model_name: str) -> None:
        with self._lock:
            entry = self._registry.pop(model_name, None)
            if entry and entry.loaded:
                self._unload_entry(entry)
            if self._active_name == model_name:
                self._active_name = None
            logger.info("Unregistered model: %s", model_name)

    # ------------------------------------------------------------------ #
    #  Loading / unloading
    # ------------------------------------------------------------------ #

    def load(self, model_name: str) -> None:
        """Load a model into memory, evicting LRU if cache is full."""
        with self._lock:
            entry = self._require(model_name)
            if entry.loaded:
                logger.debug("Model '%s' already loaded", model_name)
                return

            # Enforce cache limit
            loaded = [e for e in self._registry.values() if e.loaded]
            if len(loaded) >= self._cache_max:
                lru = min(loaded, key=lambda e: e.last_used or datetime.min)
                logger.info(
                    "Cache full (%d/%d) — evicting LRU model '%s'",
                    len(loaded), self._cache_max, lru.name,
                )
                self._unload_entry(lru)

            t0 = time.perf_counter()
            try:
                entry.adapter.load()
                entry.loaded = True
                entry.load_time_sec = time.perf_counter() - t0
                entry.last_used = datetime.utcnow()
                logger.info(
                    "Model '%s' loaded in %.3fs", model_name, entry.load_time_sec
                )
            except Exception as exc:
                entry.total_errors += 1
                raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc

    def unload(self, model_name: str) -> None:
        with self._lock:
            entry = self._require(model_name)
            if not entry.loaded:
                return
            self._unload_entry(entry)
            if self._active_name == model_name:
                self._active_name = None

    def _unload_entry(self, entry: ModelEntry) -> None:
        """Unsafe — caller must hold self._lock."""
        try:
            entry.adapter.unload()
        except Exception as exc:
            logger.warning("Error unloading model '%s': %s", entry.name, exc)
        entry.loaded = False
        logger.info("Model '%s' unloaded", entry.name)

    # ------------------------------------------------------------------ #
    #  Active model
    # ------------------------------------------------------------------ #

    def set_active(self, model_name: str) -> None:
        """Set the active model, loading it if necessary."""
        with self._lock:
            self._require(model_name)
            self.load(model_name)
            self._active_name = model_name
            logger.info("Active model set to '%s'", model_name)

    def get_active_name(self) -> Optional[str]:
        with self._lock:
            return self._active_name

    def get_active_entry(self) -> Optional[ModelEntry]:
        with self._lock:
            if self._active_name is None:
                return None
            return self._registry.get(self._active_name)

    # ------------------------------------------------------------------ #
    #  Inference via active model
    # ------------------------------------------------------------------ #

    def run_active(self, frame: Any) -> Tuple[List[Dict], float]:
        """
        Run inference using the currently active model.

        Returns:
            (detections, latency_ms)

        Raises:
            RuntimeError if no active model is set or model is not loaded.
        """
        with self._lock:
            if self._active_name is None:
                raise RuntimeError("No active model set. Call set_active() first.")
            entry = self._registry[self._active_name]
            if not entry.loaded:
                raise RuntimeError(
                    f"Active model '{self._active_name}' is not loaded."
                )

        try:
            detections, latency_ms = entry.adapter.predict(frame)
            with self._lock:
                entry.last_used = datetime.utcnow()
                entry.total_inferences += 1
            return detections, latency_ms
        except Exception as exc:
            with self._lock:
                entry.total_errors += 1
            raise

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #

    def list_models(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.as_info() for e in self._registry.values()]

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._registry.get(model_name)
            return entry.as_info() if entry else None

    def available_names(self) -> List[str]:
        with self._lock:
            return list(self._registry.keys())

    def _require(self, model_name: str) -> ModelEntry:
        """Return entry or raise KeyError."""
        entry = self._registry.get(model_name)
        if entry is None:
            raise KeyError(
                f"Model '{model_name}' not registered. "
                f"Available: {list(self._registry.keys())}"
            )
        return entry


# Module-level singleton
model_manager = ModelManager()
