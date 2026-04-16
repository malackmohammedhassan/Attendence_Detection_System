"""
Application configuration using Pydantic v2 BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BASE_DIR.parent                                 # face-webui-platform/
ML_ENGINE_ROOT = REPO_ROOT.parent / "realtime-face-detection-dl"


class Settings(BaseSettings):
    """Central configuration object.  Loaded once and cached."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    #  Server
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="ML Evaluation Dashboard API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)

    # ------------------------------------------------------------------ #
    #  CORS
    # ------------------------------------------------------------------ #
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:5176",
            "http://localhost:5177",
            "http://localhost:5178",
            "http://localhost:5179",
            "http://localhost:5180",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
            "http://127.0.0.1:5176",
        ]
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["*"])
    cors_allow_headers: List[str] = Field(default=["*"])

    # ------------------------------------------------------------------ #
    #  Paths
    # ------------------------------------------------------------------ #
    ml_engine_root: Path = Field(default=ML_ENGINE_ROOT)
    models_dir: Path = Field(default=ML_ENGINE_ROOT / "models")
    data_dir: Path = Field(default=ML_ENGINE_ROOT / "data")
    results_dir: Path = Field(default=BASE_DIR / "results")
    logs_dir: Path = Field(default=BASE_DIR / "logs")
    exports_dir: Path = Field(default=BASE_DIR / "exports")

    # ------------------------------------------------------------------ #
    #  Model defaults
    # ------------------------------------------------------------------ #
    default_model: str = Field(default="scratch_cnn")
    supported_models: List[str] = Field(default=["scratch_cnn", "mtcnn", "yolo"])
    model_cache_max: int = Field(default=3)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    inference_input_width: int = Field(default=640)
    inference_input_height: int = Field(default=480)
    inference_confidence_threshold: float = Field(default=0.5)
    inference_max_batch_size: int = Field(default=32)

    # ------------------------------------------------------------------ #
    #  Performance tracking
    # ------------------------------------------------------------------ #
    perf_sample_interval_sec: float = Field(default=1.0)
    perf_rolling_window: int = Field(default=60)   # samples kept in rolling buffer

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # ------------------------------------------------------------------ #
    #  WebSocket
    # ------------------------------------------------------------------ #
    ws_heartbeat_interval_sec: float = Field(default=30.0)
    ws_max_connections: int = Field(default=50)

    # ------------------------------------------------------------------ #
    #  Benchmark
    # ------------------------------------------------------------------ #
    benchmark_warmup_runs: int = Field(default=10)
    benchmark_measure_runs: int = Field(default=50)
    benchmark_iou_threshold: float = Field(default=0.5)   # PASCAL VOC IoU threshold
    # Security: set to False to refuse activation of models whose SHA-256 doesn't
    # match the stored hash (i.e. the model file was modified after registration).
    allow_tampered_models: bool = Field(default=True)

    # ------------------------------------------------------------------ #
    #  WebSocket live stream
    # ------------------------------------------------------------------ #
    live_max_fps: float = Field(default=30.0)   # server-side per-connection rate cap

    # ------------------------------------------------------------------ #
    #  Database
    # ------------------------------------------------------------------ #
    db_path: Path = Field(default=BASE_DIR / "results" / "dashboard.db")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    def ensure_directories(self) -> None:
        """Create all required output directories if they do not exist."""
        for directory in (self.results_dir, self.logs_dir, self.exports_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings


