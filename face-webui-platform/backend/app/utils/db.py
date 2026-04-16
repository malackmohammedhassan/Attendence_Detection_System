"""
SQLite persistence layer.

Stores benchmark results and training epoch history so data
survives server restarts.

All writes are synchronous (run in thread pool from caller).
Schema is created automatically on first access.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_DB_PATH: Path = settings.db_path
# Bump this whenever you add a new migration step below.
_SCHEMA_VERSION = 6


class Database:
    """
    Thread-safe SQLite wrapper.

    Uses a per-thread connection via threading.local so each thread
    gets its own connection (SQLite is not safe to share across threads
    without WAL + care).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure schema exists (use a dedicated boot connection)
        with self._connect() as conn:
            self._migrate(conn)
        logger.info("Database ready at %s", self._path)

    # ------------------------------------------------------------------ #
    #  Connection management
    # ------------------------------------------------------------------ #

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a thread-local connection, creating it if needed."""
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=10,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    #  Versioned schema migration
    # ------------------------------------------------------------------ #
    #
    #  How it works
    #  ─────────────
    #  _SCHEMA_VERSION is the target version (defined at module level).
    #  On startup, _run_migrations() reads the current version stored in
    #  schema_version, then applies each step in order until caught up.
    #  Adding a new version: add an entry to _MIGRATION_STEPS.
    #
    # ------------------------------------------------------------------ #

    # Each value is a raw SQL block executed as a single executescript().
    # Keys are the TARGET version reached after applying the step.
    _MIGRATION_STEPS: Dict[int, str] = {
        # Version 1 — initial tables + indexes
        1: """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS benchmark_results (
                run_id           TEXT PRIMARY KEY,
                model_name       TEXT NOT NULL,
                status           TEXT NOT NULL,
                frame_size       TEXT,
                is_full_eval     INTEGER DEFAULT 0,
                frames_evaluated INTEGER DEFAULT 0,
                precision        REAL,
                recall           REAL,
                f1               REAL,
                false_positives  INTEGER DEFAULT 0,
                iou_threshold    REAL,
                gt_source        TEXT,
                avg_fps          REAL,
                cpu_avg          REAL,
                memory_avg_mb    REAL,
                memory_delta_mb  REAL,
                latency_mean_ms  REAL,
                latency_p50_ms   REAL,
                latency_p95_ms   REAL,
                latency_p99_ms   REAL,
                latency_stdev_ms REAL,
                duration_sec     REAL,
                error_message    TEXT,
                started_at       TEXT,
                completed_at     TEXT,
                created_at       TEXT,
                raw_json         TEXT
            );
            CREATE TABLE IF NOT EXISTS training_epochs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id      TEXT NOT NULL,
                model_name  TEXT NOT NULL,
                epoch       INTEGER NOT NULL,
                train_loss  REAL,
                val_loss    REAL,
                train_acc   REAL,
                val_acc     REAL,
                lr          REAL,
                duration_sec REAL,
                created_at  TEXT,
                UNIQUE(job_id, epoch)
            );
        """,

        # Version 2 — add indexes for query performance
        2: """
            CREATE INDEX IF NOT EXISTS idx_bench_model
                ON benchmark_results(model_name);
            CREATE INDEX IF NOT EXISTS idx_bench_created
                ON benchmark_results(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_bench_status
                ON benchmark_results(status);
            CREATE INDEX IF NOT EXISTS idx_epoch_job
                ON training_epochs(job_id);
            CREATE INDEX IF NOT EXISTS idx_epoch_model
                ON training_epochs(model_name);
        """,

        # Version 3 — add CelebA / detailed memory / TP/FN columns
        3: """
            ALTER TABLE benchmark_results ADD COLUMN true_positives   INTEGER DEFAULT 0;
            ALTER TABLE benchmark_results ADD COLUMN false_negatives   INTEGER DEFAULT 0;
            ALTER TABLE benchmark_results ADD COLUMN memory_baseline_mb REAL;
            ALTER TABLE benchmark_results ADD COLUMN memory_peak_mb    REAL;
            ALTER TABLE benchmark_results ADD COLUMN memory_growth_mb  REAL;
            ALTER TABLE benchmark_results ADD COLUMN celeba_coverage   REAL;
        """,

        # Version 4 — PR curve + reproducibility snapshot
        4: """
            ALTER TABLE benchmark_results ADD COLUMN pr_curve               TEXT;
            ALTER TABLE benchmark_results ADD COLUMN auc_pr                 REAL;
            ALTER TABLE benchmark_results ADD COLUMN best_f1_threshold      REAL;
            ALTER TABLE benchmark_results ADD COLUMN precision_at_recall_90 REAL;
            ALTER TABLE benchmark_results ADD COLUMN model_sha256           TEXT;
            ALTER TABLE benchmark_results ADD COLUMN dataset_adapter        TEXT;
            ALTER TABLE benchmark_results ADD COLUMN eval_config            TEXT;
        """,

        # Version 5 — confidence distribution histogram
        5: """
            ALTER TABLE benchmark_results ADD COLUMN confidence_histogram TEXT;
        """,

        # Version 6 — experiment tagging + calibration curve
        6: """
            ALTER TABLE benchmark_results ADD COLUMN run_tag           TEXT;
            ALTER TABLE benchmark_results ADD COLUMN run_notes         TEXT;
            ALTER TABLE benchmark_results ADD COLUMN calibration_curve TEXT;
        """,
    }

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Return the current schema version stored in DB (0 = fresh DB)."""
        try:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            # schema_version table does not exist yet
            return 0

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply all pending migrations — each version step is fully atomic.

        Each step is wrapped in an explicit ``BEGIN IMMEDIATE … COMMIT``
        transaction so that a mid-migration crash leaves the DB at the
        last *successfully committed* version.  On restart, only pending
        versions are replayed.

        SQLite supports transactional DDL (CREATE TABLE, CREATE INDEX,
        ALTER TABLE ADD COLUMN are all rolled back on ROLLBACK), so a
        partial v3 ALTER TABLE is safe to retry.
        """
        current = self._get_schema_version(conn)
        if current >= _SCHEMA_VERSION:
            logger.debug("Schema at v%d — no migrations needed", current)
            return

        for version in sorted(self._MIGRATION_STEPS):
            if version <= current:
                continue
            sql = self._MIGRATION_STEPS[version]
            logger.info("Applying DB migration v%d → v%d", version - 1, version)
            try:
                # Use an explicit savepoint so we can roll back just this
                # version's statements without touching prior committed work.
                conn.execute("BEGIN IMMEDIATE")
                for stmt in sql.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                # Version record goes inside the SAME transaction so it is
                # committed atomically with the DDL above.
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version(version) VALUES (?)", (version,)
                )
                conn.execute("COMMIT")
                logger.info("Migration v%d committed", version)
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass  # already not in a transaction
                # ALTER TABLE … ADD COLUMN raises if the column already exists
                # (e.g. interrupted migration replayed on restart).  Treat as
                # no-op: stamp the version and move on.
                if "duplicate column" in str(exc).lower():
                    logger.debug(
                        "Migration v%d: column already exists — marking done", version
                    )
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_version(version) VALUES (?)", (version,)
                    )
                    conn.execute("COMMIT")
                else:
                    logger.error("Migration v%d failed — rolled back: %s", version, exc)
                    raise

        logger.info("Schema is now at v%d", _SCHEMA_VERSION)

    # ------------------------------------------------------------------ #
    #  Benchmark CRUD
    # ------------------------------------------------------------------ #

    def save_benchmark(self, result_dict: Dict[str, Any]) -> None:
        """Upsert a benchmark result dict (from BenchmarkResult.as_dict())."""
        run_id: str = result_dict["run_id"]
        ls = result_dict.get("latency_stats") or {}
        lm = ls.get("latency_ms") or {}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_results (
                    run_id, model_name, status, frame_size, is_full_eval,
                    frames_evaluated, precision, recall, f1, false_positives,
                    iou_threshold, gt_source,
                    avg_fps, cpu_avg, memory_avg_mb, memory_delta_mb,
                    latency_mean_ms, latency_p50_ms, latency_p95_ms,
                    latency_p99_ms, latency_stdev_ms,
                    duration_sec, error_message,
                    started_at, completed_at, created_at, raw_json,
                    true_positives, false_negatives,
                    memory_baseline_mb, memory_peak_mb, memory_growth_mb,
                    celeba_coverage,
                    pr_curve, auc_pr, best_f1_threshold, precision_at_recall_90,
                    model_sha256, dataset_adapter, eval_config,
                    confidence_histogram,
                    run_tag, run_notes, calibration_curve
                ) VALUES (
                    :run_id, :model_name, :status, :frame_size, :is_full_eval,
                    :frames_evaluated, :precision, :recall, :f1, :false_positives,
                    :iou_threshold, :gt_source,
                    :avg_fps, :cpu_avg, :memory_avg_mb, :memory_delta_mb,
                    :latency_mean_ms, :latency_p50_ms, :latency_p95_ms,
                    :latency_p99_ms, :latency_stdev_ms,
                    :duration_sec, :error_message,
                    :started_at, :completed_at, :created_at, :raw_json,
                    :true_positives, :false_negatives,
                    :memory_baseline_mb, :memory_peak_mb, :memory_growth_mb,
                    :celeba_coverage,
                    :pr_curve, :auc_pr, :best_f1_threshold, :precision_at_recall_90,
                    :model_sha256, :dataset_adapter, :eval_config,
                    :confidence_histogram,
                    :run_tag, :run_notes, :calibration_curve
                )
                """,
                {
                    "run_id":             run_id,
                    "model_name":         result_dict.get("model_name"),
                    "status":             result_dict.get("status"),
                    "frame_size":         result_dict.get("frame_size"),
                    "is_full_eval":       int(result_dict.get("is_full_eval", False)),
                    "frames_evaluated":   result_dict.get("frames_evaluated", 0),
                    "precision":          result_dict.get("precision"),
                    "recall":             result_dict.get("recall"),
                    "f1":                 result_dict.get("f1"),
                    "false_positives":    result_dict.get("false_positives", 0),
                    "iou_threshold":      result_dict.get("iou_threshold"),
                    "gt_source":          result_dict.get("gt_source"),
                    "avg_fps":            ls.get("throughput_fps") if ls else result_dict.get("avg_fps"),
                    "cpu_avg":            result_dict.get("cpu_avg"),
                    "memory_avg_mb":      result_dict.get("memory_avg_mb"),
                    "memory_delta_mb":    result_dict.get("memory_delta_mb"),
                    "latency_mean_ms":    lm.get("mean"),
                    "latency_p50_ms":     lm.get("p50"),
                    "latency_p95_ms":     lm.get("p95"),
                    "latency_p99_ms":     lm.get("p99"),
                    "latency_stdev_ms":   lm.get("stdev"),
                    "duration_sec":       result_dict.get("duration_sec"),
                    "error_message":      result_dict.get("error_message"),
                    "started_at":         result_dict.get("started_at"),
                    "completed_at":       result_dict.get("completed_at"),
                    "created_at":         result_dict.get("created_at", datetime.utcnow().isoformat()),
                    "raw_json":           json.dumps(result_dict),
                    # V3 columns
                    "true_positives":     result_dict.get("true_positives", 0),
                    "false_negatives":    result_dict.get("false_negatives", 0),
                    "memory_baseline_mb": result_dict.get("memory_baseline_mb"),
                    "memory_peak_mb":     result_dict.get("memory_peak_mb"),
                    "memory_growth_mb":   result_dict.get("memory_growth_mb"),
                    "celeba_coverage":    result_dict.get("celeba_coverage"),
                    # V4 columns — PR curve stored as JSON, eval_config as JSON
                    "pr_curve":               json.dumps(result_dict["pr_curve"]) if result_dict.get("pr_curve") is not None else None,
                    "auc_pr":                 result_dict.get("auc_pr"),
                    "best_f1_threshold":      result_dict.get("best_f1_threshold"),
                    "precision_at_recall_90": result_dict.get("precision_at_recall_90"),
                    "model_sha256":           result_dict.get("model_sha256"),
                    "dataset_adapter":        result_dict.get("dataset_adapter"),
                    "eval_config":            json.dumps(result_dict["eval_config"]) if result_dict.get("eval_config") is not None else None,
                    # V5 column
                    "confidence_histogram":    json.dumps(result_dict["confidence_histogram"]) if result_dict.get("confidence_histogram") is not None else None,
                    # V6 columns
                    "run_tag":                 result_dict.get("run_tag"),
                    "run_notes":               result_dict.get("run_notes"),
                    "calibration_curve":        json.dumps(result_dict["calibration_curve"]) if result_dict.get("calibration_curve") is not None else None,
                },
            )
        logger.debug("Saved benchmark %s to DB", run_id[:8])

    def load_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Load all benchmark results from DB, most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT raw_json FROM benchmark_results ORDER BY created_at DESC"
            ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row["raw_json"]))
            except Exception as exc:
                logger.warning("Skipping corrupt benchmark row: %s", exc)
        return results

    def delete_benchmark(self, run_id: str) -> bool:
        with self._connect() as conn:
            c = conn.execute(
                "DELETE FROM benchmark_results WHERE run_id = ?", (run_id,)
            )
        return c.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Training epoch history
    # ------------------------------------------------------------------ #

    def save_epoch(
        self,
        job_id: str,
        model_name: str,
        epoch_data: Dict[str, Any],
    ) -> None:
        epoch = epoch_data.get("epoch", 0)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_epochs (
                    job_id, model_name, epoch,
                    train_loss, val_loss, train_acc, val_acc,
                    lr, duration_sec, created_at
                ) VALUES (
                    :job_id, :model_name, :epoch,
                    :train_loss, :val_loss, :train_acc, :val_acc,
                    :lr, :duration_sec, :created_at
                )
                """,
                {
                    "job_id":      job_id,
                    "model_name":  model_name,
                    "epoch":       epoch,
                    "train_loss":  epoch_data.get("train_loss"),
                    "val_loss":    epoch_data.get("val_loss"),
                    "train_acc":   epoch_data.get("train_acc"),
                    "val_acc":     epoch_data.get("val_acc"),
                    "lr":          epoch_data.get("lr"),
                    "duration_sec": epoch_data.get("duration_sec"),
                    "created_at":  datetime.utcnow().isoformat(),
                },
            )

    def load_training_history(
        self,
        model_name: Optional[str] = None,
        job_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM training_epochs WHERE 1=1"
        params: List[Any] = []
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        query += " ORDER BY job_id, epoch LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            bench_count = conn.execute(
                "SELECT COUNT(*) FROM benchmark_results"
            ).fetchone()[0]
            epoch_count = conn.execute(
                "SELECT COUNT(*) FROM training_epochs"
            ).fetchone()[0]
            schema_ver  = self._get_schema_version(conn)
        return {
            "db_path":          str(self._path),
            "schema_version":   schema_ver,
            "benchmark_results": bench_count,
            "training_epochs":  epoch_count,
            "size_kb":          round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
        }


# Module-level singleton — created once on import
db = Database(_DB_PATH)
