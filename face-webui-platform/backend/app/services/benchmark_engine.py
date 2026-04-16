"""
Benchmark Engine  (Phase 6 b — production hardened)
=====================================================

Two benchmark modes:
  1. Latency-only  – warmup + measure with synthetic frames.
  2. Full eval     – IoU-based evaluation against val/face + val/non_face.

GT strategy (priority order):
  1. CelebA list_bbox_celeba.csv  — real bounding boxes, bbox scaled to resize dims.
  2. Heuristic 10%-margin box     — fallback if image NOT in CSV.
  Non-face images always have GT = [].

FPS = wall-clock total_frames / elapsed_wall_seconds   (not latency-derived).
Memory = baseline_rss vs peak_rss during measurement   (reliable, not per-frame delta).

All blocking work runs in thread pool. Results persisted in SQLite.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from app.config import get_settings
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)
settings = get_settings()

_bench_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="benchmark")


def _get_db():  # type: ignore[return]
    from app.utils.db import db
    return db


def _get_git_commit() -> Optional[str]:
    """Return the current HEAD commit SHA (first 12 chars), or None on error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parents[3]),  # repo root
        )
        sha = result.stdout.strip()
        return sha[:12] if result.returncode == 0 and sha else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Status
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


# ─────────────────────────────────────────────────────────────────────────────
#  LatencyStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyStats:
    model_name:     str
    warmup_runs:    int
    measure_runs:   int
    latencies_ms:   List[float]
    wall_clock_fps: float = 0.0

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    def percentile(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = int(len(s) * pct / 100)
        return s[min(idx, len(s) - 1)]

    @property
    def throughput_fps(self) -> float:
        return self.wall_clock_fps if self.wall_clock_fps > 0 else (1000.0 / self.mean if self.mean > 0 else 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name":   self.model_name,
            "warmup_runs":  self.warmup_runs,
            "measure_runs": self.measure_runs,
            "latency_ms": {
                "mean":   round(self.mean,   3),
                "median": round(self.median, 3),
                "stdev":  round(self.stdev,  3),
                "p50":    round(self.percentile(50), 3),
                "p90":    round(self.percentile(90), 3),
                "p95":    round(self.percentile(95), 3),
                "p99":    round(self.percentile(99), 3),
                "min":    round(min(self.latencies_ms), 3) if self.latencies_ms else 0,
                "max":    round(max(self.latencies_ms), 3) if self.latencies_ms else 0,
            },
            "throughput_fps":    round(self.throughput_fps, 2),
            "wall_clock_fps":    round(self.wall_clock_fps, 2),
            "latency_based_fps": round(1000.0 / self.mean if self.mean > 0 else 0.0, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BenchmarkResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    run_id:       str
    model_name:   str
    status:       BenchmarkStatus
    frame_width:  int
    frame_height: int

    latency_stats: Optional[LatencyStats] = None
    error_message: Optional[str]          = None
    started_at:    Optional[datetime]     = None
    completed_at:  Optional[datetime]     = None
    created_at:    datetime               = field(default_factory=datetime.utcnow)

    frames_evaluated:  int           = 0
    progress_pct:      float         = 0.0
    precision:         Optional[float] = None
    recall:            Optional[float] = None
    f1:                Optional[float] = None
    false_positives:   int           = 0
    true_positives:    int           = 0
    false_negatives:   int           = 0
    cpu_avg:           Optional[float] = None
    # Memory tracking — baseline vs peak RSS (more reliable than per-frame delta)
    memory_baseline_mb: Optional[float] = None
    memory_peak_mb:     Optional[float] = None
    memory_growth_mb:   Optional[float] = None   # peak - baseline
    # Legacy field kept for API compat (= memory_growth_mb)
    memory_delta_mb:    Optional[float] = None
    memory_avg_mb:      Optional[float] = None
    iou_threshold:      Optional[float] = None
    gt_source:          Optional[str]   = None   # "celeba_bbox" | "heuristic" | "synthetic"
    celeba_coverage:    Optional[float] = None   # fraction of face images with real GT
    is_full_eval:       bool          = False
    avg_fps:            Optional[float] = None

    # ── Research-grade evaluation fields ──────────────────────────────
    # PR curve sweep: 50 threshold points with P/R/F1/TP/FP/FN per point
    pr_curve:               Optional[List[Dict[str, Any]]] = field(default=None)
    auc_pr:                 Optional[float]      = None
    best_f1_threshold:      Optional[float]      = None
    precision_at_recall_90: Optional[float]      = None
    # Reproducibility snapshot
    model_sha256:           Optional[str]        = None
    dataset_adapter:        Optional[str]        = None   # e.g. "CelebA" or "SimpleFaceFolder"
    eval_config:            Optional[Dict[str, Any]] = field(default=None)
    # Confidence distribution histogram (adaptive-bin, range [0,1])
    # {"bins": [0.0, 0.05, ..., 1.0], "counts": [N1, N2, ...], "n_bins": N}
    confidence_histogram:   Optional[Dict[str, Any]] = field(default=None)
    # Calibration curve: confidence bin → actual precision
    calibration_curve:      Optional[List[Dict[str, Any]]] = field(default=None)
    # Experiment tagging / provenance
    run_tag:                Optional[str]        = None
    run_notes:              Optional[str]        = None

    @property
    def duration_sec(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id":            self.run_id,
            "model_name":        self.model_name,
            "status":            self.status.value,
            "frame_size":        f"{self.frame_width}x{self.frame_height}",
            "latency_stats":     self.latency_stats.as_dict() if self.latency_stats else None,
            "error_message":     self.error_message,
            "duration_sec":      self.duration_sec,
            "started_at":        self.started_at.isoformat()   if self.started_at   else None,
            "completed_at":      self.completed_at.isoformat() if self.completed_at else None,
            "created_at":        self.created_at.isoformat(),
            "frames_evaluated":  self.frames_evaluated,
            "progress_pct":      round(self.progress_pct, 1),
            "precision":         round(self.precision, 4)   if self.precision   is not None else None,
            "recall":            round(self.recall, 4)      if self.recall      is not None else None,
            "f1":                round(self.f1, 4)          if self.f1          is not None else None,
            "true_positives":    self.true_positives,
            "false_positives":   self.false_positives,
            "false_negatives":   self.false_negatives,
            "cpu_avg":           round(self.cpu_avg, 1)         if self.cpu_avg           is not None else None,
            "memory_baseline_mb": round(self.memory_baseline_mb, 1) if self.memory_baseline_mb is not None else None,
            "memory_peak_mb":    round(self.memory_peak_mb, 1)  if self.memory_peak_mb    is not None else None,
            "memory_growth_mb":  round(self.memory_growth_mb, 2) if self.memory_growth_mb is not None else None,
            "memory_delta_mb":   round(self.memory_delta_mb, 2) if self.memory_delta_mb   is not None else None,
            "memory_avg_mb":     round(self.memory_avg_mb, 1)   if self.memory_avg_mb     is not None else None,
            "avg_fps":           round(self.avg_fps, 2)         if self.avg_fps           is not None else None,
            "iou_threshold":     self.iou_threshold,
            "gt_source":         self.gt_source,
            "celeba_coverage":   round(self.celeba_coverage, 4) if self.celeba_coverage is not None else None,
            "is_full_eval":      self.is_full_eval,
            # PR curve / research-grade eval
            "pr_curve":               self.pr_curve,
            "auc_pr":                 round(self.auc_pr, 4)                 if self.auc_pr                 is not None else None,
            "best_f1_threshold":      round(self.best_f1_threshold, 4)      if self.best_f1_threshold      is not None else None,
            "precision_at_recall_90": round(self.precision_at_recall_90, 4) if self.precision_at_recall_90 is not None else None,
            # Reproducibility
            "model_sha256":           self.model_sha256,
            "dataset_adapter":        self.dataset_adapter,
            "eval_config":            self.eval_config,
            # Confidence distribution
            "confidence_histogram":   self.confidence_histogram,
            # Calibration curve
            "calibration_curve":      self.calibration_curve,
            # Experiment tagging
            "run_tag":                self.run_tag,
            "run_notes":              self.run_notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box_a: Tuple[float, float, float, float],
         box_b: Tuple[float, float, float, float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_detections_iou(
    gt_boxes:  List[Tuple[float, float, float, float]],
    det_boxes: List[Tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """PASCAL VOC greedy matching. Returns (TP, FP, FN)."""
    matched_gt: set = set()
    tp = 0
    fp = 0

    for det in det_boxes:
        best_iou = 0.0
        best_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            val = _iou(det, gt)
            if val > best_iou:
                best_iou = val
                best_idx = gi

        if best_iou >= threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
#  PR-curve helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pr_curve(
    raw_pairs: List[Tuple[List[Tuple[float, float, float, float]], List[Dict[str, Any]]]],
    iou_threshold: float,
    steps: int = 50,
) -> Dict[str, Any]:
    """
    Post-hoc precision-recall curve by sweeping confidence thresholds.

    Runs inference was already done once; this function only re-filters the
    cached (gt_boxes, raw_detections) pairs — no model forward-passes.

    Parameters
    ──────────
    raw_pairs
        List of (gt_boxes, raw_detections) tuples collected during the
        measurement loop.  raw_detections must include a ``"confidence"`` key.
    iou_threshold
        IoU threshold passed to PASCAL-VOC greedy matching.
    steps
        Number of confidence thresholds evaluated (default 50).

    Returns
    ───────
    Dict with keys:
        ``curve``                  — list of per-threshold dicts
        ``auc_pr``                 — area under PR curve (trapezoidal)
        ``best_f1_threshold``      — confidence threshold with highest F1
        ``precision_at_recall_90`` — precision at first recall ≥ 0.90 or None
    """
    # Build adaptive threshold list from unique confidence values in raw_pairs
    all_confs: List[float] = sorted(
        {
            float(d.get("confidence", 0.0))
            for _, raw_dets in raw_pairs
            for d in raw_dets
        },
        reverse=True,
    )
    # Cap to 200 unique points to keep runtime bounded
    max_pts = 200
    if len(all_confs) > max_pts:
        indices = np.linspace(0, len(all_confs) - 1, max_pts, dtype=int)
        all_confs = [all_confs[int(i)] for i in indices]
    # Prepend 1.0 (empty-recall start) and append 0.0 (full-recall end)
    # Fallback to linspace when no detections present
    if all_confs:
        thresholds: List[float] = [1.0] + all_confs + [0.0]
    else:
        thresholds = list(np.linspace(1.0, 0.0, steps))
    curve: List[Dict[str, Any]] = []

    for thr in thresholds:
        tp = fp = fn = 0
        for gt_boxes, raw_dets in raw_pairs:
            filtered: List[Tuple[float, float, float, float]] = [
                (d["x1"], d["y1"], d["x2"], d["y2"])
                for d in raw_dets
                if d.get("confidence", 0.0) >= float(thr)
            ]
            t, f_p, f_n = _match_detections_iou(gt_boxes, filtered, iou_threshold)
            tp += t
            fp += f_p
            fn += f_n
        # When threshold is so high that nothing is detected, precision is 1 by
        # convention (undefined numerator/denominator → vacuous truth).
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        curve.append({
            "threshold": round(float(thr), 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        })

    # AUC-PR via trapezoidal integration over (recall, precision)
    sorted_curve = sorted(curve, key=lambda p: p["recall"])
    recalls    = [p["recall"]    for p in sorted_curve]
    precisions = [p["precision"] for p in sorted_curve]
    auc_pr = float(np.trapz(precisions, recalls)) if len(recalls) > 1 else 0.0

    # Best F1 threshold
    best_point      = max(curve, key=lambda p: p["f1"])
    best_f1_thr     = best_point["threshold"]

    # Precision@Recall≥0.90 — take the max precision across all points ≥ 0.90 recall
    high_recall_pts = [p for p in curve if p["recall"] >= 0.90]
    p_at_r90: Optional[float] = (
        round(max(p["precision"] for p in high_recall_pts), 4)
        if high_recall_pts else None
    )

    return {
        "curve":                   curve,
        "auc_pr":                  round(max(0.0, auc_pr), 4),
        "best_f1_threshold":       best_f1_thr,
        "precision_at_recall_90":  p_at_r90,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-detection TP/FP labeling (for calibration curve)
# ─────────────────────────────────────────────────────────────────────────────

def _label_detections_iou(
    gt_boxes:      List[Tuple[float, float, float, float]],
    raw_dets:      List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Tuple[float, bool]]:
    """
    Return ``[(confidence, is_tp), ...]`` for every detection in *raw_dets*.
    Uses PASCAL VOC greedy matching (highest-IoU-first) to assign TP/FP labels.
    """
    matched_gt: set = set()
    det_boxes   = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in raw_dets]
    confidences = [float(d.get("confidence", 0.0)) for d in raw_dets]
    sorted_order = sorted(range(len(det_boxes)), key=lambda i: -confidences[i])
    labels: List[bool] = [False] * len(det_boxes)
    for di in sorted_order:
        best_iou = 0.0
        best_gi  = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            val = _iou(det_boxes[di], gt)
            if val > best_iou:
                best_iou = val
                best_gi  = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            labels[di] = True
            matched_gt.add(best_gi)
    return list(zip(confidences, labels))


def _compute_calibration_curve(
    raw_pairs:     List[Tuple[List[Tuple[float, float, float, float]], List[Dict[str, Any]]]],
    iou_threshold: float,
    n_bins:        int = 10,
) -> List[Dict[str, Any]]:
    """
    Confidence-calibration curve: split all detections into *n_bins* equal
    confidence bins and measure actual precision (TP / total) per bin.

    Returns a list of ``{"bin_start", "bin_end", "bin_center", "mean_confidence",
    "actual_precision", "count"}`` dicts (only non-empty bins included).
    """
    all_labels: List[Tuple[float, bool]] = []
    for gt_boxes, raw_dets in raw_pairs:
        all_labels.extend(_label_detections_iou(gt_boxes, raw_dets, iou_threshold))

    if not all_labels:
        return []

    bin_width = 1.0 / n_bins
    curve: List[Dict[str, Any]] = []
    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width
        in_bin = [
            (conf, tp)
            for conf, tp in all_labels
            if lo <= conf < hi or (b == n_bins - 1 and conf == 1.0)
        ]
        if not in_bin:
            continue
        n_tp      = sum(1 for _, tp in in_bin if tp)
        total     = len(in_bin)
        mean_conf = sum(c for c, _ in in_bin) / total
        curve.append({
            "bin_start":        round(lo, 3),
            "bin_end":          round(hi, 3),
            "bin_center":       round(lo + bin_width / 2, 3),
            "mean_confidence":  round(mean_conf, 4),
            "actual_precision": round(n_tp / total, 4),
            "count":            total,
        })
    return curve


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset hash helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dataset_hash(
    face_paths:    List[Path],
    nonface_paths: List[Path],
) -> str:
    """
    SHA-256 fingerprint over sorted image filenames + file sizes.
    Returns first 16 hex chars for a compact, reproducible identifier.
    """
    hasher = hashlib.sha256()
    for p in sorted(face_paths + nonface_paths, key=lambda x: x.name):
        hasher.update(p.name.encode())
        try:
            hasher.update(str(p.stat().st_size).encode())
        except OSError:
            pass
    return hasher.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
#  ComparativeResult — same-frame multi-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparativeResult:
    """Result of a same-frame comparative benchmark across two or more models."""
    run_id:        str
    model_names:   List[str]
    status:        BenchmarkStatus
    frame_width:   int
    frame_height:  int

    frames_evaluated: int           = 0
    num_frames:       int           = 0
    progress_pct:     float         = 0.0
    started_at:       Optional[datetime] = None
    completed_at:     Optional[datetime] = None
    created_at:       datetime           = field(default_factory=datetime.utcnow)
    error_message:    Optional[str]      = None
    run_tag:          Optional[str]      = None
    run_notes:        Optional[str]      = None

    # Per-model aggregate stats: {model_name: {precision, recall, f1, avg_fps, auc_pr, tp, fp, fn}}
    model_stats:          Optional[Dict[str, Dict[str, Any]]] = field(default=None)
    # Per-frame comparison: [{frame_idx, gt_count, a_tp, b_tp, f1_delta, agreement, ...}]
    per_frame_data:       Optional[List[Dict[str, Any]]]      = field(default=None)
    # Disagreement analysis summary
    disagreement_analysis: Optional[Dict[str, Any]]           = field(default=None)
    # PR curves keyed by model name
    pr_curves:            Optional[Dict[str, List[Dict[str, Any]]]] = field(default=None)
    # Histogram of per-frame confidence deltas (model_A - model_B)
    conf_shift_histogram: Optional[Dict[str, Any]]            = field(default=None)

    @property
    def duration_sec(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id":               self.run_id,
            "model_names":          self.model_names,
            "status":               self.status.value,
            "frame_size":           f"{self.frame_width}x{self.frame_height}",
            "frames_evaluated":     self.frames_evaluated,
            "num_frames":           self.num_frames,
            "progress_pct":         round(self.progress_pct, 1),
            "started_at":           self.started_at.isoformat() if self.started_at else None,
            "completed_at":         self.completed_at.isoformat() if self.completed_at else None,
            "created_at":           self.created_at.isoformat(),
            "duration_sec":         self.duration_sec,
            "error_message":        self.error_message,
            "run_tag":              self.run_tag,
            "run_notes":            self.run_notes,
            "model_stats":          self.model_stats,
            "per_frame_data":       self.per_frame_data,
            "disagreement_analysis": self.disagreement_analysis,
            "pr_curves":            self.pr_curves,
            "conf_shift_histogram": self.conf_shift_histogram,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BenchmarkEngine
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkEngine:

    def __init__(self) -> None:
        self._results: Dict[str, BenchmarkResult] = {}
        self._warmup  = settings.benchmark_warmup_runs
        self._measure = settings.benchmark_measure_runs
        # CelebA bbox cache: {image_id: (x1, y1, x2, y2)} in ORIGINAL image coords
        self._celeba_bboxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
        self._comparative_results: Dict[str, ComparativeResult] = {}
        self._load_from_db()

    # ------------------------------------------------------------------ #
    #  DB integration
    # ------------------------------------------------------------------ #

    def _load_from_db(self) -> None:
        try:
            records = _get_db().load_all_benchmarks()
            for rec in records:
                run_id = rec.get("run_id", "")
                if not run_id or run_id in self._results:
                    continue
                size_str = rec.get("frame_size", "640x480")
                fw, fh = (int(x) for x in size_str.split("x")) if "x" in size_str else (640, 480)
                br = BenchmarkResult(
                    run_id=run_id,
                    model_name=rec.get("model_name", ""),
                    status=BenchmarkStatus(rec.get("status", "completed")),
                    frame_width=fw,
                    frame_height=fh,
                    is_full_eval=bool(rec.get("is_full_eval", False)),
                    frames_evaluated=rec.get("frames_evaluated", 0),
                    precision=rec.get("precision"),
                    recall=rec.get("recall"),
                    f1=rec.get("f1"),
                    false_positives=rec.get("false_positives", 0),
                    true_positives=rec.get("true_positives", 0),
                    false_negatives=rec.get("false_negatives", 0),
                    cpu_avg=rec.get("cpu_avg"),
                    memory_avg_mb=rec.get("memory_avg_mb"),
                    memory_baseline_mb=rec.get("memory_baseline_mb"),
                    memory_peak_mb=rec.get("memory_peak_mb"),
                    memory_growth_mb=rec.get("memory_growth_mb"),
                    memory_delta_mb=rec.get("memory_delta_mb"),
                    avg_fps=rec.get("avg_fps"),
                    iou_threshold=rec.get("iou_threshold"),
                    gt_source=rec.get("gt_source"),
                    celeba_coverage=rec.get("celeba_coverage"),
                    error_message=rec.get("error_message"),
                    progress_pct=100.0 if rec.get("status") == "completed" else 0.0,
                )
                ls = rec.get("latency_stats")
                if ls:
                    lm = ls.get("latency_ms", {})
                    br.latency_stats = LatencyStats(
                        model_name=br.model_name,
                        warmup_runs=ls.get("warmup_runs", self._warmup),
                        measure_runs=ls.get("measure_runs", self._measure),
                        latencies_ms=[lm.get("mean", 0.0)] * max(1, ls.get("measure_runs", 1)),
                        wall_clock_fps=float(ls.get("wall_clock_fps") or ls.get("throughput_fps") or 0.0),
                    )
                self._results[run_id] = br
            logger.info("Loaded %d benchmark results from DB", len(records))
        except Exception as exc:
            logger.warning("Could not load benchmarks from DB: %s", exc)

    def _persist_result(self, result: BenchmarkResult) -> None:
        try:
            _get_db().save_benchmark(result.as_dict())
        except Exception as exc:
            logger.warning("Failed to persist benchmark %s: %s", result.run_id[:8], exc)

    # ------------------------------------------------------------------ #
    #  CelebA bbox loading (lazy, cached, thread-safe)
    # ------------------------------------------------------------------ #

    def _get_celeba_bboxes(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Load CelebA list_bbox_celeba.csv into memory once.
        Format: image_id, x_1, y_1, width, height  (top-left + w/h)
        Returns: {image_id: (x1, y1, x2, y2)}  in original image pixel coords.
        """
        if self._celeba_bboxes is not None:
            return self._celeba_bboxes

        csv_path = settings.data_dir / "CelebA" / "list_bbox_celeba.csv"
        if not csv_path.is_file():
            logger.warning("CelebA bbox CSV not found at %s; falling back to heuristic GT", csv_path)
            self._celeba_bboxes = {}
            return self._celeba_bboxes

        bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    x1 = int(row["x_1"])
                    y1 = int(row["y_1"])
                    x2 = x1 + int(row["width"])
                    y2 = y1 + int(row["height"])
                    bboxes[row["image_id"]] = (x1, y1, x2, y2)
            self._celeba_bboxes = bboxes
            logger.info("Loaded CelebA bboxes: %d entries from %s", len(bboxes), csv_path)
        except Exception as exc:
            logger.error("Failed to load CelebA bbox CSV: %s", exc)
            self._celeba_bboxes = {}
        return self._celeba_bboxes

    # ------------------------------------------------------------------ #
    #  Public async API
    # ------------------------------------------------------------------ #

    async def run(
        self,
        model_name:   str,
        warmup_runs:  Optional[int] = None,
        measure_runs: Optional[int] = None,
        frame_width:  int = 640,
        frame_height: int = 480,
    ) -> BenchmarkResult:
        run_id = str(uuid.uuid4())
        result = BenchmarkResult(
            run_id=run_id,
            model_name=model_name,
            status=BenchmarkStatus.PENDING,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self._results[run_id] = result
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _bench_executor,
            self._sync_benchmark,
            result,
            warmup_runs or self._warmup,
            measure_runs or self._measure,
        )
        return result

    async def compare(
        self,
        model_names:  List[str],
        warmup_runs:  Optional[int] = None,
        measure_runs: Optional[int] = None,
        frame_width:  int = 640,
        frame_height: int = 480,
    ) -> Dict[str, Any]:
        tasks = [
            self.run(n, warmup_runs, measure_runs, frame_width, frame_height)
            for n in model_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        comparison: Dict[str, Any] = {
            "models_compared": model_names,
            "frame_size": f"{frame_width}x{frame_height}",
            "results": {},
            "ranking_by_fps": [],
        }
        fps_scores: List[Tuple[str, float]] = []
        for model_name, r in zip(model_names, results):
            if isinstance(r, BenchmarkResult) and r.status == BenchmarkStatus.COMPLETED:
                comparison["results"][model_name] = r.as_dict()
                if r.latency_stats:
                    fps_scores.append((model_name, r.latency_stats.throughput_fps))
            else:
                comparison["results"][model_name] = {
                    "error": str(r) if isinstance(r, Exception) else "Unknown error"
                }
        comparison["ranking_by_fps"] = [
            {"rank": i + 1, "model": name, "fps": round(fps, 2)}
            for i, (name, fps) in enumerate(sorted(fps_scores, key=lambda x: -x[1]))
        ]
        return comparison

    async def start_full(
        self,
        model_name:   str,
        num_frames:   int = 200,
        frame_width:  int = 640,
        frame_height: int = 480,
        run_tag:      Optional[str] = None,
        run_notes:    Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        result = BenchmarkResult(
            run_id=run_id,
            model_name=model_name,
            status=BenchmarkStatus.PENDING,
            frame_width=frame_width,
            frame_height=frame_height,
            is_full_eval=True,
            run_tag=run_tag,
            run_notes=run_notes,
        )
        self._results[run_id] = result
        asyncio.create_task(self._run_full_task(result, num_frames))
        return run_id

    async def _run_full_task(self, result: BenchmarkResult, num_frames: int) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _bench_executor,
                self._sync_full_benchmark,
                result,
                num_frames,
            )
        except Exception as exc:
            logger.exception("Full benchmark task error: %s", exc)
            result.status        = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            result.completed_at  = datetime.utcnow()
            self._persist_result(result)

    # ------------------------------------------------------------------ #
    #  Full eval — IoU + CelebA real GT (thread pool)
    # ------------------------------------------------------------------ #

    def _sync_full_benchmark(self, result: BenchmarkResult, num_frames: int) -> None:
        """
        IoU-based evaluation using CelebA real GT bounding boxes.

        GT strategy (priority):
          1. CelebA list_bbox_celeba.csv — real boxes, scaled to resized frame.
          2. Heuristic 10%-margin box  — fallback when image not in CSV.
          val/non_face images always have GT = [] (all detections are FP).

        Memory tracking:
          Measures baseline RSS before the measurement loop and peak RSS during.
          Reports memory_baseline_mb, memory_peak_mb, memory_growth_mb.
          This is more reliable than per-frame delta (PyTorch pools memory).
        """
        result.status     = BenchmarkStatus.RUNNING
        result.started_at = datetime.utcnow()
        iou_thr           = settings.benchmark_iou_threshold

        try:
            model_manager.load(result.model_name)
            previous_active = model_manager.get_active_name()
            model_manager.set_active(result.model_name)

            face_paths, nonface_paths = self._collect_val_images()
            celeba = self._get_celeba_bboxes()

            if face_paths or nonface_paths:
                celeba_hits = sum(1 for p in face_paths if p.name in celeba)
                coverage    = celeba_hits / max(len(face_paths), 1)
                if celeba_hits > 0:
                    gt_source = "celeba_bbox"
                    logger.info(
                        "Full benchmark %s: %d face / %d non-face images "
                        "| CelebA coverage %d/%d (%.1f%%)",
                        result.model_name, len(face_paths), len(nonface_paths),
                        celeba_hits, len(face_paths), coverage * 100,
                    )
                else:
                    gt_source = "heuristic"
                    logger.warning(
                        "No CelebA matches for val/face — using heuristic GT. "
                        "Check that data/CelebA/list_bbox_celeba.csv exists."
                    )
            else:
                gt_source = "synthetic"
                coverage  = 0.0
                logger.warning("No val images found; using synthetic frames (GT=0)")

            result.iou_threshold  = iou_thr
            result.gt_source      = gt_source
            result.celeba_coverage = coverage if face_paths else None

            frame_schedule = self._build_frame_schedule(
                face_paths, nonface_paths, num_frames,
                result.frame_width, result.frame_height, celeba,
            )
            actual_frames = len(frame_schedule)

            # 10-frame warmup
            dummy = self._make_dummy_frame(result.frame_width, result.frame_height)
            for _ in range(self._warmup):
                model_manager.run_active(dummy)
            logger.info("Warmup complete (%d runs)", self._warmup)

            # Measurement loop
            proc = psutil.Process()
            latencies:   List[float] = []
            cpu_samples: List[float] = []

            # Reliable memory tracking: baseline vs peak RSS
            baseline_rss = proc.memory_info().rss
            peak_rss     = baseline_rss

            total_tp = 0
            total_fp = 0
            total_fn = 0

            # Collect (gt_boxes, raw_detections_with_conf) for post-hoc PR sweep
            raw_pairs: List[Tuple[List[Tuple[float, float, float, float]], List[Dict[str, Any]]]] = []

            wall_start = time.perf_counter()

            for i, (frame, gt_boxes) in enumerate(frame_schedule):
                detections, latency_ms = model_manager.run_active(frame)
                latencies.append(latency_ms)

                try:
                    current_rss = proc.memory_info().rss
                    if current_rss > peak_rss:
                        peak_rss = current_rss
                except Exception:
                    pass

                # Keep raw detections (all confidence levels) for PR sweep
                raw_pairs.append((gt_boxes, detections))

                det_boxes: List[Tuple[float, float, float, float]] = [
                    (d["x1"], d["y1"], d["x2"], d["y2"])
                    for d in detections
                ]
                tp, fp, fn = _match_detections_iou(gt_boxes, det_boxes, iou_thr)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                if i % 10 == 0:
                    try:
                        cpu_samples.append(proc.cpu_percent(interval=None))
                    except Exception:
                        pass

                result.frames_evaluated = i + 1
                result.progress_pct     = (i + 1) / actual_frames * 100.0

            wall_elapsed = time.perf_counter() - wall_start
            actual_fps   = actual_frames / wall_elapsed if wall_elapsed > 0 else 0.0

            baseline_mb = baseline_rss / 1_048_576
            peak_mb     = peak_rss / 1_048_576
            growth_mb   = peak_mb - baseline_mb

            lat_stats = LatencyStats(
                model_name=result.model_name,
                warmup_runs=self._warmup,
                measure_runs=actual_frames,
                latencies_ms=latencies,
                wall_clock_fps=actual_fps,
            )
            result.latency_stats       = lat_stats
            result.avg_fps             = actual_fps

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec)    if (prec + rec) > 0          else 0.0

            result.precision           = prec
            result.recall              = rec
            result.f1                  = f1
            result.true_positives      = total_tp
            result.false_positives     = total_fp
            result.false_negatives     = total_fn
            result.cpu_avg             = statistics.mean(cpu_samples) if cpu_samples else None
            result.memory_baseline_mb  = round(baseline_mb, 1)
            result.memory_peak_mb      = round(peak_mb, 1)
            result.memory_growth_mb    = round(growth_mb, 2)
            result.memory_delta_mb     = round(growth_mb, 2)  # legacy compat
            result.memory_avg_mb       = round((baseline_mb + peak_mb) / 2, 1)
            result.progress_pct        = 100.0
            result.status              = BenchmarkStatus.COMPLETED

            # ── Research-grade: PR curve + threshold sweep ─────────────
            try:
                pr_data = _compute_pr_curve(raw_pairs, iou_thr)
                result.pr_curve               = pr_data["curve"]
                result.auc_pr                 = pr_data["auc_pr"]
                result.best_f1_threshold      = pr_data["best_f1_threshold"]
                result.precision_at_recall_90 = pr_data["precision_at_recall_90"]
                logger.info(
                    "PR curve done | AUC-PR=%.4f | best-F1-thr=%.4f | P@R90=%s",
                    result.auc_pr, result.best_f1_threshold,
                    f"{result.precision_at_recall_90:.4f}" if result.precision_at_recall_90 is not None else "N/A",
                )
            except Exception as pr_exc:
                logger.warning("PR curve computation failed (non-fatal): %s", pr_exc)

            # ── Reproducibility snapshot ────────────────────────────────
            try:
                from app.services.model_registry import model_registry as _reg  # noqa: PLC0415
                reg_meta = _reg.get_metadata(result.model_name)
                result.model_sha256    = reg_meta.get("sha256") if reg_meta else None
                result.dataset_adapter = gt_source
            except Exception:
                pass
            result.eval_config = {
                "iou_threshold":  iou_thr,
                "sweep_steps":    len(result.pr_curve) if result.pr_curve else 0,
                "frame_size":     f"{result.frame_width}x{result.frame_height}",
                "gt_source":      gt_source,
                "num_frames":     actual_frames,
                "celeba_coverage": round(coverage, 4) if face_paths else None,
                "git_commit":     _get_git_commit(),
                "dataset_hash":   _compute_dataset_hash(face_paths, nonface_paths),
            }

            # ── Confidence distribution histogram ───────────────────────
            try:
                all_confidences = [
                    float(d.get("confidence", 0.0))
                    for _, raw_dets in raw_pairs
                    for d in raw_dets
                ]
                if all_confidences:
                    # Adaptive bin count via Freedman-Diaconis rule
                    _arr = np.array(all_confidences, dtype=float)
                    q75, q25 = np.percentile(_arr, [75, 25])
                    iqr = float(q75 - q25)
                    if iqr > 0 and len(_arr) > 1:
                        fd_width  = 2.0 * iqr / (len(_arr) ** (1.0 / 3.0))
                        n_bins_fd = int(np.ceil(1.0 / fd_width))
                        n_bins_fd = max(5, min(n_bins_fd, 50))
                    else:
                        n_bins_fd = 20
                    counts, bin_edges = np.histogram(
                        all_confidences, bins=n_bins_fd, range=(0.0, 1.0)
                    )
                    result.confidence_histogram = {
                        "bins":   [round(float(e), 3) for e in bin_edges.tolist()],
                        "counts": [int(c) for c in counts.tolist()],
                        "n_bins": n_bins_fd,
                        "total_detections": len(all_confidences),
                    }
                    logger.info(
                        "Confidence histogram: %d total detections across %d frames",
                        len(all_confidences), actual_frames,
                    )
            except Exception as hist_exc:
                logger.warning("Confidence histogram failed (non-fatal): %s", hist_exc)

            # ── Calibration curve ─────────────────────────────────────────
            try:
                result.calibration_curve = _compute_calibration_curve(raw_pairs, iou_thr)
                logger.info(
                    "Calibration curve: %d non-empty bins",
                    len(result.calibration_curve) if result.calibration_curve else 0,
                )
            except Exception as cal_exc:
                logger.warning("Calibration curve failed (non-fatal): %s", cal_exc)

            logger.info(
                "Full benchmark done | model=%s | fps=%.1f | "
                "P=%.3f R=%.3f F1=%.3f | TP=%d FP=%d FN=%d | "
                "mem baseline=%.0f peak=%.0f growth=%.1fMB | "
                "IoU=%.2f | gt=%s | celeba=%.0f%%",
                result.model_name, actual_fps,
                prec, rec, f1,
                total_tp, total_fp, total_fn,
                baseline_mb, peak_mb, growth_mb,
                iou_thr, gt_source,
                (coverage * 100) if face_paths else 0,
            )

            if previous_active and previous_active != result.model_name:
                try:
                    model_manager.set_active(previous_active)
                except Exception:
                    pass

        except Exception as exc:
            result.status        = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            logger.exception("Full benchmark failed for %s: %s", result.model_name, exc)
        finally:
            result.completed_at = datetime.utcnow()
            self._persist_result(result)

    # ------------------------------------------------------------------ #
    #  Val dataset + CelebA GT helpers
    # ------------------------------------------------------------------ #

    def _collect_val_images(self) -> Tuple[List[Path], List[Path]]:
        val_dir     = settings.data_dir / "val"
        face_dir    = val_dir / "face"
        nface_dir   = val_dir / "non_face"
        exts        = {"*.jpg", "*.jpeg", "*.png", "*.bmp"}

        def _glob(d: Path) -> List[Path]:
            found: List[Path] = []
            if d.is_dir():
                for ext in exts:
                    found.extend(d.glob(ext))
            return sorted(found)

        return _glob(face_dir), _glob(nface_dir)

    @staticmethod
    def _heuristic_gt_box(w: int, h: int) -> Tuple[float, float, float, float]:
        """10% margin fallback when CelebA annotation is unavailable."""
        mx = 0.10 * w
        my = 0.10 * h
        return (mx, my, w - mx, h - my)

    def _build_frame_schedule(
        self,
        face_paths:    List[Path],
        nonface_paths: List[Path],
        num_frames:    int,
        target_w:      int,
        target_h:      int,
        celeba_bboxes: Dict[str, Tuple[int, int, int, int]],
    ) -> List[Tuple[np.ndarray, List[Tuple[float, float, float, float]]]]:
        """
        Build (frame_array, gt_boxes) pairs for the measurement loop.

        For face images:
          * Looks up CelebA bbox by filename (exact match, e.g. "200145.jpg").
          * Scales bbox from original image coords to the target resize dims.
          * Falls back to heuristic 10%-margin box if not in CSV.
        For non-face images: gt_boxes = [].
        """
        rng = np.random.default_rng(seed=42)

        if not face_paths and not nonface_paths:
            return [
                (self._make_dummy_frame(target_w, target_h), [])
                for _ in range(num_frames)
            ]

        labelled: List[Tuple[Path, bool]] = (
            [(p, True)  for p in face_paths]
            + [(p, False) for p in nonface_paths]
        )

        idx = rng.permutation(len(labelled))
        if len(labelled) >= num_frames:
            idx = idx[:num_frames]
        else:
            repeats = (num_frames // len(labelled)) + 1
            idx = np.tile(idx, repeats)[:num_frames]

        schedule = []
        for i in idx:
            path, is_face = labelled[int(i)]
            frame, orig_w, orig_h = self._load_frame_with_dims(path, target_w, target_h)

            if is_face:
                celeba_box = celeba_bboxes.get(path.name)
                if celeba_box is not None:
                    # Scale from original image space to resized inference space
                    x_sf = target_w / orig_w if orig_w > 0 else 1.0
                    y_sf = target_h / orig_h if orig_h > 0 else 1.0
                    gt_box: Tuple[float, float, float, float] = (
                        celeba_box[0] * x_sf,
                        celeba_box[1] * y_sf,
                        celeba_box[2] * x_sf,
                        celeba_box[3] * y_sf,
                    )
                else:
                    # Fallback: heuristic 10% margin
                    gt_box = self._heuristic_gt_box(target_w, target_h)
                gt_boxes: List[Tuple[float, float, float, float]] = [gt_box]
            else:
                gt_boxes = []

            schedule.append((frame, gt_boxes))
        return schedule

    @staticmethod
    def _load_frame_with_dims(
        path: Path, width: int, height: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Load image and resize. Returns (frame_bgr, orig_width, orig_height).
        orig_w/h are needed to scale CelebA bboxes correctly.
        """
        try:
            import cv2  # type: ignore
            raw = cv2.imread(str(path))
            if raw is None:
                raise ValueError(f"cv2 could not read {path}")
            orig_h, orig_w = raw.shape[:2]
            return cv2.resize(raw, (width, height)), orig_w, orig_h
        except Exception:
            from PIL import Image  # type: ignore
            img  = Image.open(path).convert("RGB")
            orig_w, orig_h = img.size
            img  = img.resize((width, height))
            arr  = np.array(img, dtype=np.uint8)
            return arr[:, :, ::-1].copy(), orig_w, orig_h  # RGB -> BGR

    @staticmethod
    def _load_frame(path: Path, width: int, height: int) -> np.ndarray:
        """Load + resize without original dims (used by older callers)."""
        frame, _, _ = BenchmarkEngine._load_frame_with_dims(path, width, height)
        return frame

    # ------------------------------------------------------------------ #
    #  Latency-only benchmark (thread pool)
    # ------------------------------------------------------------------ #

    def _sync_benchmark(
        self,
        result:       BenchmarkResult,
        warmup_runs:  int,
        measure_runs: int,
    ) -> None:
        result.status     = BenchmarkStatus.RUNNING
        result.started_at = datetime.utcnow()

        try:
            model_manager.load(result.model_name)
            previous_active = model_manager.get_active_name()
            model_manager.set_active(result.model_name)

            dummy_frame = self._make_dummy_frame(result.frame_width, result.frame_height)

            logger.info("Benchmark %s: warming up (%d runs)...", result.model_name, warmup_runs)
            for _ in range(warmup_runs):
                model_manager.run_active(dummy_frame)

            logger.info("Benchmark %s: measuring (%d runs)...", result.model_name, measure_runs)
            latencies: List[float] = []
            wall_start = time.perf_counter()
            for _ in range(measure_runs):
                _, lat = model_manager.run_active(dummy_frame)
                latencies.append(lat)
            elapsed    = time.perf_counter() - wall_start
            actual_fps = measure_runs / elapsed if elapsed > 0 else 0.0

            result.latency_stats = LatencyStats(
                model_name=result.model_name,
                warmup_runs=warmup_runs,
                measure_runs=measure_runs,
                latencies_ms=latencies,
                wall_clock_fps=actual_fps,
            )
            result.avg_fps = actual_fps
            result.status  = BenchmarkStatus.COMPLETED

            logger.info(
                "Benchmark done | model=%s | mean=%.2fms | fps=%.1f (wall-clock)",
                result.model_name, result.latency_stats.mean, actual_fps,
            )

            if previous_active and previous_active != result.model_name:
                try:
                    model_manager.set_active(previous_active)
                except Exception:
                    pass

        except Exception as exc:
            result.status        = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            logger.exception("Benchmark failed for %s: %s", result.model_name, exc)
        finally:
            result.completed_at = datetime.utcnow()
            self._persist_result(result)

    @staticmethod
    def _make_dummy_frame(width: int, height: int) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    #  Query
    # ------------------------------------------------------------------ #

    def get_result(self, run_id: str) -> Optional[BenchmarkResult]:
        return self._results.get(run_id)

    def list_results(self, model_name: Optional[str] = None) -> List[Dict]:
        results = list(self._results.values())
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        results.sort(key=lambda r: r.created_at, reverse=True)
        return [r.as_dict() for r in results]

    # ------------------------------------------------------------------ #
    #  Comparative benchmark (public async API)
    # ------------------------------------------------------------------ #

    async def start_comparative(
        self,
        model_names:  List[str],
        num_frames:   int = 200,
        frame_width:  int = 640,
        frame_height: int = 480,
        run_tag:      Optional[str] = None,
        run_notes:    Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        result = ComparativeResult(
            run_id=run_id,
            model_names=model_names,
            status=BenchmarkStatus.PENDING,
            frame_width=frame_width,
            frame_height=frame_height,
            run_tag=run_tag,
            run_notes=run_notes,
        )
        self._comparative_results[run_id] = result
        asyncio.create_task(self._run_comparative_task(result, num_frames))
        return run_id

    async def _run_comparative_task(
        self, result: ComparativeResult, num_frames: int
    ) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _bench_executor,
                self._sync_comparative_benchmark,
                result,
                num_frames,
            )
        except Exception as exc:
            logger.exception("Comparative benchmark task error: %s", exc)
            result.status        = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            result.completed_at  = datetime.utcnow()

    def _sync_comparative_benchmark(
        self, result: ComparativeResult, num_frames: int
    ) -> None:
        """
        Run each model on the *same* frame sequence and compare per-frame detections.
        Both models operate on identical pixels — true same-frame comparison.
        """
        result.status     = BenchmarkStatus.RUNNING
        result.started_at = datetime.utcnow()
        iou_thr           = settings.benchmark_iou_threshold

        try:
            face_paths, nonface_paths = self._collect_val_images()
            celeba = self._get_celeba_bboxes()

            frame_schedule = self._build_frame_schedule(
                face_paths, nonface_paths, num_frames,
                result.frame_width, result.frame_height, celeba,
            )
            actual_frames     = len(frame_schedule)
            result.num_frames = actual_frames
            n_models          = len(result.model_names)

            # —— Run each model over all frames ——
            raw_pairs_by_model: Dict[str, List[Tuple[
                List[Tuple[float, float, float, float]],
                List[Dict[str, Any]]
            ]]] = {}
            fps_by_model: Dict[str, float] = {}

            for mi, model_name in enumerate(result.model_names):
                model_manager.load(model_name)
                model_manager.set_active(model_name)

                dummy = self._make_dummy_frame(result.frame_width, result.frame_height)
                for _ in range(self._warmup):
                    model_manager.run_active(dummy)

                pairs: List[Tuple[List, List]] = []
                wall_start = time.perf_counter()
                for fi, (frame, gt_boxes) in enumerate(frame_schedule):
                    dets, _ = model_manager.run_active(frame)
                    pairs.append((gt_boxes, dets))
                    done = mi * actual_frames + fi + 1
                    result.frames_evaluated = done
                    result.progress_pct     = done / (n_models * actual_frames) * 100.0

                fps_by_model[model_name]       = actual_frames / max(time.perf_counter() - wall_start, 1e-9)
                raw_pairs_by_model[model_name] = pairs

            # —— Per-model aggregate stats + PR curves ——
            model_stats:    Dict[str, Dict[str, Any]]            = {}
            pr_curves_out:  Dict[str, List[Dict[str, Any]]]      = {}

            for model_name, raw_pairs in raw_pairs_by_model.items():
                tp = fp = fn = 0
                for gt_boxes, dets in raw_pairs:
                    det_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets]
                    t, f_p, f_n = _match_detections_iou(gt_boxes, det_boxes, iou_thr)
                    tp += t; fp += f_p; fn += f_n  # noqa: E702
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

                try:
                    pr_data = _compute_pr_curve(raw_pairs, iou_thr)
                    pr_curves_out[model_name] = pr_data["curve"]
                    auc_pr: Optional[float]   = pr_data["auc_pr"]
                except Exception:
                    auc_pr = None

                model_stats[model_name] = {
                    "precision": round(prec, 4),
                    "recall":    round(rec, 4),
                    "f1":        round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn,
                    "avg_fps":   round(fps_by_model[model_name], 2),
                    "auc_pr":    auc_pr,
                }

            # —— Per-frame comparison (primary pair: first two models) ——
            m0 = result.model_names[0]
            m1 = result.model_names[1] if len(result.model_names) > 1 else m0

            agreement_counts: Dict[str, int] = {
                "both_detect": 0, "both_miss": 0, "a_only": 0, "b_only": 0
            }
            f1_deltas:    List[float] = []
            conf_deltas:  List[float] = []
            per_frame_data: List[Dict[str, Any]] = []

            for fi in range(actual_frames):
                _frame, gt_boxes = frame_schedule[fi]
                _, dets_0   = raw_pairs_by_model[m0][fi]
                _, dets_1   = raw_pairs_by_model[m1][fi]

                boxes_0 = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets_0]
                boxes_1 = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets_1]

                tp0, fp0, fn0 = _match_detections_iou(gt_boxes, boxes_0, iou_thr)
                tp1, fp1, fn1 = _match_detections_iou(gt_boxes, boxes_1, iou_thr)

                f1_0 = 2*tp0/(2*tp0+fp0+fn0) if (2*tp0+fp0+fn0) > 0 else 0.0
                f1_1 = 2*tp1/(2*tp1+fp1+fn1) if (2*tp1+fp1+fn1) > 0 else 0.0

                mc0  = (sum(d.get("confidence", 0.0) for d in dets_0) / len(dets_0)) if dets_0 else 0.0
                mc1  = (sum(d.get("confidence", 0.0) for d in dets_1) / len(dets_1)) if dets_1 else 0.0

                a_det = len(boxes_0) > 0
                b_det = len(boxes_1) > 0
                if a_det and b_det:
                    agr = "both_detect"
                elif not a_det and not b_det:
                    agr = "both_miss"
                elif a_det:
                    agr = "a_only"
                else:
                    agr = "b_only"

                agreement_counts[agr] += 1
                delta_f1   = f1_0 - f1_1
                delta_conf = mc0 - mc1
                f1_deltas.append(delta_f1)
                conf_deltas.append(delta_conf)

                per_frame_data.append({
                    "frame_idx":   fi,
                    "gt_count":    len(gt_boxes),
                    "a_n_dets":    len(boxes_0),
                    "b_n_dets":    len(boxes_1),
                    "a_tp": tp0, "a_fp": fp0, "a_fn": fn0,
                    "b_tp": tp1, "b_fp": fp1, "b_fn": fn1,
                    "a_f1":        round(f1_0, 4),
                    "b_f1":        round(f1_1, 4),
                    "f1_delta":    round(delta_f1, 4),
                    "a_mean_conf": round(mc0, 4),
                    "b_mean_conf": round(mc1, 4),
                    "conf_delta":  round(delta_conf, 4),
                    "agreement":   agr,
                })

            # —— Confidence-shift histogram ——
            if conf_deltas:
                c_counts, c_edges = np.histogram(conf_deltas, bins=20, range=(-1.0, 1.0))
                conf_shift_histogram: Optional[Dict[str, Any]] = {
                    "bins":    [round(float(e), 3) for e in c_edges.tolist()],
                    "counts":  [int(c) for c in c_counts.tolist()],
                    "model_a": m0,
                    "model_b": m1,
                }
            else:
                conf_shift_histogram = None

            # —— Disagreement analysis ——
            total = max(actual_frames, 1)
            disagreement_analysis: Dict[str, Any] = {
                "total_frames":      actual_frames,
                "both_detect_rate":  round(agreement_counts["both_detect"] / total, 4),
                "both_miss_rate":    round(agreement_counts["both_miss"]   / total, 4),
                "a_only_rate":       round(agreement_counts["a_only"]      / total, 4),
                "b_only_rate":       round(agreement_counts["b_only"]      / total, 4),
                "agreement_rate":    round((agreement_counts["both_detect"] + agreement_counts["both_miss"]) / total, 4),
                "disagreement_rate": round((agreement_counts["a_only"] + agreement_counts["b_only"]) / total, 4),
                "mean_f1_delta":     round(sum(f1_deltas)   / len(f1_deltas)   if f1_deltas   else 0.0, 4),
                "mean_conf_delta":   round(sum(conf_deltas) / len(conf_deltas) if conf_deltas else 0.0, 4),
                "model_a":           m0,
                "model_b":           m1,
                "dataset_hash":      _compute_dataset_hash(face_paths, nonface_paths),
                "git_commit":        _get_git_commit(),
            }

            result.model_stats           = model_stats
            result.per_frame_data        = per_frame_data
            result.disagreement_analysis = disagreement_analysis
            result.pr_curves             = pr_curves_out
            result.conf_shift_histogram  = conf_shift_histogram
            result.progress_pct          = 100.0
            result.status                = BenchmarkStatus.COMPLETED

            logger.info(
                "Comparative benchmark done | models=%s | frames=%d | "
                "disagreement=%.1f%% | mean_f1_delta=%.4f",
                result.model_names, actual_frames,
                disagreement_analysis["disagreement_rate"] * 100,
                disagreement_analysis["mean_f1_delta"],
            )

        except Exception as exc:
            result.status        = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            logger.exception("Comparative benchmark failed: %s", exc)
        finally:
            result.completed_at = datetime.utcnow()

    # ------------------------------------------------------------------ #
    #  Comparative queries
    # ------------------------------------------------------------------ #

    def get_comparative(self, run_id: str) -> Optional[ComparativeResult]:
        return self._comparative_results.get(run_id)

    def list_comparative(self) -> List[Dict[str, Any]]:
        results = sorted(
            self._comparative_results.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        return [r.as_dict() for r in results]


# Module-level singleton
benchmark_engine = BenchmarkEngine()
