"""
Export routes.

GET  /export/metrics/{model_name}      — Export inference metrics as JSON
GET  /export/metrics/{model_name}/csv  — Export inference metrics as CSV
GET  /export/training/{model_name}     — Export training history as JSON
GET  /export/training/{model_name}/csv — Export training history as CSV
GET  /export/benchmark/{run_id}        — Export a benchmark result as JSON
GET  /export/benchmark/all             — Export ALL benchmark results as JSON
GET  /export/benchmark/all/csv         — Export ALL benchmark results as CSV (pandas)
GET  /export/report                    — Export a full system report as JSON
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import get_settings
from app.services.benchmark_engine import benchmark_engine
from app.utils.metrics_collector import metrics_collector
from app.utils.performance_tracker import performance_tracker

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/export", tags=["export"])


# ────────────────────────────────────────────── Inference metrics ─── #

@router.get("/metrics/{model_name}", response_class=JSONResponse)
async def export_inference_metrics_json(model_name: str) -> Dict[str, Any]:
    """Export all inference records for a model as JSON."""
    records = metrics_collector.get_inference_records(model_name=model_name, limit=10000)
    stats = metrics_collector.get_inference_stats(model_name)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inference data for model '{model_name}'.",
        )
    payload = {
        "model_name": model_name,
        "exported_at": datetime.utcnow().isoformat(),
        "stats": stats.as_dict() if stats else None,
        "records": records,
    }
    return _json_download_response(payload, filename=f"inference_{model_name}.json")


@router.get("/metrics/{model_name}/csv")
async def export_inference_metrics_csv(model_name: str) -> StreamingResponse:
    """Export inference records for a model as a downloadable CSV."""
    records = metrics_collector.get_inference_records(model_name=model_name, limit=10000)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inference data for model '{model_name}'.",
        )
    return _records_to_csv_response(
        records,
        filename=f"inference_{model_name}.csv",
    )


# ────────────────────────────────────────────── Training history ─── #

@router.get("/training/{model_name}", response_class=JSONResponse)
async def export_training_json(model_name: str) -> Dict[str, Any]:
    """Export full training epoch history as JSON."""
    history = metrics_collector.get_training_history(model_name)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training history for model '{model_name}'.",
        )
    best = metrics_collector.get_best_epoch(model_name)
    payload = {
        "model_name": model_name,
        "exported_at": datetime.utcnow().isoformat(),
        "best_epoch": best,
        "epochs": history,
    }
    return _json_download_response(payload, filename=f"training_{model_name}.json")


@router.get("/training/{model_name}/csv")
async def export_training_csv(model_name: str) -> StreamingResponse:
    """Export training epoch history as CSV."""
    history = metrics_collector.get_training_history(model_name)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training history for model '{model_name}'.",
        )
    return _records_to_csv_response(
        history,
        filename=f"training_{model_name}.csv",
    )


# ────────────────────────────────────────────── Benchmark results ── #

@router.get("/benchmark/all", response_class=JSONResponse)
async def export_all_benchmarks_json(
    model_name: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Export all benchmark results as a structured JSON download."""
    results = benchmark_engine.list_results(model_name=model_name)
    payload = {
        "exported_at": datetime.utcnow().isoformat(),
        "total": len(results),
        "model_filter": model_name,
        "results": results,
    }
    return _json_download_response(payload, filename="benchmark_results.json")


@router.get("/benchmark/all/csv")
async def export_all_benchmarks_csv(
    model_name: Optional[str] = Query(default=None),
) -> StreamingResponse:
    """
    Export all benchmark results as a flat CSV download (pandas-based).
    Latency stats are inlined as individual columns.
    """
    results = benchmark_engine.list_results(model_name=model_name)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark results to export.",
        )

    try:
        import pandas as pd  # type: ignore

        rows = []
        for r in results:
            ls = r.get("latency_stats") or {}
            lm = ls.get("latency_ms") or {}
            rows.append({
                "run_id": r.get("run_id"),
                "model_name": r.get("model_name"),
                "status": r.get("status"),
                "frame_size": r.get("frame_size"),
                "frames_evaluated": r.get("frames_evaluated", 0),
                "is_full_eval": r.get("is_full_eval", False),
                "throughput_fps": ls.get("throughput_fps"),
                "latency_mean_ms": lm.get("mean"),
                "latency_p50_ms": lm.get("p50"),
                "latency_p95_ms": lm.get("p95"),
                "latency_p99_ms": lm.get("p99"),
                "latency_stdev_ms": lm.get("stdev"),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "f1": r.get("f1"),
                "false_positives": r.get("false_positives", 0),
                "cpu_avg_pct": r.get("cpu_avg"),
                "memory_avg_mb": r.get("memory_avg_mb"),
                "duration_sec": r.get("duration_sec"),
                "started_at": r.get("started_at"),
                "completed_at": r.get("completed_at"),
                "created_at": r.get("created_at"),
            })

        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

    except ImportError:
        # pandas not available – fall back to stdlib csv
        flat_rows = []
        for r in results:
            ls = r.get("latency_stats") or {}
            lm = ls.get("latency_ms") or {}
            flat_rows.append({
                "run_id": r.get("run_id"),
                "model_name": r.get("model_name"),
                "status": r.get("status"),
                "frame_size": r.get("frame_size"),
                "frames_evaluated": r.get("frames_evaluated", 0),
                "throughput_fps": ls.get("throughput_fps"),
                "latency_mean_ms": lm.get("mean"),
                "latency_p95_ms": lm.get("p95"),
                "latency_p99_ms": lm.get("p99"),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "f1": r.get("f1"),
                "false_positives": r.get("false_positives", 0),
                "cpu_avg_pct": r.get("cpu_avg"),
                "memory_avg_mb": r.get("memory_avg_mb"),
                "duration_sec": r.get("duration_sec"),
                "created_at": r.get("created_at"),
            })
        buf = io.StringIO()
        if flat_rows:
            writer = csv.DictWriter(buf, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="benchmark_results.csv"',
        },
    )


@router.get("/benchmark/{run_id}/report.html")
async def export_benchmark_html_report(run_id: str) -> Response:
    """
    Generate a self-contained HTML report for a specific benchmark run.
    Includes PR curve, calibration curve, confidence histogram, and
    reproducibility snapshot — suitable for one-click PDF printing.
    """
    result = benchmark_engine.get_result(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found.",
        )
    html = _generate_benchmark_report_html(result.as_dict())
    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="benchmark_report_{run_id[:8]}.html"',
        },
    )


@router.get("/benchmark/{run_id}", response_class=JSONResponse)
async def export_benchmark_json(run_id: str) -> Dict[str, Any]:
    """Download a specific benchmark result as JSON."""
    result = benchmark_engine.get_result(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found.",
        )
    payload = {
        "exported_at": datetime.utcnow().isoformat(),
        **result.as_dict(),
    }
    return _json_download_response(payload, filename=f"benchmark_{run_id[:8]}.json")


# ────────────────────────────────────────────── Full system report ─ #

@router.get("/report", response_class=JSONResponse)
async def export_full_report() -> Dict[str, Any]:
    """
    Export a comprehensive snapshot of the entire system state.
    Includes performance history, all model inference stats, training records,
    and benchmark results.
    """
    snap = performance_tracker.snapshot()
    report = {
        "report_generated_at": datetime.utcnow().isoformat(),
        "system_snapshot": snap.as_dict() if snap else None,
        "system_history_last_60s": performance_tracker.history_as_dicts(n=60),
        "inference_stats": metrics_collector.get_all_inference_stats(),
        "collector_summary": metrics_collector.summary(),
        "benchmark_results": benchmark_engine.list_results(),
    }
    return _json_download_response(
        report,
        filename=f"ml_dashboard_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    )


# ────────────────────────────────────────────── Helpers ───────────── #

def _generate_benchmark_report_html(r: Dict[str, Any]) -> str:
    """
    Build a self-contained HTML report from a BenchmarkResult dict.
    All charts are rendered client-side via Chart.js (CDN).
    Suitable for direct browser printing to PDF.
    """
    import json as _json

    def _fmt(v: Any, suffix: str = "", fallback: str = "—") -> str:
        if v is None:
            return fallback
        if isinstance(v, float):
            return f"{v:.3f}{suffix}"
        return f"{v}{suffix}"

    def _pct(v: Any, fallback: str = "—") -> str:
        if v is None:
            return fallback
        return f"{float(v)*100:.1f}%"

    ls   = r.get("latency_stats") or {}
    lm   = ls.get("latency_ms") or {}
    cfg  = r.get("eval_config") or {}

    tag_html   = (f'<span class="tag tag-violet">{r["run_tag"]}</span>' if r.get("run_tag") else "")
    notes_html = (f'<br><em style="color:#94a3b8">{r["run_notes"]}</em>'  if r.get("run_notes") else "")

    pr_json   = _json.dumps(r.get("pr_curve")            or [])
    cal_json  = _json.dumps(r.get("calibration_curve")   or [])
    hist_json = _json.dumps(r.get("confidence_histogram") or {})

    fps = ls.get("throughput_fps") or r.get("avg_fps")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark Report — {r.get('model_name','')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}}
h1{{font-size:1.4rem;font-weight:700;color:#f1f5f9}}
.tag{{display:inline-block;padding:.15rem .6rem;border-radius:9999px;font-size:.72rem;font-weight:600;margin-left:.4rem}}
.tag-violet{{background:rgba(139,92,246,.15);color:#a78bfa}}
.meta{{font-size:.72rem;color:#64748b;margin-top:.35rem;line-height:1.6}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:.7rem;margin:1.5rem 0}}
.card{{background:#1e293b;border-radius:.45rem;padding:.85rem 1rem}}
.c-label{{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:.2rem}}
.c-val{{font-size:1.4rem;font-weight:700;font-family:'Consolas',monospace}}
.c-sub{{font-size:.6rem;color:#475569;margin-top:.1rem}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
.chart-card{{background:#1e293b;border-radius:.45rem;padding:1rem}}
.chart-title{{font-size:.78rem;font-weight:600;color:#cbd5e1;margin-bottom:.5rem}}
canvas{{max-height:220px}}
.section-title{{font-size:.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin:.1rem 0 .6rem;padding-bottom:.35rem;border-bottom:1px solid #1e293b}}
.repro{{background:#1e293b;border-radius:.45rem;padding:1rem;font-size:.72rem;display:grid;grid-template-columns:auto 1fr;gap:.3rem 1.2rem}}
.rk{{color:#64748b}}.rv{{color:#a78bfa;font-family:monospace;word-break:break-all}}
.hdr{{margin-bottom:1.5rem;border-bottom:1px solid #1e293b;padding-bottom:.9rem}}
@media print{{body{{background:white;color:#0f172a}}.card,.chart-card,.repro{{background:#f8fafc;border:1px solid #e2e8f0}}}}
</style>
</head>
<body>
<div class="hdr">
  <h1>{r.get('model_name','')} {tag_html}</h1>
  <div class="meta">
    Run&nbsp;{r.get('run_id','')[:8]} &nbsp;·&nbsp;
    {r.get('completed_at','')[:19].replace('T',' ')} &nbsp;·&nbsp;
    {r.get('frames_evaluated',0)} frames &nbsp;·&nbsp;
    IoU&nbsp;{_fmt(r.get('iou_threshold',0.5))} &nbsp;·&nbsp;
    {r.get('gt_source','')}{notes_html}
  </div>
</div>

<div class="grid">
  <div class="card"><div class="c-label">Throughput</div>
    <div class="c-val" style="color:#34d399">{_fmt(fps,' fps')}</div>
    <div class="c-sub">p50 {_fmt(lm.get('p50'),' ms')}</div></div>
  <div class="card"><div class="c-label">Precision</div>
    <div class="c-val" style="color:#a78bfa">{_pct(r.get('precision'))}</div></div>
  <div class="card"><div class="c-label">Recall</div>
    <div class="c-val" style="color:#22d3ee">{_pct(r.get('recall'))}</div></div>
  <div class="card"><div class="c-label">F1 Score</div>
    <div class="c-val" style="color:#f59e0b">{_pct(r.get('f1'))}</div></div>
  <div class="card"><div class="c-label">AUC-PR</div>
    <div class="c-val" style="color:#10b981">{_fmt(r.get('auc_pr'))}</div></div>
  <div class="card"><div class="c-label">Best-F1 Thr</div>
    <div class="c-val" style="color:#f59e0b">{_fmt(r.get('best_f1_threshold'))}</div>
    <div class="c-sub">confidence threshold</div></div>
  <div class="card"><div class="c-label">P@R≥90%</div>
    <div class="c-val" style="color:#8b5cf6">{_fmt(r.get('precision_at_recall_90'))}</div></div>
  <div class="card"><div class="c-label">Mem Peak</div>
    <div class="c-val" style="color:#f472b6">{_fmt(r.get('memory_peak_mb'),' MB')}</div>
    <div class="c-sub">+{_fmt(r.get('memory_growth_mb'),' MB')} growth</div></div>
</div>

<div class="charts-row">
  <div class="chart-card">
    <div class="chart-title">Precision–Recall Curve</div>
    <canvas id="prChart"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">Confidence Calibration</div>
    <canvas id="calChart"></canvas>
  </div>
</div>
<div class="chart-card" style="margin-bottom:1rem">
  <div class="chart-title">Detection Confidence Distribution</div>
  <canvas id="histChart" style="max-height:160px"></canvas>
</div>

<div class="section-title">Reproducibility Snapshot</div>
<div class="repro">
  <span class="rk">Model SHA256</span><span class="rv">{r.get('model_sha256') or '—'}</span>
  <span class="rk">Dataset Hash</span><span class="rv">{cfg.get('dataset_hash') or '—'}</span>
  <span class="rk">Git Commit</span><span class="rv">{cfg.get('git_commit') or '—'}</span>
  <span class="rk">GT Source</span><span class="rv">{r.get('gt_source') or '—'}</span>
  <span class="rk">CelebA Coverage</span><span class="rv">{_pct(r.get('celeba_coverage'))}</span>
  <span class="rk">Frame Size</span><span class="rv">{r.get('frame_size','')}</span>
  <span class="rk">IoU Threshold</span><span class="rv">{_fmt(r.get('iou_threshold'))}</span>
  <span class="rk">Sweep Steps</span><span class="rv">{cfg.get('sweep_steps','—')}</span>
</div>

<script>
const PR={pr_json};
const CAL={cal_json};
const HIST={hist_json};
const gc='rgba(100,116,139,0.15)',tc='#64748b';
const base=(xl,yl)=>({{responsive:true,maintainAspectRatio:true,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{grid:{{color:gc}},ticks:{{color:tc,font:{{size:10}}}},title:{{display:true,text:xl,color:tc,font:{{size:10}}}}}},
           y:{{grid:{{color:gc}},ticks:{{color:tc,font:{{size:10}}}},title:{{display:true,text:yl,color:tc,font:{{size:10}}}}}}}}
}});
if(PR&&PR.length){{
  const s=[...PR].sort((a,b)=>a.recall-b.recall);
  const bp=PR.reduce((b,p)=>p.f1>b.f1?p:b,PR[0]);
  new Chart(document.getElementById('prChart'),{{type:'line',
    data:{{datasets:[
      {{data:s.map(p=>({{x:p.recall,y:p.precision}})),borderColor:'#a78bfa',borderWidth:2,pointRadius:0,tension:0.2}},
      {{data:[{{x:bp.recall,y:bp.precision}}],borderColor:'#f59e0b',backgroundColor:'#f59e0b',pointRadius:8,pointStyle:'star'}}
    ]}},
    options:{{...base('Recall','Precision'),scales:{{x:{{...base().scales?.x,type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tc}}}},y:{{type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tc}}}}}}}}
  }});
}}
if(CAL&&CAL.length){{
  new Chart(document.getElementById('calChart'),{{type:'line',
    data:{{datasets:[
      {{data:[{{x:0,y:0}},{{x:1,y:1}}],borderColor:'rgba(100,116,139,0.4)',borderDash:[6,4],borderWidth:1.5,pointRadius:0}},
      {{data:CAL.map(p=>({{x:p.mean_confidence,y:p.actual_precision}})),borderColor:'#f97316',borderWidth:2,pointRadius:4,backgroundColor:'rgba(249,115,22,0.1)',fill:true}}
    ]}},
    options:{{...base('Mean Confidence','Actual Precision'),scales:{{x:{{type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tc}}}},y:{{type:'linear',min:0,max:1,grid:{{color:gc}},ticks:{{color:tc}}}}}}}}
  }});
}}
if(HIST&&HIST.bins&&HIST.counts){{
  new Chart(document.getElementById('histChart'),{{type:'bar',
    data:{{labels:HIST.bins.slice(0,-1).map(b=>b.toFixed(2)),datasets:[{{data:HIST.counts,backgroundColor:'#34d399',borderRadius:3}}]}},
    options:{{...base('Confidence','Count'),plugins:{{legend:{{display:false}}}}}}
  }});
}}
</script>
</body></html>"""


def _json_download_response(
    data: Any, filename: str
) -> JSONResponse:
    content = json.dumps(data, indent=2, default=str)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def _records_to_csv_response(
    records: List[Dict[str, Any]],
    filename: str,
) -> StreamingResponse:
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No records to export.",
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
