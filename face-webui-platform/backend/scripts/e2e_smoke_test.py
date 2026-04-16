"""Quick end-to-end smoke test for the ML Dashboard API."""
import json
import time
import urllib.request

BASE = "http://localhost:8000/api/v1"
PASS = []
FAIL = []


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def check(label, ok, detail=""):
    if ok:
        PASS.append(label)
        print(f"  \033[32mPASS\033[0m  {label}  {detail}")
    else:
        FAIL.append(label)
        print(f"  \033[31mFAIL\033[0m  {label}  {detail}")


print("\n=== 1. Health ===")
with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
    check("GET /health", r.status == 200, f"status={r.status}")

print("\n=== 2. Models ===")
_, models = get("/inference/models")
check("Models registered", len(models) >= 1, f"{len(models)} model(s)")
scratch = next((m for m in models if m["name"] == "scratch_cnn"), None)
check("scratch_cnn present", scratch is not None)
check("scratch_cnn loaded", scratch is not None and scratch["loaded"])
mtcnn = next((m for m in models if m["name"] == "mtcnn"), None)
check("mtcnn present", mtcnn is not None)

print("\n=== 3. Latency benchmark ===")
_, lb = post(
    "/benchmark/run",
    {
        "model_name": "scratch_cnn",
        "benchmark_type": "latency_only",
        "warmup_runs": 2,
        "measure_runs": 8,
        "frame_width": 320,
        "frame_height": 240,
        "run_tag": "e2e-smoke",
    },
)
run_id = lb.get("run_id")
check("Benchmark started", run_id is not None, f"run_id={run_id}")

# Poll up to 30s
for _ in range(30):
    time.sleep(1)
    _, res = get(f"/benchmark/results/{run_id}")
    if res.get("status") in ("completed", "failed"):
        break

check("Benchmark completed", res.get("status") == "completed", f"status={res.get('status')}")
fps = res.get("latency_stats", {}).get("throughput_fps") if res.get("latency_stats") else None
check("FPS > 0", fps and fps > 0, f"fps={fps:.1f}" if fps else "no fps")

print("\n=== 4. Benchmark history + export ===")
_, hist = get("/benchmark/results")
check("History non-empty", len(hist) >= 1, f"{len(hist)} run(s)")
_, exp_json = get("/export/benchmark/all")
check("Export JSON", len(exp_json) >= 1, f"{len(exp_json)} run(s)")

print("\n=== 5. HTML report ===")
url = f"{BASE}/export/benchmark/{run_id}/report.html"
req = urllib.request.Request(url)
with urllib.request.urlopen(req, timeout=10) as r:
    html = r.read().decode()
check("HTML report served", "<!DOCTYPE html>" in html.lower() or "<html" in html.lower())
check("HTML has Chart.js", "chart.js" in html.lower() or "cdn.jsdelivr" in html.lower())
check("HTML has run_id", run_id[:8] in html)

print("\n=== 6. Comparative benchmark ===")
_, cb = post(
    "/benchmark/comparative",
    {
        "model_names": ["scratch_cnn", "scratch_cnn"],
        "num_frames": 15,
        "frame_width": 320,
        "frame_height": 240,
        "run_tag": "e2e-comparative",
    },
)
cid = cb.get("run_id")
check("Comparative started", cid is not None, f"cid={cid}")

# Poll up to 60s
for _ in range(60):
    time.sleep(1)
    _, cr = get(f"/benchmark/comparative/{cid}")
    if cr.get("status") in ("completed", "failed"):
        break

check("Comparative completed", cr.get("status") == "completed", f"status={cr.get('status')}")
check("Has disagreement_analysis", bool(cr.get("disagreement_analysis")))
check("Has model_stats", bool(cr.get("model_stats")))
check("Has per_frame_data", bool(cr.get("per_frame_data")))

da = cr.get("disagreement_analysis", {})
check("dataset_hash present", bool(da.get("dataset_hash")), da.get("dataset_hash", "missing"))

print("\n=== 7. Comparative list ===")
_, clist = get("/benchmark/comparative")
check("Comparative list has run", len(clist) >= 1, f"{len(clist)} run(s)")

print(f"\n{'='*50}")
print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print(f"  Failed: {', '.join(FAIL)}")
print("="*50)
