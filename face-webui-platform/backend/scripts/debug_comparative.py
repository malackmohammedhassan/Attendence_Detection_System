import sys, traceback, uuid
sys.path.insert(0, '.')
sys.path.insert(0, 'd:/PROJECTS/Collage_Projects/SC_Project/realtime-face-detection-dl/src')

from app.services.benchmark_engine import benchmark_engine, ComparativeResult

result = ComparativeResult(
    run_id=str(uuid.uuid4()),
    model_names=['scratch_cnn', 'scratch_cnn'],
    num_frames=5,
    frame_width=320,
    frame_height=240,
    run_tag='debug',
)
try:
    benchmark_engine._sync_comparative_benchmark(result, 5)
    print('Status:', result.status)
    print('Error:', result.error_message)
    if result.model_stats:
        print('model_stats keys:', list(result.model_stats.keys()))
    if result.disagreement_analysis:
        print('da keys:', list(result.disagreement_analysis.keys()))
except Exception:
    traceback.print_exc()
