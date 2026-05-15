from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_BUILD_DIR = DEFAULT_RESULTS_DIR / "processed" / "build"
DEFAULT_WORKLOAD_THRESHOLD = 50_000
SUPPORTED_ALGORITHMS = {"kmeans", "fuzzy-cmeans"}
