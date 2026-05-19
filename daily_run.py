from datetime import datetime
import traceback

from data_collection.collect_data import main as collect_data
from run_daily_pipeline import main as run_daily_pipeline

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main() -> None:
    print(f"[{ts()}] Starting full daily job (collect + pipeline)...", flush=True)

    try:
        print(f"[{ts()}] Running data collection...", flush=True)
        collect_data()
        print(f"[{ts()}] Data collection complete.", flush=True)

        print(f"[{ts()}] Running daily pipeline...", flush=True)
        run_daily_pipeline()
        print(f"[{ts()}] Daily pipeline complete.", flush=True)

        print(f"[{ts()}] Full daily job completed successfully.", flush=True)
    except Exception:
        print(f"[{ts()}] Full daily job failed.", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
