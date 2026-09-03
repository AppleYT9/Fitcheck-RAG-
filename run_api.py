import os
import sys

# Limit RAM consumption for PyTorch, OpenMP & Tokenizers on cloud instances (e.g. Render Free 512MB RAM)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[INIT] Starting FastAPI Backend on http://{host}:{port}...", flush=True)
    uvicorn.run("api_server:app", host=host, port=port, log_level="info", timeout_keep_alive=120)
