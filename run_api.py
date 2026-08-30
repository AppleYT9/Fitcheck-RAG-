import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[INIT] Starting FastAPI Backend on http://{host}:{port}...", flush=True)
    uvicorn.run("api_server:app", host=host, port=port, log_level="info")
