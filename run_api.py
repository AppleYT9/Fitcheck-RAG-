import sys
import uvicorn

if __name__ == "__main__":
    print("[INIT] Starting FastAPI Backend on http://127.0.0.1:8000 with auto-reload...", flush=True)
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
