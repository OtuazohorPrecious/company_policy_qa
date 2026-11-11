"""Start the service with `python -m app` (port from the PORT environment variable, default 8000)."""

import uvicorn

from app.config import load_settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=load_settings().port)
