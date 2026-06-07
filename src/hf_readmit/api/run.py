"""Entry point to serve the FastAPI app with uvicorn.

Run with ``python -m hf_readmit.api.run`` (listens on 0.0.0.0:8000).
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Start the uvicorn server for the discharge-planning API."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("hf_readmit.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
