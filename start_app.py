"""
Full-Stack Application Launcher: CNN vs Swin-Tiny Animal Classification.

Runs the FastAPI backend and serves the compiled React + Vite + Tailwind frontend.
Access the web application at: http://127.0.0.1:8000
"""

import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("Starting CNN vs Swin Transformer Full-Stack Web Application")
    print("=" * 65)
    print("  Backend API:  http://127.0.0.1:8000/api/health")
    print("  Frontend UI:  http://127.0.0.1:8000")
    print("  Swagger Docs: http://127.0.0.1:8000/docs")
    print("=" * 65)
    print("Opening web browser...\n")

    # Open browser after slight delay
    webbrowser.open("http://127.0.0.1:8000")

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
