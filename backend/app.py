"""FastAPI application for the vehicle loan default prediction system.

Run from the project root:
    uvicorn backend.app:app --reload
then open http://127.0.0.1:8000/ for the web page (the React build in
frontend/dist/) or http://127.0.0.1:8000/docs for the API documentation.

The final model is loaded once at start-up. If it cannot be loaded, the API
still starts, and every endpoint answers 503 with the reason, so the frontend
can show a clear message instead of a blank screen.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.model_store import ModelNotAvailable, load_model_store
from backend.routes.prediction import router
from src.models.train_final import FINAL_DIR

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

NOT_BUILT_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Frontend not built</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px">
<h1>The web page has not been built</h1>
<p>The API is running. To build the page, run these commands in the <code>frontend</code> folder,
then restart the API:</p>
<pre>npm install
npm run build</pre>
<p>The API documentation is at <a href="/docs">/docs</a>.</p>
</body></html>"""


def create_app(model_dir=FINAL_DIR, frontend_dir=FRONTEND_DIST) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.store, app.state.load_error = load_model_store(model_dir), None
        except ModelNotAvailable as error:
            logger.error("Model not loaded: %s", error)
            app.state.store, app.state.load_error = None, str(error)
        yield

    app = FastAPI(
        title="Vehicle Loan Default Prediction API",
        description="Scores one vehicle-loan application. Decision support only.",
        lifespan=lifespan,
    )
    # The frontend is a static page that may be opened from another address;
    # this academic prototype accepts requests from any origin
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(router)

    # The page is mounted after the API routes, so /health, /schema, /predict
    # and /examples are matched first and everything else is a page file
    frontend_dir = Path(frontend_dir)
    if (frontend_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def frontend_not_built():
            return NOT_BUILT_PAGE

    return app


app = create_app()
