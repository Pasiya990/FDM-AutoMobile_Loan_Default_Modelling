"""FastAPI application for the vehicle loan default prediction system.

Run from the project root:
    uvicorn backend.app:app --reload
then open http://127.0.0.1:8000/docs for the interactive API documentation.

The final model is loaded once at start-up. If it cannot be loaded, the API
still starts, and every endpoint answers 503 with the reason, so the frontend
can show a clear message instead of a blank screen.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.model_store import ModelNotAvailable, load_model_store
from backend.routes.prediction import router
from src.models.train_final import FINAL_DIR

logger = logging.getLogger(__name__)


def create_app(model_dir=FINAL_DIR) -> FastAPI:
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
    return app


app = create_app()
