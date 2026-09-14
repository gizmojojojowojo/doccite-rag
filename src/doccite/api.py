"""Optional FastAPI interface. Run locally with: uvicorn doccite.api:app."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from doccite.answer import GenerationError, OpenAIGenerator, ask
from doccite.models import Answer
from doccite.search import SearchIndex


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        examples=["What is the default timeout for network inactivity?"],
    )
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


def create_app(index: SearchIndex | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.index = index or SearchIndex(
            Path(os.environ.get("DOCCITE_INDEX", ".doccite/index.sqlite"))
        )
        selected = os.environ.get("DOCCITE_GENERATOR", "extractive")
        if selected not in {"extractive", "openai"}:
            raise ValueError("DOCCITE_GENERATOR must be extractive or openai")
        app.state.generator = OpenAIGenerator() if selected == "openai" else None
        app.state.lock = Lock()
        yield

    app = FastAPI(
        title="DocCite",
        version="0.1.0",
        lifespan=lifespan,
        description="Answers, exact source passages, abstentions, and retrieval traces. Try POST /ask.",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "index": app.state.index.metadata}

    @app.post("/ask", response_model=Answer)
    def answer(request: QuestionRequest):
        # Bound expensive generation concurrency for this local demonstration.
        if not app.state.lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="Another question is running; try again shortly"
            )
        try:
            return ask(
                app.state.index,
                request.question,
                k=request.k,
                mode=request.mode,
                generator=app.state.generator,
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        finally:
            app.state.lock.release()

    return app


app = create_app()
