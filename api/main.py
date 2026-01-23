from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import Base, engine
from api.routers.runs import router as runs_router
from api.settings import get_cors_origins


def create_app() -> FastAPI:
    app = FastAPI(
        title="星体分类可视化实验台 API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"] ,
        allow_headers=["*"] ,
    )

    Base.metadata.create_all(bind=engine)
    app.include_router(runs_router)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    return app


app = create_app()

