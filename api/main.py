from __future__ import annotations

import time
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

    # 初始化数据库表，带重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  数据库连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"❌ 数据库连接失败，已重试 {max_retries} 次")
                raise

    app.include_router(runs_router)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    return app


app = create_app()

