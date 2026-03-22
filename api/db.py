from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

from api.settings import get_database_url


class Base(DeclarativeBase):
    pass


# 创建引擎，配置连接池和重连机制
engine = create_engine(
    get_database_url(),
    pool_pre_ping=True,  # 每次使用前检查连接是否有效
    pool_recycle=3600,   # 每小时回收连接
    pool_size=5,         # 连接池大小
    max_overflow=10,     # 最多额外连接数
    echo=False,          # 不输出 SQL 日志
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

