import os


def get_database_url() -> str:
    v = os.environ.get("DATABASE_URL")
    if not v:
        return "postgresql+psycopg://postgres:postgres@localhost:5433/starvis"
    return v


def get_cors_origins() -> list[str]:
    v = os.environ.get("CORS_ORIGINS")
    if not v:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ]
    return [s.strip() for s in v.split(",") if s.strip()]
