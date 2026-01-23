from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("idx_runs_created_at", "created_at"),
        Index("idx_runs_dataset_name", "dataset_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    dataset_name: Mapped[str] = mapped_column(String, nullable=False)
    target_column: Mapped[str] = mapped_column(String, nullable=False)
    feature_columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    test_size: Mapped[float] = mapped_column(Float, nullable=False)
    random_state: Mapped[int | None] = mapped_column(Integer, nullable=True)

    model_type: Mapped[str] = mapped_column(String, nullable=False)
    model_params: Mapped[dict] = mapped_column(JSONB, nullable=False)

    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confusion_matrix: Mapped[list[list[int]]] = mapped_column(JSONB, nullable=False)
    labels: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
