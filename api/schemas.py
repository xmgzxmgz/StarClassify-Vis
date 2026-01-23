from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GnbParams(BaseModel):
    varSmoothing: float | None = Field(default=None, ge=0)


class RunCreateRequest(BaseModel):
    datasetName: str
    targetColumn: str
    featureColumns: list[str]
    testSize: float = Field(ge=0.01, le=0.99)
    randomState: int | None = None
    modelType: Literal["gaussian_nb"]
    gnbParams: GnbParams


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class RunResult(BaseModel):
    id: str
    createdAt: datetime
    request: RunCreateRequest
    metrics: Metrics
    confusionMatrix: list[list[int]]
    labels: list[str]


class RunListResponse(BaseModel):
    items: list[RunResult]
    total: int

