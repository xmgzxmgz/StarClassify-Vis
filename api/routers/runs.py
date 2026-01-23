from __future__ import annotations

import uuid
from io import StringIO

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.db import get_db
from api.ml import train_gaussian_nb
from api.models import Run
from api.schemas import RunCreateRequest, RunListResponse, RunResult


router = APIRouter(prefix="/api/runs", tags=["runs"])


def _to_result(row: Run) -> RunResult:
    """将数据库记录转换为 API 响应结构。"""
    req = RunCreateRequest(
        datasetName=row.dataset_name,
        targetColumn=row.target_column,
        featureColumns=list(row.feature_columns),
        testSize=row.test_size,
        randomState=row.random_state,
        modelType="gaussian_nb",
        gnbParams={
            "varSmoothing": (row.model_params or {}).get("var_smoothing"),
        },
    )
    return RunResult(
        id=str(row.id),
        createdAt=row.created_at,
        request=req,
        metrics=row.metrics,
        confusionMatrix=row.confusion_matrix,
        labels=row.labels,
    )


def _infer_target_column(df: pd.DataFrame) -> str:
    """根据列名推断目标列。"""
    hints = ["class", "label", "target", "type", "类别", "分类", "星类", "star_type"]
    cols = list(df.columns)
    lower = [str(c).lower() for c in cols]
    for hint in hints:
        for i, c in enumerate(lower):
            if c == hint:
                return cols[i]
    for hint in hints:
        for i, c in enumerate(lower):
            if hint in c:
                return cols[i]
    return cols[-1] if cols else ""


def _infer_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """根据数据类型推断特征列。"""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    features = [c for c in numeric_cols if c != target]
    if features:
        return features
    return [c for c in df.columns if c != target]


@router.post("", response_model=RunResult)
def create_run(
    file: UploadFile = File(...),
    payload: str = Form(...),
    db: Session = Depends(get_db),
):
    """上传 CSV 执行训练并保存结果。"""
    try:
        req = RunCreateRequest.model_validate_json(payload)
    except Exception:
        raise HTTPException(status_code=400, detail="payload 不是合法的 JSON 或字段不符合要求")

    if req.modelType != "gaussian_nb":
        raise HTTPException(status_code=400, detail="仅支持 gaussian_nb")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件")

    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="文件编码不支持，请使用 UTF-8")

    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        raise HTTPException(status_code=400, detail="CSV 解析失败，请检查分隔符与格式")

    dataset_name = (req.datasetName or "").strip() or file.filename
    target_column = (req.targetColumn or "").strip()
    if not target_column:
        target_column = _infer_target_column(df)
    if not target_column or target_column not in df.columns:
        raise HTTPException(status_code=400, detail="未能识别目标列，请在高级设置中手动选择")

    feature_columns = [c for c in req.featureColumns or [] if c != target_column]
    if not feature_columns:
        feature_columns = _infer_feature_columns(df, target_column)
    if not feature_columns:
        raise HTTPException(status_code=400, detail="未能识别特征列，请在高级设置中手动选择")

    try:
        out = train_gaussian_nb(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=req.testSize,
            random_state=req.randomState,
            var_smoothing=req.gnbParams.varSmoothing,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="训练/预测失败")

    row = Run(
        id=uuid.uuid4(),
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=req.testSize,
        random_state=req.randomState,
        model_type=req.modelType,
        model_params={
            "var_smoothing": req.gnbParams.varSmoothing,
        },
        metrics=out.metrics,
        confusion_matrix=out.confusion_matrix,
        labels=out.labels,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_result(row)


@router.get("", response_model=RunListResponse)
def list_runs(
    query: str | None = None,
    page: int = 1,
    pageSize: int = 20,
    db: Session = Depends(get_db),
):
    """分页查询历史结果记录。"""
    if page < 1 or pageSize < 1 or pageSize > 100:
        raise HTTPException(status_code=400, detail="分页参数不合法")

    stmt = select(Run)
    count_stmt = select(func.count()).select_from(Run)

    if query:
        like = f"%{query}%"
        stmt = stmt.where(Run.dataset_name.ilike(like))
        count_stmt = count_stmt.where(Run.dataset_name.ilike(like))

    total = int(db.execute(count_stmt).scalar_one())
    stmt = stmt.order_by(Run.created_at.desc()).offset((page - 1) * pageSize).limit(pageSize)
    items = [
        _to_result(r)
        for r in db.execute(stmt).scalars().all()
    ]
    return RunListResponse(items=items, total=total)


@router.get("/{run_id}", response_model=RunResult)
def get_run(run_id: str, db: Session = Depends(get_db)):
    """获取单条实验记录详情。"""
    try:
        uid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="id 不合法")

    row = db.get(Run, uid)
    if not row:
        raise HTTPException(status_code=404, detail="记录不存在")
    return _to_result(row)
