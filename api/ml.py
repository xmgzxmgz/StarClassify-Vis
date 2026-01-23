from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MlOutput:
    metrics: dict
    confusion_matrix: list[list[int]]
    labels: list[str]


def train_gaussian_nb(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    test_size: float,
    random_state: int | None,
    var_smoothing: float | None,
) -> MlOutput:
    for c in [target_column, *feature_columns]:
        if c not in df.columns:
            raise ValueError(f"CSV 中缺少列: {c}")

    cols = [*feature_columns, target_column]
    df2 = df[cols].copy()
    df2 = df2.dropna(axis=0, how="any")
    if len(df2) < 10:
        raise ValueError("有效样本过少（去除缺失值后 < 10 行），无法训练")

    X_raw = df2[feature_columns]
    X = X_raw.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("特征列必须为数值型；请检查是否包含非数字或缺失值")

    y_raw = df2[target_column]
    y = y_raw.astype(str)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values if y.nunique() > 1 else None,
    )

    if len(set(y_train)) < 2:
        raise ValueError("训练集类别数不足（<2），无法训练分类器")

    from sklearn.naive_bayes import GaussianNB

    kwargs = {}
    if var_smoothing is not None:
        if not (isinstance(var_smoothing, (int, float)) and var_smoothing >= 0 and math.isfinite(var_smoothing)):
            raise ValueError("var_smoothing 必须为非负有限数")
        kwargs["var_smoothing"] = float(var_smoothing)

    model = GaussianNB(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    labels = sorted(set(y_test.tolist()), key=lambda s: s)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return MlOutput(
        metrics={
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        confusion_matrix=cm.astype(int).tolist(),
        labels=labels,
    )

