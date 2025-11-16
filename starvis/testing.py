"""
系统测试与案例演示模块（Testing & Cases）

职责：
- 提供基础单元测试函数，验证数据加载与模型预测输出形状等；
- 为每个核心模块提供可调用的测试 API；
- 支持一键快速跑通端到端流程并返回评估指标。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def test_data_not_empty(df: pd.DataFrame) -> bool:
    """测试数据加载是否为空。

    参数：
        df: 数据框
    返回：
        bool：为 True 则通过，False 则失败
    物理意义：
        基础检查，避免后续步骤因空数据而崩溃。
    """
    return len(df) > 0 and len(df.columns) > 0


def test_predict_shape(y_proba: np.ndarray) -> bool:
    """测试模型预测概率输出形状是否正确。

    参数：
        y_proba: 预测概率矩阵
    返回：
        bool：为 True 则通过，False 则失败
    物理意义：
        保证科研筛选工具中概率输出满足后续处理的需求。
    """
    return y_proba.ndim == 2 and y_proba.shape[1] >= 2


# ===== 模块级测试 API =====

def test_data_loader_api(loader) -> Tuple[bool, pd.DataFrame, Dict[str, float]]:
    """数据加载模块测试 API：生成或读取模拟 SDSS 数据并返回信息。

    参数：
        loader: DataLoader 实例
    返回：
        (通过标志, DataFrame, 信息字典)
    物理意义：
        快速验证数据入口与缓存机制是否可用。
    """
    df = loader.generate_mock_sdss(n=5000)
    ok = test_data_not_empty(df)
    info = {"rows": float(len(df)), "cols": float(len(df.columns)), "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2))}
    return ok, df, info


def test_feature_engineer_api(engineer, df: pd.DataFrame):
    """特征工程模块测试 API：构建与筛选特征。

    参数：
        engineer: FeatureEngineer 实例
        df: 原始数据框
    返回：
        (features_df, feature_names)
    物理意义：
        验证颜色指数与空间/物理特征构建是否正常。
    """
    features_df, names = engineer.build_features(df)
    return features_df, names


def test_preprocessor_api(preprocessor, features_df: pd.DataFrame, raw_df: pd.DataFrame):
    """预处理模块测试 API：缺失值填充、3σ 剔除、标签编码与标准化。

    参数：
        preprocessor: Preprocessor 实例
        features_df: 特征数据框
        raw_df: 原始数据框（包含 class 标签）
    返回：
        (X, y, label_encoder)
    物理意义：
        验证预处理流程可在标准数据集上稳定运行。
    """
    X, y, le = preprocessor.prepare_xy(features_df, raw_df)
    return X, y, le


def test_model_trainer_api(trainer, X: np.ndarray, y: np.ndarray):
    """模型训练模块测试 API：训练软投票模型并返回概率输出形状。

    参数：
        trainer: ModelTrainer 实例
        X: 特征矩阵
        y: 标签向量
    返回：
        (model, y_proba_shape)
    物理意义：
        验证集成模型训练与预测概率接口是否正常。
    """
    # 小样本训练更快
    n = min(2000, len(X))
    model = trainer.train_voting_classifier(X[:n], y[:n], weights=(0.6, 0.4))
    proba = trainer.predict_proba(model, X[:100])
    return model, proba.shape


def test_evaluator_api(evaluator, model, X: np.ndarray, y: np.ndarray):
    """评估模块测试 API：计算指标与混淆矩阵图。

    参数：
        evaluator: Evaluator 实例
        model: 已训练模型
        X: 特征矩阵（测试集）
        y: 标签向量（测试集）
    返回：
        (metrics, fig_cm)
    物理意义：
        验证评估输出是否可视化与数值化正常。
    """
    metrics, fig = evaluator.evaluate(model, X, y)
    return metrics, fig


def run_pipeline_quick(loader, engineer, preprocessor, trainer, evaluator):
    """一键快速跑通端到端流程并返回评估指标。

    参数：
        loader: DataLoader
        engineer: FeatureEngineer
        preprocessor: Preprocessor
        trainer: ModelTrainer
        evaluator: Evaluator
    返回：
        dict：包含训练/评估的关键数值与形状信息
    物理意义：
        用于 Streamlit 测试中心页面的端到端验证，保障系统整体可用性。
    """
    df = loader.generate_mock_sdss(n=8000)
    feats, names = engineer.build_features(df)
    X, y, le = preprocessor.prepare_xy(feats, df)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
    model = trainer.train_voting_classifier(X_train, y_train, weights=(0.6, 0.4))
    metrics, fig_cm = evaluator.evaluate(model, X_test, y_test)

    return {
        "rows": len(df),
        "n_features": len(names),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "fig_cm": fig_cm,
    }