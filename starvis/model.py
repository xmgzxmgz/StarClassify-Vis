"""
轻量化集成模型训练模块（Model Training）

职责：
- 使用逻辑回归与高斯朴素贝叶斯作为基模型；
- 采用 VotingClassifier（软投票，voting='soft'）进行概率加权融合；
- 提供预测概率接口，便于科研筛选与案例演示。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ModelTrainer:
    """模型训练器，封装集成策略与预测接口。"""

    def __init__(self) -> None:
        """初始化模型训练器。"""
        self._model = None

    def train_voting_classifier(self, X_train: np.ndarray, y_train: np.ndarray, weights: Tuple[float, float] = (0.6, 0.4)):
        """训练软投票集成分类器（逻辑回归 + 朴素贝叶斯）。

        参数：
            X_train: 训练集特征矩阵
            y_train: 训练集标签向量
            weights: 两个基模型在软投票中的权重（默认 0.6 : 0.4）
        返回：
            训练好的 VotingClassifier 模型对象
        物理意义：
            逻辑回归擅长线性边界，朴素贝叶斯擅长高维概率分布；软投票融合提升稳健性与精度。
        """
        # 懒加载 scikit-learn 依赖
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import VotingClassifier

        # 去除 n_jobs 以提高不同版本 scikit-learn 的兼容性
        lr = LogisticRegression(max_iter=500, random_state=42)
        nb = GaussianNB()
        model = VotingClassifier(estimators=[("lr", lr), ("nb", nb)], voting="soft", weights=list(weights))
        model.fit(X_train, y_train)
        self._model = model
        return model

    def predict_proba(self, model, X_new: np.ndarray) -> np.ndarray:
        """预测概率分布。

        参数：
            model: 已训练模型对象
            X_new: 新的特征矩阵
        返回：
            概率分布矩阵（shape: [n_samples, n_classes]）
        物理意义：
            在科研筛选场景下，置信度能指导是否需要人工复核。
        """
        return model.predict_proba(X_new)