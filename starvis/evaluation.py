"""
模型评估与特征分析模块（Evaluation & Analysis）

职责：
- 计算准确率、精确率、召回率与 F1 分数；
- 生成混淆矩阵图表；
- 提取特征重要性（逻辑回归系数或排列重要性）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# 尝试懒加载 seaborn；若不可用，则使用 matplotlib 退化绘图，避免应用启动失败
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


class Evaluator:
    """评估器，提供指标计算与可视化能力。"""

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray):
        """计算分类指标并绘制混淆矩阵。

        参数：
            model: 已训练模型
            X_test: 测试集特征
            y_test: 测试集标签
        返回：
            (metrics_dict, fig_cm)
        物理意义：
            定量评估模型表现，识别易误判的类别以指导特征工程优化。
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        if _HAS_SNS:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        else:
            im = ax.imshow(cm, cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            # 在格子中标注数值
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
        ax.set_title("混淆矩阵")
        ax.set_xlabel("预测类")
        ax.set_ylabel("真实类")
        fig.tight_layout()
        return metrics, fig

    def feature_importance(self, model, X_test: np.ndarray):
        """估计特征重要性。

        参数：
            model: VotingClassifier 模型
            X_test: 测试集特征
        返回：
            matplotlib.figure.Figure 或 None
        物理意义：
            在 VotingClassifier 无直接 importances 时，采用逻辑回归系数或排列重要性近似解释。
        """
        # 使用逻辑回归系数作为第一优先解释
        lr = None
        for name, est in model.named_estimators_.items():
            if name == "lr":
                lr = est
                break
        if lr is None:
            return None

        try:
            coefs = lr.coef_.mean(axis=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(coefs)), coefs)
            ax.set_title("逻辑回归系数（近似特征重要性）")
            ax.set_xlabel("特征索引")
            ax.set_ylabel("系数值")
            fig.tight_layout()
            return fig
        except Exception:
            # 回退：排列重要性（可能较慢）
            try:
                from sklearn.inspection import permutation_importance
                r = permutation_importance(lr, X_test, model.predict(X_test), n_repeats=5, random_state=42)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(len(r.importances_mean)), r.importances_mean)
                ax.set_title("排列重要性（近似）")
                ax.set_xlabel("特征索引")
                ax.set_ylabel("重要性均值")
                fig.tight_layout()
                return fig
            except Exception:
                return None