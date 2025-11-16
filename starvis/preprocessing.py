"""
数据清洗与预处理模块（Data Preprocessing）

职责：
- 缺失值处理：数值型均值填充，分类型众数填充；
- 异常值剔除：对红移与温度应用 3σ 法则剔除噪点；
- 标签编码：将恒星类别字符串映射为数值索引；
- 标准化：对连续变量进行标准化，消除量纲影响。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class Preprocessor:
    """预处理器，封装清洗与特征标准化逻辑。"""

    def __init__(self) -> None:
        """初始化预处理器。"""
        self._scaler = None
        self._label_encoder = None

    def remove_outliers_3sigma(self, df: pd.DataFrame, cols=("redshift", "temp")) -> pd.DataFrame:
        """对指定列应用 3σ 法则剔除异常样本。

        参数：
            df: 原始数据框
            cols: 需要剔除异常值的列名元组
        返回：
            剔除异常后的数据框
        物理意义：
            减少极端噪点对模型训练的干扰，尤其是红移与温度等物理量。
        """
        clean_df = df.copy()
        for c in cols:
            if c not in clean_df.columns:
                continue
            mu = clean_df[c].mean()
            sigma = clean_df[c].std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                continue
            mask = (clean_df[c] >= mu - 3 * sigma) & (clean_df[c] <= mu + 3 * sigma)
            clean_df = clean_df[mask]
        return clean_df

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据框进行缺失值填充：数值型用均值，分类型用众数。

        参数：
            df: 数据框
        返回：
            填充后的数据框
        物理意义：
            保持数据完整性，避免因缺失造成训练偏差或崩溃。
        """
        fill_df = df.copy()
        for c in fill_df.columns:
            if pd.api.types.is_numeric_dtype(fill_df[c]):
                fill_df[c] = fill_df[c].fillna(fill_df[c].mean())
            else:
                mode_val = fill_df[c].mode().iloc[0] if not fill_df[c].mode().empty else "UNKNOWN"
                fill_df[c] = fill_df[c].fillna(mode_val)
        return fill_df

    def prepare_xy(self, features_df: pd.DataFrame, raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, object]:
        """组合特征与标签并完成标准化与编码。

        参数：
            features_df: 特征数据框（已筛选）
            raw_df: 原始数据框（包含 class 标签）
        返回：
            X: 标准化后的特征矩阵
            y: 数值化的标签向量
            label_encoder: 标签编码器对象
        物理意义：
            将清洗后的特征与标签准备为模型可直接使用的数值表示。
        """
        # 懒加载依赖
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        df = features_df.copy()
        df = self.fill_missing(df)
        df = self.remove_outliers_3sigma(df, cols=("redshift", "temp")) if "temp" in df.columns else self.remove_outliers_3sigma(df, cols=("redshift",))

        # 标签编码（与过滤同步）
        label_series = raw_df.loc[df.index, "class"].copy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_series.astype(str))

        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)

        self._scaler = scaler
        self._label_encoder = label_encoder
        return X, y, label_encoder

    def train_test_split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
        """按照给定比例划分训练集与测试集，保证可复现性。

        参数：
            X: 特征矩阵
            y: 标签向量
            train_ratio: 训练集比例（默认 0.8）
        返回：
            X_train, X_test, y_train, y_test
        物理意义：
            通过固定随机种子实现可复现的实验设置。
        """
        from sklearn.model_selection import train_test_split as _split

        return _split(X, y, train_size=train_ratio, random_state=42, stratify=y)

    def transform_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """将新的特征数据框转换为已训练标准化器下的数值矩阵。

        参数：
            features_df: 特征数据框
        返回：
            X_new: 标准化后的特征矩阵
        物理意义：
            保证科研筛选与案例演示等环节与训练阶段的特征量纲一致。
        """
        if self._scaler is None:
            raise RuntimeError("尚未初始化标准化器，请先进行训练阶段的 prepare_xy。")
        df = self.fill_missing(features_df)
        return self._scaler.transform(df.values)