"""
通用工具模块（Utils）

职责：
- 目录管理、 Plotly 绘图工具、赫罗图交互支持；
- 科普解释文本与示意图；
- 案例数据构建。
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px


def ensure_data_dir(path: str = "data") -> None:
    """确保数据目录存在，若不存在则自动创建。

    参数：
        path: 目录路径（默认 'data'）
    返回：
        None
    物理意义：
        统一数据存放位置，降低用户操作复杂度。
    """
    os.makedirs(path, exist_ok=True)


def plot_sky_distribution(df: pd.DataFrame):
    """绘制恒星在天球上的空间分布散点图（ra vs dec）。

    参数：
        df: 数据框，需包含 'ra' 与 'dec'
    返回：
        Plotly Figure
    物理意义：
        从空间角度理解恒星分布，有助于区分不同恒星种群的空间规律。
    """
    if not set(["ra", "dec"]).issubset(df.columns):
        raise ValueError("数据不包含 ra 或 dec 列，无法绘制空间分布。")
    fig = px.scatter(df.sample(min(len(df), 5000), random_state=42), x="ra", y="dec",
                     title="天球空间分布（采样）", opacity=0.6, height=400)
    fig.update_layout(xaxis_title="赤经 (deg)", yaxis_title="赤纬 (deg)")
    return fig


def plot_hr_diagram(df: pd.DataFrame, features_df: pd.DataFrame, highlight_point: bool = False, temp: int = 6000):
    """绘制赫罗图（近似）：横轴为颜色指数/温度，纵轴为星等/光度。

    参数：
        df: 原始数据框（包含 g 与 r 星等）
        features_df: 特征数据框（包含 g_r 或 temp）
        highlight_point: 是否高亮用户选择的点
        temp: 用户选择的有效温度，用于近似映射到颜色指数
    返回：
        Plotly Figure
    物理意义：
        赫罗图展示了恒星光度与颜色（温度）的关系，是理解恒星演化的核心图表。
    """
    # 统一在一个 DataFrame 中构造所需列，避免传入长度不一致的 Series
    data = df.copy()
    if "g" in data.columns and "r" in data.columns:
        data["g_r"] = data["g"] - data["r"]
    elif "g_r" in features_df.columns:
        # 对齐索引后再赋值，防止长度不一致
        aligned = features_df.reindex(data.index)
        data["g_r"] = aligned["g_r"].values
    else:
        # 无法计算颜色指数时，近似以温度替代
        if "temp" in data.columns:
            data["g_r"] = - (data["temp"] - data["temp"].mean()) / data["temp"].std(ddof=0)
        else:
            data["g_r"] = 0.0

    # 选择 y 轴列名，并确保存在于同一 DataFrame
    y_col = "r" if "r" in data.columns else ("g" if "g" in data.columns else None)
    if y_col is None:
        # 构造一个代理星等列，以免绘图失败
        y_col = "mag_proxy"
        data[y_col] = 0.5

    # 采样后绘图，x/y 使用列名字符串，避免 Narwhals 对齐报错
    sample_df = data.sample(min(len(data), 6000), random_state=42)
    fig = px.scatter(sample_df, x="g_r", y=y_col,
                     title="赫罗图（颜色指数 vs r 星等，采样）", opacity=0.6, height=420)
    fig.update_layout(xaxis_title="颜色指数 g-r（温度代理）", yaxis_title="r 星等（光度代理）", yaxis_autorange="reversed")

    if highlight_point:
        # 简单温度到颜色指数的近似映射：温度高 -> 颜色更蓝 (g-r 更小)
        gr_point = - (temp - 6000) / 3000
        fig.add_scatter(x=[gr_point], y=[sample_df[y_col].mean()], mode="markers", marker=dict(size=12, color="red"), name="当前选择")
    return fig


def classify_by_rules(temp: float, feh: float) -> str:
    """基于温度与金属丰度的简单规则判定恒星类型（科普近似）。

    参数：
        temp: 有效温度（K）
        feh: 金属丰度 [Fe/H]
    返回：
        类型字符串，如 'RED_GIANT'/'MAIN_SEQUENCE'/'WHITE_DWARF'
    物理意义：
        科普模式下的简化判定，帮助用户直观理解不同类型的典型特征范围。
    """
    if temp > 9000 and feh < -0.5:
        return "WHITE_DWARF"
    if temp < 5000 and feh > -0.5:
        return "RED_GIANT"
    return "MAIN_SEQUENCE"


def explain_star_class(cls: str) -> Tuple[str, str]:
    """返回恒星类型的中文解释与示意图链接。

    参数：
        cls: 恒星类型字符串
    返回：
        (解释文本, 图片 URL)
    物理意义：
        提供科普背景与艺术示意图，增强学习体验。
    """
    if cls == "RED_GIANT":
        return (
            "这是一颗红巨星：核心氦聚变或壳层氢聚变活跃，体积膨胀、温度较低但光度很高，生命已进入晚年阶段。",
            "https://upload.wikimedia.org/wikipedia/commons/7/77/Red_Giant_Sun.png",
        )
    if cls == "WHITE_DWARF":
        return (
            "这是一颗白矮星：主序阶段后塌缩形成的致密天体，温度高但体积极小，光度不高，缓慢冷却。",
            "https://upload.wikimedia.org/wikipedia/commons/6/6f/Procyon_system.jpg",
        )
    return (
        "这是一颗主序星：核心稳定进行氢聚变，处于生命的漫长中年阶段，是银河系中最常见的恒星类型。",
        "https://upload.wikimedia.org/wikipedia/commons/9/99/Sun_in_true_color.jpg",
    )


def build_case_dataset() -> pd.DataFrame:
    """构建典型案例数据：太阳、天狼星（含白矮星伴星）、参宿四。

    返回：
        DataFrame：包含必要的光度与物理特征。
    物理意义：
        用知名恒星辅助教学，直观展示模型分类与概率输出。
    """
    # 数值非精确天文值，仅用于教学示例
    data = [
        {"name": "太阳", "u": 5.0, "g": 4.83, "r": 4.50, "i": 4.40, "z": 4.30, "redshift": 0.0, "feh": 0.0, "logg": 4.44, "ra": 0.0, "dec": 0.0, "temp": 5772, "class": "MAIN_SEQUENCE"},
        {"name": "天狼星A", "u": 1.0, "g": 1.40, "r": 1.50, "i": 1.55, "z": 1.60, "redshift": 0.0, "feh": -0.2, "logg": 4.3, "ra": 101.287, "dec": -16.716, "temp": 9900, "class": "MAIN_SEQUENCE"},
        {"name": "天狼星B", "u": 0.5, "g": 1.0, "r": 1.2, "i": 1.3, "z": 1.4, "redshift": 0.0, "feh": -0.7, "logg": 8.7, "ra": 101.287, "dec": -16.716, "temp": 25000, "class": "WHITE_DWARF"},
        {"name": "参宿四", "u": 0.0, "g": 0.3, "r": -0.5, "i": -1.0, "z": -1.3, "redshift": 0.0, "feh": -0.2, "logg": 1.5, "ra": 88.793, "dec": 7.407, "temp": 3600, "class": "RED_GIANT"},
    ]
    return pd.DataFrame(data)