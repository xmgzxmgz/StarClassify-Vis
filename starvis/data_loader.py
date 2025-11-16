"""
数据获取模块（Data Acquisition）

职责：
- 负责从本地 CSV 或上传缓冲区读取数据，并进行基本信息汇总；
- 若文件不存在，自动生成一份模拟的 SDSS 样本数据，确保系统可运行；
- 提供简单的缓存机制，避免重复读取大文件；

物理意义：
在大数据场景下，稳定的数据入口与可回退的模拟数据能提升系统健壮性。
"""

from __future__ import annotations

import os
import io
import time
from typing import Dict

import numpy as np
import pandas as pd


class DataLoader:
    """数据加载器，封装文件读取与模拟数据生成逻辑。"""

    def __init__(self, cache_dir: str = "data") -> None:
        """初始化数据加载器。

        参数：
            cache_dir: 用于存放缓存文件的目录。
        返回：
            None
        物理意义：
            统一数据缓存位置，便于管理与加速重复读取。
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, source_key: str) -> str:
        """根据数据源键生成缓存路径。"""
        safe_key = source_key.replace(os.sep, "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"cache_{safe_key}.parquet")

    def load_csv_or_mock(self, csv_path: str) -> pd.DataFrame:
        """从 CSV 加载数据，若不存在则生成模拟 SDSS 数据。

        参数：
            csv_path: 本地 CSV 路径。
        返回：
            DataFrame
        物理意义：
            保证系统在缺少真实数据时仍可运行与教学演示。
        """
        try:
            if os.path.exists(csv_path):
                return self.load_csv(csv_path)
            df = self.generate_mock_sdss(n=10000)
            df.to_csv(csv_path, index=False)
            return df
        except Exception as e:
            raise RuntimeError(f"加载或生成模拟数据失败：{e}")

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """读取 CSV 并基于修改时间进行简单缓存。

        参数：
            csv_path: CSV 文件路径。
        返回：
            DataFrame
        物理意义：
            基于时间戳的轻量缓存可减少重复 IO 并提升响应速度。
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在：{csv_path}")

        mtime = os.path.getmtime(csv_path)
        cache_path = self._cache_path(f"{csv_path}_{mtime}")

        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                # 缓存损坏则回退到重新读取
                pass

        df = pd.read_csv(csv_path)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            # 某些环境可能缺少 parquet 依赖，忽略缓存错误
            pass
        return df

    def load_from_buffer(self, buffer: io.BytesIO) -> pd.DataFrame:
        """从上传缓冲区读取 CSV。

        参数：
            buffer: Streamlit 上传的文件缓冲区。
        返回：
            DataFrame
        物理意义：
            便于用户直接上传自有数据进行分析与训练。
        """
        buffer.seek(0)
        return pd.read_csv(buffer)

    def get_info(self, df: pd.DataFrame) -> Dict[str, float]:
        """输出数据基本信息（行列数、列名、内存占用）。

        参数：
            df: 数据框
        返回：
            包含 rows, cols, memory_mb 的字典
        物理意义：
            快速了解数据规模与内存压力，便于合理设置训练参数。
        """
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        return {"rows": float(len(df)), "cols": float(len(df.columns)), "memory_mb": float(memory_mb)}

    def generate_mock_sdss(self, n: int = 10000) -> pd.DataFrame:
        """生成模拟 SDSS 恒星数据，包含光谱、空间与环境特征。

        参数：
            n: 样本量
        返回：
            模拟数据 DataFrame
        物理意义：
            用概率规则生成不同类型恒星，支持教学演示与可视化。
        """
        rng = np.random.default_rng(42)
        # 基本光度（类似 SDSS 五个波段星等）
        u = rng.normal(18.5, 1.2, n)
        g = rng.normal(17.8, 1.1, n)
        r = rng.normal(17.3, 1.0, n)
        i = rng.normal(17.0, 1.0, n)
        z = rng.normal(16.7, 1.0, n)

        # 红移与空间坐标
        redshift = rng.normal(0.001, 0.002, n)
        ra = rng.uniform(0, 360, n)
        dec = rng.uniform(-90, 90, n)

        # 观测环境
        mjd = rng.integers(52000, 54000, n)
        plate = rng.integers(1, 1000, n)

        # 物理参数近似：表面重力与金属丰度
        logg = rng.normal(4.2, 0.5, n)
        feh = rng.normal(-0.2, 0.3, n)

        # 类型规则生成（简单近似）：主序星、红巨星、白矮星
        temp = rng.normal(6000, 1500, n)
        types = []
        for t, g_r, lg in zip(temp, g - r, logg):
            if t > 9000 and lg > 7.5:
                types.append("WHITE_DWARF")
            elif t < 5000 and g_r > 0.6 and lg < 3.5:
                types.append("RED_GIANT")
            else:
                types.append("MAIN_SEQUENCE")

        df = pd.DataFrame({
            "u": u, "g": g, "r": r, "i": i, "z": z,
            "redshift": redshift, "ra": ra, "dec": dec,
            "mjd": mjd, "plate": plate,
            "logg": logg, "feh": feh,
            "temp": temp,
            "class": types,
        })
        return df