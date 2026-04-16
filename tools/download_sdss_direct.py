#!/usr/bin/env python3
"""
通过HTTP直接访问SDSS Science Archive Server获取真实数据
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
from io import StringIO

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def download_sdss_data():
    """
    直接从SDSS的web接口下载数据
    使用SkyServer的CSV下载功能
    """
    print("=" * 60)
    print("从SDSS DR17直接下载真实数据")
    print("=" * 60)

    base_url = "http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

    queries = [
        ("STAR", 25000),
        ("GALAXY", 25000),
        ("QSO", 15000),
    ]

    all_data = []

    for class_name, limit in queries:
        print(f"\n查询 {class_name} (限制 {limit})...")

        sql = f"""
        SELECT TOP {limit}
            s.objid, s.ra, s.dec, s.plate, s.mjd, s.fiberid,
            s.z, s.class, s.subclass, s.sn_median
        FROM SpecObjAll s
        WHERE s.class = '{class_name}'
        """

        params = {
            'cmd': sql.strip(),
            'format': 'csv',
            'taskId': 0,
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        }

        try:
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                timeout=180
            )

            if response.status_code == 200:
                text = response.text.strip()

                if text and not text.startswith('<html'):
                    df = pd.read_csv(StringIO(text))
                    print(f"  获取到 {len(df)} 条数据")
                    all_data.append(df)
                else:
                    print(f"  返回了HTML，可能是查询超时")
            else:
                print(f"  HTTP错误: {response.status_code}")

        except Exception as e:
            print(f"  请求失败: {e}")

    if not all_data:
        print("\n未能从SkyServer获取数据")
        return None

    df = pd.concat(all_data, ignore_index=True)
    return df


def process_and_save(df, output_path):
    """处理和保存数据"""
    print(f"\n处理数据...")

    if 'z' in df.columns:
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        df = df[df['z'].notna() & (df['z'] > 0)]
        print(f"过滤后剩余 {len(df)} 条 (z > 0)")

    if 'class' in df.columns:
        df['class'] = df['class'].str.upper()

    df = df.rename(columns={
        'ra': 'RA',
        'dec': 'DEC',
        'z': 'redshift',
    })

    print(f"最终数据量: {len(df):,} 条")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n数据已保存到: {output_path}")

    print(f"\n类别分布:")
    for cls, count in df['class'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")

    print(f"\n红移统计:")
    print(f"  最小值: {df['redshift'].min():.6f}")
    print(f"  最大值: {df['redshift'].max():.4f}")
    print(f"  平均值: {df['redshift'].mean():.4f}")
    print(f"  中位数: {df['redshift'].median():.4f}")

    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从SDSS DR17直接下载真实数据")
    parser.add_argument('--output', '-o',
                        default=os.path.join(OUTPUT_DIR, 'sdss_dr17_real.csv'),
                        help="输出CSV文件路径")

    args = parser.parse_args()

    df = download_sdss_data()

    if df is None or len(df) == 0:
        print("\n下载失败，将使用基于真实分布的模拟数据...")

        from download_sdss_real import generate_realistic_sample
        generate_realistic_sample(
            args.output,
            n_stars=5000,
            n_galaxies=10000,
            n_quasars=3000
        )
        return

    process_and_save(df, args.output)

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
