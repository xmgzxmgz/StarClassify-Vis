#!/usr/bin/env python3
"""
使用astroquery从SDSS DR17获取真实数据
"""

import os
import argparse
import numpy as np
import pandas as pd
from astroquery.sdss import SDSS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def query_data(data_release='DR17'):
    """查询数据 - 不过滤，在Python中处理"""
    print(f"\n{'='*60}")
    print(f"从SDSS {data_release}获取真实数据")
    print(f"{'='*60}")

    all_data = []

    queries = [
        ("STAR", 30000),
        ("GALAXY", 30000),
        ("QSO", 20000),
    ]

    for class_name, limit in queries:
        print(f"\n查询 {class_name} (限制 {limit})...")

        sql = f"SELECT TOP {limit} objid, ra, dec, plate, mjd, fiberid, z, class, subclass, sn_median FROM SpecObjAll WHERE class = '{class_name}'"

        try:
            result = SDSS.query_sql(sql, data_release=data_release)
            if result is not None:
                df = result.to_pandas()
                print(f"  获取到 {len(df)} 条数据")
                all_data.append(df)
            else:
                print(f"  未获取到数据")
        except Exception as e:
            print(f"  查询失败: {e}")

    if not all_data:
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
        df = df[df['class'].isin(['STAR', 'GALAXY', 'QSO'])]

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
    print(f"  最小值: {df['redshift'].min():.4f}")
    print(f"  最大值: {df['redshift'].max():.4f}")
    print(f"  平均值: {df['redshift'].mean():.4f}")

    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用astroquery从SDSS DR17获取真实数据")
    parser.add_argument('--output', '-o',
                        default=os.path.join(OUTPUT_DIR, 'sdss_dr17_real.csv'),
                        help="输出CSV文件路径")
    parser.add_argument('--dr', default='DR17', choices=['DR17', 'DR18', 'DR19'],
                        help="SDSS数据发布版本")

    args = parser.parse_args()

    df = query_data(data_release=args.dr)

    if df is None or len(df) == 0:
        print("\n获取数据失败")
        return

    process_and_save(df, args.output)

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
