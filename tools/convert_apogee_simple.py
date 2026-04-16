#!/usr/bin/env python3
"""
将APOGEE FITS文件转换为CSV格式（简化版本）
"""

import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def read_apogee_fits(filepath):
    """
    读取APOGEE FITS文件
    """
    print(f"读取APOGEE FITS文件: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / (1024**3):.2f} GB")

    try:
        with fits.open(filepath) as hdul:
            print(f"HDU数量: {len(hdul)}")
            print(f"主数据HDU: {hdul[1].name}")
            print(f"记录数: {len(hdul[1].data)}")
            print(f"列数: {len(hdul[1].columns)}")

            data = hdul[1].data
            columns = hdul[1].columns

            return data, columns

    except Exception as e:
        print(f"读取失败: {e}")
        return None, None


def convert_to_csv_simple(data, output_path, max_rows=100000):
    """
    简化的转换方法，避免字节序问题
    """
    print(f"\n转换数据...")

    total_rows = len(data)
    if max_rows and max_rows < total_rows:
        total_rows = max_rows

    print(f"处理 {total_rows} 条记录")

    # 选择需要的列
    columns_to_extract = [
        'APOGEE_ID', 'RA', 'DEC', 'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
        'FE_H', 'FE_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 'C_FE', 'N_FE', 'O_FE',
        'MG_FE', 'SI_FE', 'S_FE', 'CA_FE', 'NI_FE', 'VHELIO_AVG', 'SNR',
        'NVISITS', 'FIELD', 'LOCATION_ID'
    ]

    # 检查哪些列可用
    available_columns = []
    for col in columns_to_extract:
        if col in data.names:
            available_columns.append(col)

    print(f"可用列: {len(available_columns)}")

    # 创建数据字典
    data_dict = {}

    # 逐个处理列
    for col_name in tqdm(available_columns, desc="处理列"):
        try:
            # 直接获取数据并转换为列表
            values = data[col_name][:total_rows]
            
            # 对于字符串类型，确保正确处理
            if isinstance(values[0], bytes):
                values = [v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else v for v in values]
            
            data_dict[col_name.lower()] = values
        except Exception as e:
            print(f"处理 {col_name} 时出错: {e}")

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 过滤数据
    print(f"\n过滤数据...")
    initial_count = len(df)

    # 过滤温度
    if 'teff' in df.columns:
        df = df[(df['teff'] > 3000) & (df['teff'] < 15000)]

    # 过滤表面重力
    if 'logg' in df.columns:
        df = df[(df['logg'] > 0) & (df['logg'] < 6)]

    # 过滤金属丰度
    if 'fe_h' in df.columns:
        df = df[(df['fe_h'] > -5) & (df['fe_h'] < 1)]

    # 过滤信噪比
    if 'snr' in df.columns:
        df = df[df['snr'] > 5]

    print(f"过滤后: {len(df)} / {initial_count} 条")

    # 重命名列
    df = df.rename(columns={
        'fe_h': 'feh',
        'fe_h_err': 'feh_err',
        'alpha_m': 'alpha_m',
        'alpha_m_err': 'alpha_m_err',
        'vhelio_avg': 'vhelio',
        'nvisits': 'nvisits',
        'location_id': 'location_id'
    })

    # 保存为CSV
    print(f"\n保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    return df


def generate_report(df, output_path):
    """
    生成报告
    """
    print(f"\n{'='*60}")
    print(f"APOGEE DR17 数据转换报告")
    print(f"{'='*60}")

    print(f"\n输出文件: {output_path}")
    print(f"总记录数: {len(df):,}")

    print(f"\n恒星参数统计:")
    if 'teff' in df.columns:
        print(f"  有效温度: {df['teff'].min():.0f} - {df['teff'].max():.0f} K")
        print(f"  平均温度: {df['teff'].mean():.0f} K")

    if 'logg' in df.columns:
        print(f"  表面重力: {df['logg'].min():.2f} - {df['logg'].max():.2f}")
        print(f"  平均重力: {df['logg'].mean():.2f}")

    if 'feh' in df.columns:
        print(f"  金属丰度: {df['feh'].min():.2f} - {df['feh'].max():.2f}")
        print(f"  平均丰度: {df['feh'].mean():.2f}")

    if 'snr' in df.columns:
        print(f"  信噪比: {df['snr'].min():.1f} - {df['snr'].max():.1f}")
        print(f"  平均SNR: {df['snr'].mean():.1f}")

    print(f"\n{'='*60}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="将APOGEE FITS文件转换为CSV格式")
    parser.add_argument('--input', '-i',
                        default=os.path.join(OUTPUT_DIR, 'allStar-dr17-synspec_rev1.fits'),
                        help="输入FITS文件路径")
    parser.add_argument('--output', '-o',
                        default=os.path.join(OUTPUT_DIR, 'apogee_dr17.csv'),
                        help="输出CSV文件路径")
    parser.add_argument('--limit', '-l', type=int, default=100000,
                        help="限制处理的记录数")

    args = parser.parse_args()

    data, columns = read_apogee_fits(args.input)

    if data is None:
        print("读取FITS文件失败")
        return

    df = convert_to_csv_simple(data, args.output, args.limit)

    generate_report(df, args.output)

    print(f"\n完成! 数据已保存到: {args.output}")


if __name__ == "__main__":
    main()
