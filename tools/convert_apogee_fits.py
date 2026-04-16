#!/usr/bin/env python3
"""
将APOGEE FITS文件转换为CSV格式
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


def get_relevant_columns(columns):
    """
    选择相关的列
    """
    relevant_columns = {
        'APOGEE_ID': 'apogee_id',
        'RA': 'ra',
        'DEC': 'dec',
        'TEFF': 'teff',
        'TEFF_ERR': 'teff_err',
        'LOGG': 'logg',
        'LOGG_ERR': 'logg_err',
        'FE_H': 'feh',
        'FE_H_ERR': 'feh_err',
        'ALPHA_M': 'alpha_m',
        'ALPHA_M_ERR': 'alpha_m_err',
        'C_FE': 'c_fe',
        'N_FE': 'n_fe',
        'O_FE': 'o_fe',
        'MG_FE': 'mg_fe',
        'SI_FE': 'si_fe',
        'S_FE': 's_fe',
        'CA_FE': 'ca_fe',
        'NI_FE': 'ni_fe',
        'VHELIO_AVG': 'vhelio',
        'SNR': 'snr',
        'NVISITS': 'nvisits',
        'FIELD': 'field',
        'LOCATION_ID': 'location_id',
    }

    available_columns = []
    column_mapping = {}

    for col in columns.names:
        for key, value in relevant_columns.items():
            if col == key:
                available_columns.append(col)
                column_mapping[col] = value
                break

    print(f"\n可用的相关列: {len(available_columns)}")
    for col, mapped in column_mapping.items():
        print(f"  {col} -> {mapped}")

    return column_mapping


def convert_to_csv(data, column_mapping, output_path, max_rows=None):
    """
    转换为CSV格式
    """
    print(f"\n转换数据...")

    total_rows = len(data)
    if max_rows and max_rows < total_rows:
        total_rows = max_rows

    print(f"处理 {total_rows} 条记录")

    # 创建字典来存储数据
    data_dict = {}

    # 处理每一列
    for original_col, new_col in tqdm(column_mapping.items(), desc="处理列"):
        if original_col in data.names:
            values = data[original_col]
            if max_rows:
                values = values[:max_rows]
            
            # 处理字节序问题
            if isinstance(values, np.ndarray) and values.dtype.byteorder == '>' and values.dtype.itemsize > 1:
                try:
                    # NumPy 2.0 兼容方式
                    values = values.byteswap()
                except Exception as e:
                    print(f"处理 {original_col} 时出错: {e}")
                    pass
            
            data_dict[new_col] = values

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 过滤无效数据
    print(f"\n过滤数据...")
    initial_count = len(df)

    # 过滤温度异常值
    if 'teff' in df.columns:
        df = df[(df['teff'] > 3000) & (df['teff'] < 15000) & (df['teff'].notna())]

    # 过滤表面重力异常值
    if 'logg' in df.columns:
        df = df[(df['logg'] > 0) & (df['logg'] < 6) & (df['logg'].notna())]

    # 过滤金属丰度异常值
    if 'feh' in df.columns:
        df = df[(df['feh'] > -5) & (df['feh'] < 1) & (df['feh'].notna())]

    # 过滤信噪比
    if 'snr' in df.columns:
        df = df[df['snr'] > 5]

    print(f"过滤后: {len(df)} / {initial_count} 条")

    # 保存为CSV
    print(f"\n保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    return df


def generate_sample_report(df, output_path):
    """
    生成数据质量报告
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

    if 'alpha_m' in df.columns:
        print(f"  α元素丰度: {df['alpha_m'].min():.2f} - {df['alpha_m'].max():.2f}")
        print(f"  平均α丰度: {df['alpha_m'].mean():.2f}")

    if 'snr' in df.columns:
        print(f"  信噪比: {df['snr'].min():.1f} - {df['snr'].max():.1f}")
        print(f"  平均SNR: {df['snr'].mean():.1f}")

    print(f"\n元素丰度统计:")
    abundance_cols = ['c_fe', 'n_fe', 'o_fe', 'mg_fe', 'si_fe', 's_fe', 'ca_fe', 'ni_fe']
    for col in abundance_cols:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"  {col}: {valid.min():.2f} - {valid.max():.2f} (mean: {valid.mean():.2f})")

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
    parser.add_argument('--limit', '-l', type=int,
                        help="限制处理的记录数")

    args = parser.parse_args()

    data, columns = read_apogee_fits(args.input)

    if data is None:
        print("读取FITS文件失败")
        return

    column_mapping = get_relevant_columns(columns)

    if not column_mapping:
        print("未找到相关列")
        return

    df = convert_to_csv(data, column_mapping, args.output, args.limit)

    generate_sample_report(df, args.output)

    print(f"\n完成! 数据已保存到: {args.output}")


if __name__ == "__main__":
    main()
