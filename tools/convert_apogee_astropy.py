#!/usr/bin/env python3
"""
将APOGEE FITS文件转换为CSV格式（使用astropy Table）
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table
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

            # 使用astropy Table读取
            table = Table.read(hdul[1])
            print(f"Table创建成功，记录数: {len(table)}")

            return table

    except Exception as e:
        print(f"读取失败: {e}")
        return None


def convert_to_csv(table, output_path, max_rows=100000):
    """
    转换为CSV格式
    """
    print(f"\n转换数据...")

    total_rows = len(table)
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
        if col in table.colnames:
            available_columns.append(col)

    print(f"可用列: {len(available_columns)}")

    # 重命名映射
    column_mapping = {
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
        'LOCATION_ID': 'location_id'
    }

    # 重命名列
    table_renamed = table.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in table_renamed.colnames:
            table_renamed.rename_column(old_name, new_name)

    # 过滤数据
    print(f"\n过滤数据...")
    initial_count = len(table_renamed)

    # 应用过滤条件
    if 'teff' in table_renamed.colnames:
        table_renamed = table_renamed[(table_renamed['teff'] > 3000) & (table_renamed['teff'] < 15000)]
    
    if 'logg' in table_renamed.colnames:
        table_renamed = table_renamed[(table_renamed['logg'] > 0) & (table_renamed['logg'] < 6)]
    
    if 'feh' in table_renamed.colnames:
        table_renamed = table_renamed[(table_renamed['feh'] > -5) & (table_renamed['feh'] < 1)]
    
    if 'snr' in table_renamed.colnames:
        table_renamed = table_renamed[table_renamed['snr'] > 5]

    print(f"过滤后: {len(table_renamed)} / {initial_count} 条")

    # 限制行数
    if max_rows and len(table_renamed) > max_rows:
        table_renamed = table_renamed[:max_rows]

    # 保存为CSV
    print(f"\n保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 使用astropy Table的write方法
    table_renamed.write(output_path, format='ascii.csv', overwrite=True)

    print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    return table_renamed


def generate_report(table, output_path):
    """
    生成报告
    """
    print(f"\n{'='*60}")
    print(f"APOGEE DR17 数据转换报告")
    print(f"{'='*60}")

    print(f"\n输出文件: {output_path}")
    print(f"总记录数: {len(table):,}")

    print(f"\n恒星参数统计:")
    if 'teff' in table.colnames:
        print(f"  有效温度: {table['teff'].min():.0f} - {table['teff'].max():.0f} K")
        print(f"  平均温度: {np.mean(table['teff']):.0f} K")

    if 'logg' in table.colnames:
        print(f"  表面重力: {table['logg'].min():.2f} - {table['logg'].max():.2f}")
        print(f"  平均重力: {np.mean(table['logg']):.2f}")

    if 'feh' in table.colnames:
        print(f"  金属丰度: {table['feh'].min():.2f} - {table['feh'].max():.2f}")
        print(f"  平均丰度: {np.mean(table['feh']):.2f}")

    if 'snr' in table.colnames:
        print(f"  信噪比: {table['snr'].min():.1f} - {table['snr'].max():.1f}")
        print(f"  平均SNR: {np.mean(table['snr']):.1f}")

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

    table = read_apogee_fits(args.input)

    if table is None:
        print("读取FITS文件失败")
        return

    table_filtered = convert_to_csv(table, args.output, args.limit)

    generate_report(table_filtered, args.output)

    print(f"\n完成! 数据已保存到: {args.output}")


if __name__ == "__main__":
    main()
