#!/usr/bin/env python3
"""
将APOGEE FITS文件转换为CSV格式（测试版本）
"""

import os
import argparse
from astropy.io import fits
from astropy.table import Table

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="将APOGEE FITS文件转换为CSV格式")
    parser.add_argument('--input', '-i',
                        default=os.path.join(OUTPUT_DIR, 'allStar-dr17-synspec_rev1.fits'),
                        help="输入FITS文件路径")
    parser.add_argument('--output', '-o',
                        default=os.path.join(OUTPUT_DIR, 'apogee_dr17_test.csv'),
                        help="输出CSV文件路径")
    parser.add_argument('--limit', '-l', type=int, default=1000,
                        help="限制处理的记录数")

    args = parser.parse_args()

    print(f"读取APOGEE FITS文件: {args.input}")
    print(f"文件大小: {os.path.getsize(args.input) / (1024**3):.2f} GB")

    try:
        # 直接使用Table.read读取整个文件
        print("正在读取数据...")
        table = Table.read(args.input, hdu=1)
        print(f"读取成功，总记录数: {len(table)}")

        # 限制行数
        if len(table) > args.limit:
            table = table[:args.limit]
            print(f"限制为 {args.limit} 条记录")

        # 选择需要的列
        columns_to_keep = [
            'APOGEE_ID', 'RA', 'DEC', 'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
            'FE_H', 'FE_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 'C_FE', 'N_FE', 'O_FE',
            'MG_FE', 'SI_FE', 'S_FE', 'CA_FE', 'NI_FE', 'VHELIO_AVG', 'SNR',
            'NVISITS', 'FIELD', 'LOCATION_ID'
        ]

        # 过滤列
        available_columns = [col for col in columns_to_keep if col in table.colnames]
        table = table[available_columns]
        print(f"保留 {len(available_columns)} 列")

        # 重命名列
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

        for old_name, new_name in column_mapping.items():
            if old_name in table.colnames:
                table.rename_column(old_name, new_name)

        # 保存为CSV
        print(f"保存到: {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        table.write(args.output, format='ascii.csv', overwrite=True)

        print(f"文件大小: {os.path.getsize(args.output) / (1024**2):.2f} MB")
        print(f"完成! 数据已保存到: {args.output}")

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
