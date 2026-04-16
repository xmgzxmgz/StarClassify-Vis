#!/usr/bin/env python3
"""
为APOGEE数据添加恒星类型分类标签
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def add_stellar_class_labels(input_path, output_path):
    """
    基于恒星参数添加分类标签
    """
    print(f"读取APOGEE数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df):,} 条")

    # 首先清理数据 - 移除第一行VESTA（不是恒星）
    df = df[df['apogee_id'] != 'VESTA'].copy()
    print(f"移除校准星后: {len(df):,} 条")

    # 基于温度、表面重力和金属丰度创建恒星类型分类
    def classify_star(row):
        teff = row['teff']
        logg = row['logg']
        feh = row['feh']

        if pd.isna(teff) or pd.isna(logg):
            return 'UNKNOWN'

        # 基于赫罗图分类
        if teff > 7000 and logg < 3:
            # 热巨星 / 蓝巨星
            return 'HOT_GIANT'
        elif teff > 6000 and logg > 3.5:
            # 热主序星（A/F型）
            return 'HOT_MAIN_SEQUENCE'
        elif teff > 5000 and teff <= 6000 and logg > 3.5:
            # 类太阳恒星（G型）
            return 'SOLAR_TYPE'
        elif teff > 3500 and teff <= 5000 and logg > 3.5:
            # 冷主序星（K/M型矮星）
            return 'COOL_DWARF'
        elif teff > 3500 and teff <= 5000 and logg <= 3.5:
            # 红巨星
            return 'RED_GIANT'
        elif teff <= 3500:
            # 极冷恒星（M型矮星/褐矮星候选）
            return 'VERY_COOL'
        else:
            return 'OTHER'

    print("\n正在为恒星分类...")
    df['class'] = df.apply(classify_star, axis=1)

    # 显示分类结果
    print("\n恒星类型分布:")
    class_counts = df['class'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")

    # 显示各类型的平均参数
    print("\n各类型平均参数:")
    types = df['class'].unique()
    for star_type in sorted(types):
        subset = df[df['class'] == star_type]
        print(f"\n{star_type} (n={len(subset):,}):")
        print(f"  平均温度: {subset['teff'].mean():.0f} K")
        print(f"  平均logg: {subset['logg'].mean():.2f}")
        print(f"  平均[Fe/H]: {subset['feh'].mean():.2f}")

    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"\n处理完成!")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    return df


def main():
    input_file = os.path.join(OUTPUT_DIR, 'apogee_dr17.csv')
    output_file = os.path.join(OUTPUT_DIR, 'apogee_dr17_with_classes.csv')

    add_stellar_class_labels(input_file, output_file)


if __name__ == "__main__":
    main()
