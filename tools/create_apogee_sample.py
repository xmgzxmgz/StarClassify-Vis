#!/usr/bin/env python3
"""
创建APOGEE分类数据的小样本用于测试
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def create_sample(input_path, output_path, sample_size=50000):
    """
    创建小样本文件
    """
    print(f"读取完整文件: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df):,} 条")

    # 按类别分层抽样
    classes = df['class'].unique()
    samples_per_class = max(1000, sample_size // len(classes))

    sampled_dfs = []
    for cls in classes:
        class_df = df[df['class'] == cls]
        n_sample = min(samples_per_class, len(class_df))
        sampled_dfs.append(class_df.sample(n=n_sample, random_state=42))

    df_sample = pd.concat(sampled_dfs, ignore_index=True)
    print(f"抽样后: {len(df_sample):,} 条")

    # 保存
    df_sample.to_csv(output_path, index=False)
    print(f"保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    print("\n样本类别分布:")
    for cls, count in df_sample['class'].value_counts().items():
        print(f"  {cls}: {count:,}")

    return df_sample


def main():
    input_file = os.path.join(OUTPUT_DIR, 'apogee_dr17_with_classes.csv')
    output_file = os.path.join(OUTPUT_DIR, 'apogee_sample_classified.csv')

    create_sample(input_file, output_file, sample_size=50000)


if __name__ == "__main__":
    main()
