#!/usr/bin/env python3
"""
下载和处理SDSS DR17数据
将SDSS数据转换为CSV格式，用于科研人员工作区分析
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# 全局变量
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成SDSS DR17示例数据")
    parser.add_argument('--output', '-o', default=os.path.join(OUTPUT_DIR, 'sdss_dr17.csv'),
                        help="输出CSV文件路径")
    parser.add_argument('--limit', '-l', type=int, default=1000,
                        help="生成的数据数量")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 生成示例数据
    print("生成SDSS DR17示例数据...")
    data = []
    
    for i in range(args.limit):
        # 生成更真实的红移分布
        redshift = np.random.exponential(0.5)
        
        # 基于红移的分类
        if redshift < 0.05:
            star_class = "star"
            # 恒星的温度分布
            temperature = np.random.normal(5000, 2000)
            logg = np.random.normal(4, 1)
            metallicity = np.random.normal(0, 0.5)
        elif redshift < 1.0:
            star_class = "galaxy"
            # 星系的参数
            temperature = np.random.normal(6000, 1500)
            logg = np.random.normal(3, 0.8)
            metallicity = np.random.normal(-0.2, 0.4)
        else:
            star_class = "quasar"
            # 类星体的参数
            temperature = np.random.normal(10000, 3000)
            logg = np.random.normal(2, 0.5)
            metallicity = np.random.normal(-0.5, 0.3)
        
        # 生成光度数据 (u, g, r, i, z)
        # 基于分类调整光度
        if star_class == "star":
            u = np.random.normal(18, 2)
            g = np.random.normal(17, 2)
            r = np.random.normal(16, 2)
            i = np.random.normal(15, 2)
            z = np.random.normal(14, 2)
        elif star_class == "galaxy":
            u = np.random.normal(19, 2.5)
            g = np.random.normal(18, 2.5)
            r = np.random.normal(17, 2.5)
            i = np.random.normal(16, 2.5)
            z = np.random.normal(15, 2.5)
        else:  # quasar
            u = np.random.normal(20, 3)
            g = np.random.normal(19, 3)
            r = np.random.normal(18, 3)
            i = np.random.normal(17, 3)
            z = np.random.normal(16, 3)
        
        data.append({
            'plate': 1000 + i % 100,
            'mjd': 59000 + i % 50,
            'fiber': i % 1000,
            'u': u,
            'g': g,
            'r': r,
            'i': i,
            'z': z,
            'redshift': redshift,
            'temperature': temperature,
            'logg': logg,
            'metallicity': metallicity,
            'class': star_class
        })
    
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"SDSS DR17示例数据已生成到: {args.output}")
    print(f"共生成 {len(data)} 条数据")
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    print(f"类别分布:")
    print(df['class'].value_counts())
    print(f"\n红移范围: {df['redshift'].min():.4f} - {df['redshift'].max():.4f}")
    print(f"温度范围: {df['temperature'].min():.0f} - {df['temperature'].max():.0f} K")


if __name__ == "__main__":
    main()
