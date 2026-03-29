#!/usr/bin/env python3
"""
生成高质量的恒星分类示例数据集
用于 StarClassify-Vis 系统测试
"""

import csv
import random
import math
from pathlib import Path


def generate_star_data(num_samples=1000):
    """
    生成恒星分类数据集
    
    特征:
    - u_mag: 紫外波段星等
    - g_mag: 绿光波段星等
    - r_mag: 红光波段星等
    - i_mag: 近红外波段星等
    - z_mag: 远红外波段星等
    - redshift: 红移
    - petroR50_u: 星体半径 (紫外)
    - petroR50_r: 星体半径 (红光)
    
    分类:
    - STAR: 恒星
    - GALAXY: 星系
    - QSO: 类星体
    """
    
    data = []
    
    # 恒星特征 (约 40%)
    num_stars = int(num_samples * 0.4)
    for _ in range(num_stars):
        u_mag = random.gauss(17.5, 1.5)
        g_mag = u_mag + random.gauss(0.3, 0.2)
        r_mag = g_mag + random.gauss(0.2, 0.15)
        i_mag = r_mag + random.gauss(0.15, 0.1)
        z_mag = i_mag + random.gauss(0.1, 0.08)
        
        data.append({
            'u_mag': round(u_mag, 3),
            'g_mag': round(g_mag, 3),
            'r_mag': round(r_mag, 3),
            'i_mag': round(i_mag, 3),
            'z_mag': round(z_mag, 3),
            'redshift': round(random.gauss(0.001, 0.0005), 6),
            'petroR50_u': round(random.gauss(0.5, 0.1), 3),
            'petroR50_r': round(random.gauss(0.5, 0.1), 3),
            'class': 'STAR'
        })
    
    # 星系特征 (约 45%)
    num_galaxies = int(num_samples * 0.45)
    for _ in range(num_galaxies):
        u_mag = random.gauss(19.5, 2.0)
        g_mag = u_mag + random.gauss(0.5, 0.3)
        r_mag = g_mag + random.gauss(0.4, 0.25)
        i_mag = r_mag + random.gauss(0.3, 0.2)
        z_mag = i_mag + random.gauss(0.2, 0.15)
        
        data.append({
            'u_mag': round(u_mag, 3),
            'g_mag': round(g_mag, 3),
            'r_mag': round(r_mag, 3),
            'i_mag': round(i_mag, 3),
            'z_mag': round(z_mag, 3),
            'redshift': round(random.gauss(0.3, 0.2), 6),
            'petroR50_u': round(random.gauss(2.5, 0.8), 3),
            'petroR50_r': round(random.gauss(2.5, 0.8), 3),
            'class': 'GALAXY'
        })
    
    # 类星体特征 (约 15%)
    num_qsos = num_samples - num_stars - num_galaxies
    for _ in range(num_qsos):
        u_mag = random.gauss(18.5, 1.8)
        g_mag = u_mag + random.gauss(0.2, 0.15)
        r_mag = g_mag + random.gauss(0.15, 0.1)
        i_mag = r_mag + random.gauss(0.1, 0.08)
        z_mag = i_mag + random.gauss(0.08, 0.05)
        
        data.append({
            'u_mag': round(u_mag, 3),
            'g_mag': round(g_mag, 3),
            'r_mag': round(r_mag, 3),
            'i_mag': round(i_mag, 3),
            'z_mag': round(z_mag, 3),
            'redshift': round(random.gauss(1.5, 0.8), 6),
            'petroR50_u': round(random.gauss(1.0, 0.3), 3),
            'petroR50_r': round(random.gauss(1.0, 0.3), 3),
            'class': 'QSO'
        })
    
    # 打乱顺序
    random.shuffle(data)
    
    return data


def save_dataset(data, filename):
    """保存数据集为 CSV 文件"""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag',
            'redshift', 'petroR50_u', 'petroR50_r', 'class'
        ])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ 数据集已保存: {path}")
    print(f"   文件大小: {path.stat().st_size / 1024:.1f} KB")
    print(f"   样本数: {len(data)}")
    
    # 统计分类
    classes = {}
    for row in data:
        cls = row['class']
        classes[cls] = classes.get(cls, 0) + 1
    
    print(f"   分类统计:")
    for cls, count in sorted(classes.items()):
        print(f"     - {cls}: {count} ({count/len(data)*100:.1f}%)")


def main():
    print("=" * 70)
    print("🌟 恒星分类数据集生成器")
    print("=" * 70)
    print()
    
    # 生成三个不同大小的数据集
    datasets = [
        ("DB/star_data_small.csv", 500, "小型数据集"),
        ("DB/star_data_medium.csv", 1000, "中型数据集"),
        ("DB/star_data_large.csv", 2000, "大型数据集"),
    ]
    
    for filename, num_samples, description in datasets:
        print(f"📊 生成{description} ({num_samples} 样本)...")
        data = generate_star_data(num_samples)
        save_dataset(data, filename)
        print()
    
    print("=" * 70)
    print("✨ 所有数据集生成完成！")
    print("=" * 70)
    print()
    print("📍 数据集位置:")
    print("   - DB/star_data_small.csv (500 样本)")
    print("   - DB/star_data_medium.csv (1000 样本)")
    print("   - DB/star_data_large.csv (2000 样本)")
    print()
    print("💡 使用方法:")
    print("   1. 在前端上传任意一个 CSV 文件")
    print("   2. 选择 'class' 作为目标列")
    print("   3. 选择其他列作为特征")
    print("   4. 点击'开始分析'进行训练")
    print()


if __name__ == "__main__":
    main()
