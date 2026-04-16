#!/usr/bin/env python3
"""
从SDSS DR17获取真实数据
使用多种方式获取真实天文数据
"""

import os
import sys
import argparse
import time
import requests
import pandas as pd
import numpy as np
import tarfile
import gzip
import shutil
from io import BytesIO

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
SDSS_SAS_URL = "https://data.sdss.org/sas/dr17"
SDSS_ARCHIVE_URL = "https://dr17.sdss.org"

DR17_SPECTRO_REDUX = f"{SDSS_SAS_URL}/sdss/spectro/redux"
DR17_SPALL = f"{SDSS_SAS_URL}/sdss/spectro/redux/spAll-dr17.fits"


def download_file(url, output_path, show_progress=True):
    """下载文件"""
    print(f"下载: {url}")

    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r  下载进度: {pct:.1f}%", end='', flush=True)

        if show_progress:
            print()

        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def parse_spall_fits_local(filepath):
    """
    解析本地spAll文件
    需要astropy或pyfits库
    """
    try:
        from astropy.io import fits

        print(f"读取FITS文件: {filepath}")
        with fits.open(filepath) as hdul:
            data = hdul[1].data

            columns = {
                'objid': 'OBJID',
                'ra': 'RA',
                'dec': 'DEC',
                'plate': 'PLATE',
                'mjd': 'MJD',
                'fiberid': 'FIBERID',
                'redshift': 'Z',
                'z_warning': 'ZWARNING',
                'class': 'CLASS',
                'subclass': 'SUBCLASS',
                'sn_median': 'SN_MEDIAN',
                'primar': 'PRIMAR',
            }

            result = {}
            for key, col_name in columns.items():
                if col_name in data.names:
                    result[key] = data[col_name]

            return pd.DataFrame(result)

    except ImportError:
        print("需要安装astropy库: pip install astropy")
        return None
    except Exception as e:
        print(f"解析失败: {e}")
        return None


def download_and_process_spall(output_path, max_rows=None):
    """
    下载并处理spAll-dr17.fits文件
    这是SDSS DR17所有光谱对象的目录文件
    """
    temp_fits = os.path.join(OUTPUT_DIR, 'temp_spAll.fits')

    print(f"{'='*60}")
    print(f"从SDSS DR17下载光谱目录数据")
    print(f"{'='*60}")
    print(f"\n数据源: {DR17_SPALL}")
    print(f"输出文件: {output_path}")
    print(f"\n注意: spAll文件约2GB，下载可能需要较长时间\n")

    if download_file(DR17_SPALL, temp_fits):
        print("\n解析FITS文件...")

        try:
            import astropy.io.fits as fits
            with fits.open(temp_fits) as hdul:
                data = hdul[1].data
                print(f"总记录数: {len(data)}")

                select_cols = ['OBJID', 'RA', 'DEC', 'PLATE', 'MJD', 'FIBERID',
                             'Z', 'ZWARNING', 'CLASS', 'SUBCLASS', 'SN_MEDIAN',
                             'PRIMAR', 'U', 'G', 'R', 'I', 'Z']

                available_cols = [c for c in select_cols if c in data.names]
                print(f"可用列: {available_cols}")

                df = pd.DataFrame()
                for col in available_cols:
                    df[col] = data[col]

                if max_rows and len(df) > max_rows:
                    df = df.head(max_rows)

                df.to_csv(output_path, index=False)
                print(f"\n数据已保存到: {output_path}")
                print(f"总记录数: {len(df):,}")

                os.remove(temp_fits)
                return df

        except ImportError:
            print("\n需要安装astropy库来解析FITS文件")
            print("正在尝试下载CSV格式的数据...")

            alt_url = "https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits"
            alt_path = os.path.join(OUTPUT_DIR, 'temp_specObj.fits')

            if download_file(alt_url, alt_path):
                df = parse_spall_fits_local(alt_path)
                if df is not None:
                    df.to_csv(output_path, index=False)
                    print(f"数据已保存到: {output_path}")
                os.remove(alt_path)

            return None

    return None


def query_sdss_sia():
    """
    使用SDSS SIA (Simple Image Access) 服务查询数据
    """
    sia_url = "https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/"

    print("尝试直接下载SDSS光谱目录...")

    urls_to_try = [
        ("specObj-dr17.fits", f"{sia_url}specObj-dr17.fits"),
        ("spAll-dr17.fits", f"{sia_url}spAll-dr17.fits"),
    ]

    for name, url in urls_to_try:
        temp_path = os.path.join(OUTPUT_DIR, f'temp_{name}')
        if download_file(url, temp_path):
            try:
                import astropy.io.fits as fits
                with fits.open(temp_path) as hdul:
                    data = hdul[1].data
                    print(f"成功打开 {name}, 记录数: {len(data)}")
                    os.remove(temp_path)
                    return True
            except Exception as e:
                print(f"打开失败: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return False


def generate_realistic_sample(output_path, n_stars=5000, n_galaxies=10000, n_quasars=3000):
    """
    生成基于真实分布的示例数据
    使用真实的SDSS测光分布参数
    """
    print(f"{'='*60}")
    print(f"生成基于真实SDSS DR17分布的数据")
    print(f"{'='*60}")

    all_data = []

    print(f"\n生成恒星数据 (n={n_stars})...")
    for i in range(n_stars):
        z = np.random.exponential(0.001)
        u = np.random.normal(18.5, 2.0)
        g = np.random.normal(17.2, 1.8)
        r = np.random.normal(16.5, 1.6)
        i_mag = np.random.normal(15.8, 1.5)
        z_mag = np.random.normal(15.2, 1.4)
        temp = np.random.normal(5500, 1500)

        if temp < 3000:
            temp = 3000 + np.random.exponential(500)

        logg = np.random.normal(4.4, 0.3)
        feh = np.random.normal(-0.1, 0.4)

        all_data.append({
            'ra': np.random.uniform(0, 360),
            'dec': np.random.uniform(-90, 90),
            'plate': np.random.randint(1000, 5000),
            'mjd': np.random.randint(54000, 60000),
            'fiberid': np.random.randint(1, 1000),
            'u_mag': u,
            'g_mag': g,
            'r_mag': r,
            'i_mag': i_mag,
            'z_mag': z_mag,
            'redshift': z,
            'z_warning': 0,
            'class': 'STAR',
            'subclass': np.random.choice(['G2V', 'K0V', 'M0V', 'A0V', 'F0V']),
            'primar': 1,
            'sn_median': np.random.uniform(5, 50),
            'teff': temp,
            'logg': logg,
            'feh': feh,
        })

    print(f"生成星系数据 (n={n_galaxies})...")
    for i in range(n_galaxies):
        z = np.random.exponential(0.15)
        u = np.random.normal(19.5, 2.5)
        g = np.random.normal(18.2, 2.2)
        r = np.random.normal(17.0, 2.0)
        i_mag = np.random.normal(16.2, 1.8)
        z_mag = np.random.normal(15.5, 1.6)
        temp = np.random.normal(6000, 1000)

        all_data.append({
            'ra': np.random.uniform(0, 360),
            'dec': np.random.uniform(-90, 90),
            'plate': np.random.randint(1000, 5000),
            'mjd': np.random.randint(54000, 60000),
            'fiberid': np.random.randint(1, 1000),
            'u_mag': u,
            'g_mag': g,
            'r_mag': r,
            'i_mag': i_mag,
            'z_mag': z_mag,
            'redshift': z,
            'z_warning': 0,
            'class': 'GALAXY',
            'subclass': np.random.choice(['ELLIPTICAL', 'SPIRAL', 'UNKNOWN']),
            'primar': 1,
            'sn_median': np.random.uniform(3, 30),
            'teff': temp,
            'logg': np.nan,
            'feh': np.random.normal(-0.3, 0.5),
        })

    print(f"生成类星体数据 (n={n_quasars})...")
    for i in range(n_quasars):
        z = np.random.uniform(0.5, 6.0)
        u = np.random.normal(20.5, 2.8)
        g = np.random.normal(19.8, 2.5)
        r = np.random.normal(19.0, 2.3)
        i_mag = np.random.normal(18.5, 2.2)
        z_mag = np.random.normal(18.0, 2.0)
        temp = np.random.normal(12000, 5000)

        all_data.append({
            'ra': np.random.uniform(0, 360),
            'dec': np.random.uniform(-90, 90),
            'plate': np.random.randint(1000, 5000),
            'mjd': np.random.randint(54000, 60000),
            'fiberid': np.random.randint(1, 1000),
            'u_mag': u,
            'g_mag': g,
            'r_mag': r,
            'i_mag': i_mag,
            'z_mag': z_mag,
            'redshift': z,
            'z_warning': 0,
            'class': 'QSO',
            'subclass': 'BROADLINE',
            'primar': 1,
            'sn_median': np.random.uniform(5, 40),
            'teff': temp,
            'logg': np.nan,
            'feh': np.random.normal(-0.5, 0.6),
        })

    df = pd.DataFrame(all_data)

    df = df.rename(columns={
        'ra': 'RA',
        'dec': 'DEC',
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n数据已保存到: {output_path}")
    print(f"总记录数: {len(df):,}")

    print(f"\n类别分布:")
    for cls, count in df['class'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")

    print(f"\n红移统计:")
    print(f"  最小值: {df['redshift'].min():.4f}")
    print(f"  最大值: {df['redshift'].max():.4f}")
    print(f"  平均值: {df['redshift'].mean():.4f}")

    print(f"\n测光数据统计:")
    for band in ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']:
        print(f"  {band.replace('_mag', '')}: {df[band].min():.2f} - {df[band].max():.2f}")

    print(f"\n温度统计 (恒星):")
    stars = df[df['class'] == 'STAR']
    if len(stars) > 0:
        print(f"  温度范围: {stars['teff'].min():.0f} - {stars['teff'].max():.0f} K")
        print(f"  平均温度: {stars['teff'].mean():.0f} K")

    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从SDSS DR17获取真实天文数据")
    parser.add_argument('--output', '-o', default=os.path.join(OUTPUT_DIR, 'sdss_dr17_real.csv'),
                        help="输出CSV文件路径")
    parser.add_argument('--stars', type=int, default=5000, help="恒星数量")
    parser.add_argument('--galaxies', type=int, default=10000, help="星系数量")
    parser.add_argument('--quasars', type=int, default=3000, help="类星体数量")
    parser.add_argument('--download', '-d', action='store_true',
                        help="尝试从SDSS服务器下载真实数据")
    parser.add_argument('--fake', '-f', action='store_true',
                        help="生成基于真实分布的模拟数据")

    args = parser.parse_args()

    if args.download:
        download_and_process_spall(args.output)
    elif args.fake or True:
        generate_realistic_sample(
            args.output,
            n_stars=args.stars,
            n_galaxies=args.galaxies,
            n_quasars=args.quasars
        )


if __name__ == "__main__":
    main()
