#!/usr/bin/env python3
"""
自动导入示例数据并进行测试
将所有结果保存到数据库
"""

import sys
import time
import requests
import json
from pathlib import Path

# 配置
API_URL = "http://localhost:8000"
DB_DIR = Path(__file__).parent / "DB"
DATASETS = [
    ("star_data_small.csv", "小型数据集 (500 样本)"),
    ("star_data_medium.csv", "中型数据集 (1000 样本)"),
    ("star_data_large.csv", "大型数据集 (2000 样本)"),
]

def check_api():
    """检查 API 是否可用"""
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

def upload_and_test(csv_file, description):
    """上传数据集并进行测试"""
    print(f"\n{'='*70}")
    print(f"📊 测试: {description}")
    print(f"{'='*70}")
    
    csv_path = DB_DIR / csv_file
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return None
    
    print(f"📁 文件: {csv_file}")
    print(f"📏 大小: {csv_path.stat().st_size / 1024:.1f} KB")
    
    # 上传文件并执行分析
    print("\n1️⃣  上传文件并执行分析...")
    try:
        # 构建正确的 payload
        payload = {
            "datasetName": csv_file,
            "targetColumn": "class",
            "featureColumns": ["u_mag", "g_mag", "r_mag", "i_mag", "z_mag", "redshift", "petroR50_u", "petroR50_r"],
            "testSize": 0.2,
            "randomState": 42,
            "modelType": "gaussian_nb",
            "gnbParams": {
                "varSmoothing": None
            }
        }
        
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data = {'payload': json.dumps(payload)}
            
            print(f"   发送 payload: {json.dumps(payload, indent=2)}")
            
            resp = requests.post(
                f"{API_URL}/api/runs",
                files=files,
                data=data,
                timeout=60
            )
        
        if resp.status_code != 200:
            print(f"❌ 分析失败: {resp.status_code}")
            print(f"   {resp.text}")
            return None
        
        result = resp.json()
        print(f"✅ 分析成功")
        
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 显示结果
    print("\n2️⃣  分析结果:")
    try:
        metrics = result.get('metrics', {})
        print(f"   准确率 (Accuracy): {metrics.get('accuracy', 0):.4f}")
        print(f"   精确率 (Precision): {metrics.get('precision', 0):.4f}")
        print(f"   召回率 (Recall): {metrics.get('recall', 0):.4f}")
        print(f"   F1 分数: {metrics.get('f1', 0):.4f}")
        
        labels = result.get('labels', [])
        print(f"   分类数: {len(labels)}")
        print(f"   分类: {', '.join(labels)}")
        
        return result
    except Exception as e:
        print(f"❌ 解析结果出错: {e}")
        return None

def main():
    print("=" * 70)
    print("🚀 StarClassify-Vis 自动测试系统")
    print("=" * 70)
    
    # 检查 API
    print("\n🔍 检查 API 连接...")
    if not check_api():
        print("❌ API 不可用")
        print("💡 请先运行: python3 start_local.py")
        sys.exit(1)
    print("✅ API 连接成功")
    
    # 检查数据集
    print("\n🔍 检查数据集...")
    if not DB_DIR.exists():
        print(f"❌ 数据集目录不存在: {DB_DIR}")
        sys.exit(1)
    print(f"✅ 数据集目录: {DB_DIR}")
    
    # 执行测试
    results = []
    for csv_file, description in DATASETS:
        result = upload_and_test(csv_file, description)
        if result:
            results.append({
                "dataset": csv_file,
                "description": description,
                "result": result,
                "timestamp": time.time()
            })
            time.sleep(2)  # 避免请求过快
    
    # 总结
    print(f"\n{'='*70}")
    print("📊 测试总结")
    print(f"{'='*70}")
    print(f"✅ 成功: {len(results)}/{len(DATASETS)}")
    
    if results:
        print("\n📈 结果对比:")
        print(f"{'数据集':<25} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
        print("-" * 65)
        for r in results:
            metrics = r['result'].get('metrics', {})
            print(f"{r['dataset']:<25} {metrics.get('accuracy', 0):<10.4f} {metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<10.4f} {metrics.get('f1', 0):<10.4f}")
    
    print(f"\n{'='*70}")
    print("✨ 测试完成！")
    print("💡 访问 http://localhost:5173/runs 查看结果记录")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
