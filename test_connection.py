#!/usr/bin/env python3
"""
数据库连接测试脚本
验证前后端与数据库的连接
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_database_connection():
    """测试数据库连接"""
    print("🔍 测试数据库连接...")
    
    try:
        from sqlalchemy import create_engine, text
        from api.settings import get_database_url
        
        db_url = get_database_url()
        print(f"   数据库 URL: {db_url.replace('postgres:postgres', '***:***')}")
        
        engine = create_engine(db_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"✅ 数据库连接成功")
            print(f"   PostgreSQL 版本: {version.split(',')[0]}")
            
            # 检查表是否存在
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'runs'
                )
            """))
            table_exists = result.scalar()
            
            if table_exists:
                print(f"✅ runs 表已存在")
                
                # 获取表统计信息
                result = conn.execute(text("SELECT COUNT(*) FROM runs;"))
                count = result.scalar()
                print(f"   当前记录数: {count}")
            else:
                print(f"⚠️  runs 表不存在，将在启动时自动创建")
        
        return True
    
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


def test_api_connection():
    """测试 API 连接"""
    print("\n🔍 测试 API 连接...")
    
    try:
        import requests
        
        resp = requests.get("http://localhost:8000/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"✅ API 服务连接成功")
            print(f"   响应: {resp.json()}")
            return True
        else:
            print(f"❌ API 返回错误状态码: {resp.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 API (http://localhost:8000)")
        print(f"   请确保 API 服务已启动")
        return False
    except Exception as e:
        print(f"❌ API 连接测试失败: {e}")
        return False


def test_web_connection():
    """测试前端连接"""
    print("\n🔍 测试前端连接...")
    
    try:
        import requests
        
        resp = requests.get("http://localhost:8080", timeout=5)
        if resp.status_code == 200:
            print(f"✅ 前端服务连接成功")
            return True
        else:
            print(f"❌ 前端返回错误状态码: {resp.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到前端 (http://localhost:8080)")
        print(f"   请确保前端服务已启动")
        return False
    except Exception as e:
        print(f"❌ 前端连接测试失败: {e}")
        return False


def main():
    print("=" * 60)
    print("StarClassify-Vis 连接测试")
    print("=" * 60)
    
    # 测试数据库
    db_ok = test_database_connection()
    
    # 测试 API
    api_ok = test_api_connection()
    
    # 测试前端
    web_ok = test_web_connection()
    
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"数据库: {'✅ 正常' if db_ok else '❌ 异常'}")
    print(f"API:   {'✅ 正常' if api_ok else '❌ 异常'}")
    print(f"前端:  {'✅ 正常' if web_ok else '❌ 异常'}")
    print("=" * 60)
    
    if db_ok and api_ok and web_ok:
        print("\n✨ 所有连接正常！系统已就绪。")
        return 0
    else:
        print("\n⚠️  部分连接异常，请检查服务状态。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
