#!/usr/bin/env python3
"""
StarClassify-Vis 启动脚本
支持本地开发和 Docker 环境
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, cwd=None, env=None):
    """执行命令并返回结果"""
    print(f"▶ 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode == 0


def main():
    project_root = Path(__file__).parent
    
    # 检查环境变量
    mode = os.environ.get("START_MODE", "docker").lower()
    
    if mode == "docker":
        print("🐳 启动 Docker 环境...")
        print("=" * 60)
        
        # 检查 docker-compose
        if not run_command(["docker-compose", "--version"]):
            print("❌ 错误: 未找到 docker-compose，请先安装 Docker Desktop")
            sys.exit(1)
        
        # 启动所有服务
        if not run_command(["docker-compose", "up", "-d"], cwd=project_root):
            print("❌ Docker 启动失败")
            sys.exit(1)
        
        print("\n⏳ 等待服务启动...")
        time.sleep(5)
        
        # 检查服务健康状态
        print("\n🔍 检查服务状态...")
        max_retries = 30
        for i in range(max_retries):
            try:
                import requests
                resp = requests.get("http://localhost:8000/api/health", timeout=2)
                if resp.status_code == 200:
                    print("✅ API 服务已就绪")
                    break
            except Exception:
                pass
            
            if i < max_retries - 1:
                print(f"   等待中... ({i+1}/{max_retries})")
                time.sleep(1)
        
        print("\n" + "=" * 60)
        print("✨ 所有服务已启动！")
        print("=" * 60)
        print("📍 前端: http://localhost:8080")
        print("📍 API:  http://localhost:8000")
        print("📍 数据库: localhost:5433 (postgres/postgres)")
        print("\n💡 查看日志: docker-compose logs -f")
        print("💡 停止服务: docker-compose down")
        print("=" * 60)
        
    elif mode == "local":
        print("🚀 启动本地开发环境...")
        print("=" * 60)
        
        # 检查 Python 版本
        if sys.version_info < (3, 9):
            print("❌ 错误: 需要 Python 3.9+")
            sys.exit(1)
        
        # 检查 PostgreSQL
        print("\n🔍 检查 PostgreSQL...")
        try:
            import psycopg
            print("✅ psycopg 已安装")
        except ImportError:
            print("⚠️  需要安装依赖: pip install -r api/requirements.txt")
        
        # 启动后端
        print("\n🔧 启动后端 API...")
        api_env = os.environ.copy()
        api_env["DATABASE_URL"] = "postgresql+psycopg://postgres:postgres@localhost:5432/starvis"
        api_env["CORS_ORIGINS"] = "http://localhost:5173,http://localhost:5174"
        
        api_process = subprocess.Popen(
            ["python", "-m", "uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd=project_root,
            env=api_env
        )
        
        # 启动前端
        print("🎨 启动前端开发服务器...")
        web_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=project_root / "web"
        )
        
        print("\n" + "=" * 60)
        print("✨ 本地开发环境已启动！")
        print("=" * 60)
        print("📍 前端: http://localhost:5173")
        print("📍 API:  http://localhost:8000")
        print("=" * 60)
        print("\n按 Ctrl+C 停止所有服务...")
        
        try:
            api_process.wait()
            web_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 停止服务...")
            api_process.terminate()
            web_process.terminate()
            api_process.wait()
            web_process.wait()
            print("✅ 已停止")
    
    else:
        print(f"❌ 未知的启动模式: {mode}")
        print("支持的模式: docker, local")
        sys.exit(1)


if __name__ == "__main__":
    main()
