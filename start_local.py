#!/usr/bin/env python3
"""
StarClassify-Vis 本地开发启动脚本
启动本地前端和后端，使用 Docker 中的 PostgreSQL 数据库
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path


def run_command(cmd, cwd=None, env=None, description=""):
    """执行命令并返回结果"""
    if description:
        print(f"▶ {description}")
    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode == 0


def main():
    project_root = Path(__file__).parent
    
    print("=" * 70)
    print("🚀 StarClassify-Vis 本地开发启动脚本")
    print("=" * 70)
    print("\n📋 启动配置:")
    print("  • 前端: React + Vite (http://localhost:5173)")
    print("  • 后端: FastAPI (http://localhost:8000)")
    print("  • 数据库: PostgreSQL in Docker (localhost:5433)")
    print("\n")
    
    # 清理旧服务
    print("🧹 清理旧服务...")
    print("=" * 70)
    
    try:
        print("停止 Docker 容器...")
        subprocess.run(
            ["docker-compose", "down"],
            cwd=project_root,
            capture_output=True,
            timeout=30
        )
        print("✅ Docker 容器已停止")
    except subprocess.TimeoutExpired:
        print("⚠️  Docker 停止超时，继续...")
    except Exception as e:
        print(f"⚠️  Docker 停止出错: {e}")
    
    # 等待端口释放
    print("⏳ 等待端口释放...")
    time.sleep(3)
    
    print("\n")
    
    # 检查 Python 版本
    print("🔍 检查 Python 版本...")
    if sys.version_info < (3, 9):
        print("❌ 错误: 需要 Python 3.9+")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # 检查 Node.js
    print("\n🔍 检查 Node.js...")
    if not run_command(["node", "--version"], description=""):
        print("❌ 错误: 未找到 Node.js，请先安装")
        sys.exit(1)
    node_version = subprocess.check_output(["node", "--version"]).decode().strip()
    print(f"✅ {node_version}")
    
    # 检查 npm
    print("\n🔍 检查 npm...")
    if not run_command(["npm", "--version"], description=""):
        print("❌ 错误: 未找到 npm")
        sys.exit(1)
    npm_version = subprocess.check_output(["npm", "--version"]).decode().strip()
    print(f"✅ npm {npm_version}")
    
    # 检查 Docker
    print("\n🔍 检查 Docker...")
    if not run_command(["docker", "--version"], description=""):
        print("❌ 错误: 未找到 Docker，请先安装 Docker Desktop")
        sys.exit(1)
    docker_version = subprocess.check_output(["docker", "--version"]).decode().strip()
    print(f"✅ {docker_version}")
    
    # 检查 Docker Compose
    print("\n🔍 检查 Docker Compose...")
    if not run_command(["docker-compose", "--version"], description=""):
        print("❌ 错误: 未找到 docker-compose")
        sys.exit(1)
    compose_version = subprocess.check_output(["docker-compose", "--version"]).decode().strip()
    print(f"✅ {compose_version}")
    
    # 启动 Docker 数据库
    print("\n" + "=" * 70)
    print("📦 启动 Docker 数据库...")
    print("=" * 70)
    
    # 启动数据库，带重试
    max_db_retries = 3
    db_started = False
    
    for attempt in range(max_db_retries):
        print(f"\n📦 启动 Docker 数据库 (尝试 {attempt + 1}/{max_db_retries})...")
        print("=" * 70)
        
        if run_command(
            ["docker-compose", "up", "-d", "db"],
            cwd=project_root,
            description="启动 PostgreSQL 容器"
        ):
            db_started = True
            break
        else:
            if attempt < max_db_retries - 1:
                print(f"⚠️  启动失败，等待 5 秒后重试...")
                time.sleep(5)
    
    if not db_started:
        print("❌ Docker 数据库启动失败，已重试 3 次")
        sys.exit(1)
    
    print("⏳ 等待数据库就绪...")
    time.sleep(3)
    
    # 检查数据库健康状态，带更好的重试
    max_retries = 60
    db_ready = False
    for i in range(max_retries):
        try:
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "db", "pg_isready", "-U", "postgres"],
                cwd=project_root,
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✅ 数据库已就绪")
                db_ready = True
                break
        except Exception:
            pass
        
        if i < max_retries - 1:
            if i % 10 == 0:
                print(f"   等待中... ({i+1}/{max_retries})")
            time.sleep(1)
    
    if not db_ready:
        print("❌ 数据库启动超时")
        print("💡 尝试手动清理: docker-compose down -v")
        sys.exit(1)
    
    # 安装后端依赖
    print("\n" + "=" * 70)
    print("📚 安装后端依赖...")
    print("=" * 70)
    
    api_dir = project_root / "api"
    if not run_command(
        ["pip", "install", "-r", "requirements.txt"],
        cwd=api_dir,
        description="安装 Python 依赖"
    ):
        print("❌ 后端依赖安装失败")
        sys.exit(1)
    
    # 安装前端依赖，带重试
    print("\n" + "=" * 70)
    print("📚 安装前端依赖...")
    print("=" * 70)
    
    web_dir = project_root / "web"
    max_npm_retries = 3
    npm_installed = False
    
    for attempt in range(max_npm_retries):
        if attempt > 0:
            print(f"\n⚠️  npm 安装失败，清理缓存后重试 (尝试 {attempt + 1}/{max_npm_retries})...")
            # 清理 npm 缓存
            subprocess.run(["npm", "cache", "clean", "--force"], cwd=web_dir, capture_output=True)
            time.sleep(2)
        
        if run_command(
            ["npm", "install"],
            cwd=web_dir,
            description="安装 npm 依赖"
        ):
            npm_installed = True
            break
    
    if not npm_installed:
        print("❌ 前端依赖安装失败，已重试 3 次")
        print("💡 尝试手动安装: cd web && npm install")
        sys.exit(1)
    
    # 启动后端
    print("\n" + "=" * 70)
    print("🔧 启动后端 API...")
    print("=" * 70)
    
    api_env = os.environ.copy()
    api_env["DATABASE_URL"] = "postgresql+psycopg://postgres:postgres@localhost:5433/starvis"
    api_env["CORS_ORIGINS"] = "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173"
    api_env["PYTHONUNBUFFERED"] = "1"
    
    print("📍 API 地址: http://localhost:8000")
    print("📍 API 文档: http://localhost:8000/docs")
    print("\n启动 Uvicorn 服务器...\n")
    
    api_process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=project_root,
        env=api_env
    )
    
    # 等待后端启动
    time.sleep(3)
    
    # 启动前端
    print("\n" + "=" * 70)
    print("🎨 启动前端开发服务器...")
    print("=" * 70)
    
    print("📍 前端地址: http://localhost:5173")
    print("\n启动 Vite 开发服务器...\n")
    
    web_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=web_dir
    )
    
    # 等待前端启动
    time.sleep(3)
    
    print("\n" + "=" * 70)
    print("✨ 所有服务已启动！")
    print("=" * 70)
    print("\n📍 访问地址:")
    print("  • 前端:     http://localhost:5173")
    print("  • API:      http://localhost:8000")
    print("  • API 文档: http://localhost:8000/docs")
    print("  • 数据库:   localhost:5433 (postgres/postgres)")
    print("\n💡 快捷操作:")
    print("  • 查看数据库日志: docker-compose logs -f db")
    print("  • 停止数据库:     docker-compose down")
    print("  • 按 Ctrl+C 停止前端和后端")
    print("\n" + "=" * 70)
    
    # 等待服务完全启动
    print("\n🔍 等待服务完全启动...")
    time.sleep(5)
    
    # 自动测试连接
    print("\n🧪 测试服务连接...")
    print("=" * 70)
    
    test_results = {
        "数据库": False,
        "API": False,
        "前端": False
    }
    
    # 测试数据库
    try:
        result = subprocess.run(
            ["docker-compose", "exec", "-T", "db", "pg_isready", "-U", "postgres"],
            cwd=project_root,
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ 数据库连接成功")
            test_results["数据库"] = True
        else:
            print("❌ 数据库连接失败")
    except Exception as e:
        print(f"❌ 数据库测试出错: {e}")
    
    # 测试 API
    try:
        import requests
        resp = requests.get("http://localhost:8000/api/health", timeout=5)
        if resp.status_code == 200:
            print("✅ API 服务连接成功")
            test_results["API"] = True
        else:
            print(f"❌ API 返回错误状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ API 测试出错: {e}")
    
    # 测试前端
    try:
        import requests
        resp = requests.get("http://localhost:5173", timeout=5)
        if resp.status_code == 200:
            print("✅ 前端服务连接成功")
            test_results["前端"] = True
        else:
            print(f"❌ 前端返回错误状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ 前端测试出错: {e}")
    
    print("\n" + "=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    for service, status in test_results.items():
        status_str = "✅ 正常" if status else "❌ 异常"
        print(f"{service}: {status_str}")
    
    all_ok = all(test_results.values())
    if all_ok:
        print("\n✨ 所有服务正常！系统已就绪。")
    else:
        print("\n⚠️  部分服务异常，请检查日志。")
    
    print("=" * 70)
    
    # 处理信号
    def signal_handler(sig, frame):
        print("\n\n🛑 停止所有服务...")
        api_process.terminate()
        web_process.terminate()
        
        try:
            api_process.wait(timeout=5)
            web_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()
            web_process.kill()
        
        print("✅ 已停止前端和后端")
        print("💡 数据库仍在运行，使用 'docker-compose down' 停止")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 等待进程
    try:
        api_process.wait()
        web_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
