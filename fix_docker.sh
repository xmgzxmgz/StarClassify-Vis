#!/bin/bash

echo "🔧 Docker 权限修复脚本"
echo "======================================================================"

# 1. 关闭 Docker
echo ""
echo "1️⃣  关闭 Docker Desktop..."
killall Docker 2>/dev/null || true
sleep 2

# 2. 清理 Docker 套接字
echo "2️⃣  清理 Docker 套接字..."
rm -f ~/.docker/run/docker.sock 2>/dev/null || true

# 3. 重启 Docker
echo "3️⃣  重新启动 Docker Desktop..."
open /Applications/Docker.app

# 4. 等待 Docker 启动
echo "4️⃣  等待 Docker 启动 (约 30 秒)..."
sleep 30

# 5. 验证 Docker
echo "5️⃣  验证 Docker..."
docker --version
docker ps

echo ""
echo "✅ Docker 修复完成！"
echo ""
echo "现在可以运行: python3 start_local.py"
