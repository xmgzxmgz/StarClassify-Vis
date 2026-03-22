#!/bin/bash

# StarClassify-Vis 本地开发启动脚本 (Bash 版本)
# 启动本地前端和后端，使用 Docker 中的 PostgreSQL 数据库

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "🚀 StarClassify-Vis 本地开发启动脚本"
echo "========================================================================"
echo ""
echo "📋 启动配置:"
echo "  • 前端: React + Vite (http://localhost:5173)"
echo "  • 后端: FastAPI (http://localhost:8000)"
echo "  • 数据库: PostgreSQL in Docker (localhost:5433)"
echo ""

# 检查 Python 版本
echo "🔍 检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION"

# 检查 Node.js
echo ""
echo "🔍 检查 Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js，请先安装"
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✅ $NODE_VERSION"

# 检查 npm
echo ""
echo "🔍 检查 npm..."
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: 未找到 npm"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo "✅ npm $NPM_VERSION"

# 检查 Docker
echo ""
echo "🔍 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker，请先安装 Docker Desktop"
    exit 1
fi
DOCKER_VERSION=$(docker --version)
echo "✅ $DOCKER_VERSION"

# 检查 Docker Compose
echo ""
echo "🔍 检查 Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未找到 docker-compose"
    exit 1
fi
COMPOSE_VERSION=$(docker-compose --version)
echo "✅ $COMPOSE_VERSION"

# 启动 Docker 数据库
echo ""
echo "========================================================================"
echo "📦 启动 Docker 数据库..."
echo "========================================================================"

cd "$PROJECT_ROOT"
docker-compose up -d db

echo "⏳ 等待数据库就绪..."
sleep 3

# 检查数据库健康状态
MAX_RETRIES=30
DB_READY=false

for ((i=1; i<=MAX_RETRIES; i++)); do
    if docker-compose exec -T db pg_isready -U postgres &> /dev/null; then
        echo "✅ 数据库已就绪"
        DB_READY=true
        break
    fi
    
    if [ $i -lt $MAX_RETRIES ]; then
        echo "   等待中... ($i/$MAX_RETRIES)"
        sleep 1
    fi
done

if [ "$DB_READY" = false ]; then
    echo "❌ 数据库启动超时"
    exit 1
fi

# 安装后端依赖
echo ""
echo "========================================================================"
echo "📚 安装后端依赖..."
echo "========================================================================"

cd "$PROJECT_ROOT/api"
pip install -r requirements.txt

# 安装前端依赖
echo ""
echo "========================================================================"
echo "📚 安装前端依赖..."
echo "========================================================================"

cd "$PROJECT_ROOT/web"
npm install

# 启动后端
echo ""
echo "========================================================================"
echo "🔧 启动后端 API..."
echo "========================================================================"

echo "📍 API 地址: http://localhost:8000"
echo "📍 API 文档: http://localhost:8000/docs"
echo ""
echo "启动 Uvicorn 服务器..."
echo ""

export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5433/starvis"
export CORS_ORIGINS="http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 3

# 启动前端
echo ""
echo "========================================================================"
echo "🎨 启动前端开发服务器..."
echo "========================================================================"

echo "📍 前端地址: http://localhost:5173"
echo ""
echo "启动 Vite 开发服务器..."
echo ""

cd "$PROJECT_ROOT/web"
npm run dev &
WEB_PID=$!

sleep 3

echo ""
echo "========================================================================"
echo "✨ 所有服务已启动！"
echo "========================================================================"
echo ""
echo "📍 访问地址:"
echo "  • 前端:     http://localhost:5173"
echo "  • API:      http://localhost:8000"
echo "  • API 文档: http://localhost:8000/docs"
echo "  • 数据库:   localhost:5433 (postgres/postgres)"
echo ""
echo "💡 快捷操作:"
echo "  • 查看数据库日志: docker-compose logs -f db"
echo "  • 停止数据库:     docker-compose down"
echo "  • 按 Ctrl+C 停止前端和后端"
echo ""
echo "========================================================================"

# 处理信号
cleanup() {
    echo ""
    echo ""
    echo "🛑 停止所有服务..."
    kill $API_PID 2>/dev/null || true
    kill $WEB_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    wait $WEB_PID 2>/dev/null || true
    echo "✅ 已停止前端和后端"
    echo "💡 数据库仍在运行，使用 'docker-compose down' 停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待进程
wait
