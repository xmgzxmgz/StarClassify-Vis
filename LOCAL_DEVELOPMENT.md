# 🚀 本地开发启动指南

## 概述

本地开发启动脚本允许你在本地运行前端和后端，同时使用 Docker 中的 PostgreSQL 数据库。这种方式适合快速开发和调试。

## 启动方式对比

| 方式 | 脚本 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| **完整 Docker** | `start.py` | 生产环境、完整测试 | 环境一致、易部署 | 启动较慢 |
| **本地开发** | `start_local.py` / `start_local.sh` | 开发调试 | 快速迭代、易调试 | 需要本地环境 |

## 前置要求

### 必需
- **Python 3.9+**
- **Node.js 16+** 和 **npm 8+**
- **Docker Desktop** (已安装并运行)
- **Docker Compose 5.0+**

### 检查环境

```bash
# 检查 Python
python3 --version

# 检查 Node.js
node --version
npm --version

# 检查 Docker
docker --version
docker-compose --version
```

## 快速启动

### 方式 1: 使用 Python 脚本 (推荐)

```bash
cd /Users/xiamuguizhi/code/StarClassify-Vis
python3 start_local.py
```

### 方式 2: 使用 Bash 脚本

```bash
cd /Users/xiamuguizhi/code/StarClassify-Vis
bash start_local.sh
```

### 方式 3: 手动启动

```bash
# 1. 启动 Docker 数据库
docker-compose up -d db

# 2. 等待数据库就绪
sleep 5

# 3. 安装后端依赖
cd api
pip install -r requirements.txt

# 4. 启动后端 (新终端)
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5433/starvis"
export CORS_ORIGINS="http://localhost:5173"
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. 安装前端依赖 (新终端)
cd web
npm install

# 6. 启动前端 (新终端)
npm run dev
```

## 启动后的访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:5173 | React 开发服务器 |
| API | http://localhost:8000 | FastAPI 服务 |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| 数据库 | localhost:5433 | PostgreSQL |

## 脚本功能

### start_local.py (Python 版本)

**优点:**
- 跨平台支持 (Windows/macOS/Linux)
- 更好的错误处理
- 详细的进度提示
- 自动依赖检查

**功能:**
1. ✅ 检查 Python、Node.js、Docker 版本
2. ✅ 启动 Docker 数据库容器
3. ✅ 等待数据库就绪
4. ✅ 安装后端依赖
5. ✅ 安装前端依赖
6. ✅ 启动后端 API
7. ✅ 启动前端开发服务器
8. ✅ 优雅处理 Ctrl+C 信号

**使用:**
```bash
python3 start_local.py
```

### start_local.sh (Bash 版本)

**优点:**
- 轻量级
- 快速启动
- 适合 macOS/Linux

**功能:**
- 同 Python 版本

**使用:**
```bash
bash start_local.sh
```

## 开发工作流

### 修改后端代码

后端使用 `--reload` 标志，修改代码会自动重新加载：

```bash
# 修改 api/ 目录下的任何文件
# Uvicorn 会自动检测并重新加载
# 查看终端输出确认重新加载
```

**常见修改:**
- `api/main.py` - 应用配置
- `api/routers/runs.py` - API 端点
- `api/ml.py` - 机器学习逻辑
- `api/models.py` - 数据库模型

### 修改前端代码

前端使用 Vite 热更新，修改代码会自动刷新浏览器：

```bash
# 修改 web/src/ 目录下的任何文件
# 浏览器会自动刷新
```

**常见修改:**
- `web/src/pages/` - 页面组件
- `web/src/components/` - UI 组件
- `web/src/context/` - 全局状态
- `web/src/hooks/` - 自定义 Hook

### 修改数据库

如果修改了数据库模型，需要重启后端：

```bash
# 1. 停止后端 (Ctrl+C)
# 2. 修改 api/models.py
# 3. 重新启动后端
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 常用命令

### 查看数据库日志
```bash
docker-compose logs -f db
```

### 连接数据库
```bash
docker-compose exec db psql -U postgres -d starvis
```

### 停止数据库
```bash
docker-compose down
```

### 重启数据库
```bash
docker-compose restart db
```

### 查看 API 文档
访问: http://localhost:8000/docs

### 测试连接
```bash
python test_connection.py
```

## 故障排除

### ❌ "Python 3.9+ 未找到"

**解决方案:**
```bash
# 检查 Python 版本
python3 --version

# 如果版本过低，安装新版本
# macOS
brew install python@3.11

# 或使用 pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

### ❌ "Node.js 未找到"

**解决方案:**
```bash
# macOS
brew install node

# 或使用 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### ❌ "Docker 未运行"

**解决方案:**
```bash
# macOS
open /Applications/Docker.app

# 等待 Docker 完全启动后重试
```

### ❌ "端口被占用"

**解决方案:**
```bash
# 查看占用端口的进程
lsof -i :5173  # 前端
lsof -i :8000  # 后端
lsof -i :5433  # 数据库

# 杀死进程
kill -9 <PID>

# 或修改端口
# 前端: web/vite.config.ts
# 后端: 启动命令中的 --port
```

### ❌ "数据库连接失败"

**解决方案:**
```bash
# 查看数据库日志
docker-compose logs db

# 重启数据库
docker-compose restart db

# 等待 30 秒后重试
sleep 30

# 检查数据库是否就绪
docker-compose exec db pg_isready -U postgres
```

### ❌ "后端启动失败"

**解决方案:**
```bash
# 查看详细错误
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 检查依赖
pip install -r api/requirements.txt

# 检查数据库连接
python test_connection.py
```

### ❌ "前端启动失败"

**解决方案:**
```bash
# 清除 node_modules 和缓存
cd web
rm -rf node_modules package-lock.json
npm install

# 重新启动
npm run dev
```

### ❌ "API 无法连接到数据库"

**解决方案:**
```bash
# 检查环境变量
echo $DATABASE_URL

# 应该输出:
# postgresql+psycopg://postgres:postgres@localhost:5433/starvis

# 如果不对，重新设置
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5433/starvis"

# 重启后端
```

## 性能优化

### 后端优化

1. **使用虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r api/requirements.txt
   ```

2. **使用 uvloop**
   ```bash
   pip install uvloop
   python -m uvicorn api.main:app --reload --loop uvloop
   ```

3. **增加工作进程**
   ```bash
   python -m uvicorn api.main:app --reload --workers 4
   ```

### 前端优化

1. **使用 pnpm 替代 npm**
   ```bash
   npm install -g pnpm
   cd web
   pnpm install
   pnpm run dev
   ```

2. **清除缓存**
   ```bash
   cd web
   rm -rf .vite dist node_modules
   npm install
   npm run dev
   ```

## 调试技巧

### 后端调试

1. **添加日志**
   ```python
   # api/main.py
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   logger.debug("调试信息")
   ```

2. **使用 pdb**
   ```python
   # api/routers/runs.py
   import pdb; pdb.set_trace()
   ```

3. **查看 SQL 查询**
   ```python
   # api/db.py
   engine = create_engine(get_database_url(), echo=True)
   ```

### 前端调试

1. **浏览器开发者工具**
   - F12 打开开发者工具
   - 查看 Console、Network、Sources 标签

2. **React DevTools**
   - 安装 React DevTools 浏览器扩展
   - 查看组件树和状态

3. **添加日志**
   ```typescript
   // web/src/pages/Lab.tsx
   console.log("调试信息", data);
   ```

## 环境变量

### 后端环境变量

| 变量 | 值 | 说明 |
|------|------|------|
| DATABASE_URL | postgresql+psycopg://postgres:postgres@localhost:5433/starvis | 数据库连接 |
| CORS_ORIGINS | http://localhost:5173,http://localhost:5174 | CORS 配置 |
| PYTHONUNBUFFERED | 1 | Python 输出缓冲 |

### 前端环境变量

在 `web/.env` 中配置：

```env
VITE_API_URL=http://localhost:8000
```

## 项目结构

```
StarClassify-Vis/
├── api/                    # 后端
│   ├── main.py            # 应用入口
│   ├── db.py              # 数据库配置
│   ├── models.py          # ORM 模型
│   ├── schemas.py         # 数据验证
│   ├── ml.py              # 机器学习
│   ├── settings.py        # 配置
│   ├── requirements.txt    # 依赖
│   ├── Dockerfile         # 镜像配置
│   └── routers/
│       └── runs.py        # API 端点
│
├── web/                    # 前端
│   ├── src/
│   │   ├── pages/         # 页面
│   │   ├── components/    # 组件
│   │   ├── context/       # 状态
│   │   ├── hooks/         # Hook
│   │   ├── api/           # HTTP 客户端
│   │   └── App.tsx        # 根组件
│   ├── Dockerfile         # 镜像配置
│   ├── package.json       # 依赖
│   ├── vite.config.ts     # Vite 配置
│   └── tailwind.config.js # Tailwind 配置
│
├── docker-compose.yml     # 容器编排
├── init-db.sql           # 数据库初始化
├── start.py              # 完整 Docker 启动
├── start_local.py        # 本地开发启动 (Python)
├── start_local.sh        # 本地开发启动 (Bash)
└── 文档/
    ├── DATABASE_DESIGN.md
    ├── SYSTEM_ARCHITECTURE.md
    ├── STARTUP_GUIDE.md
    ├── LOCAL_DEVELOPMENT.md (本文件)
    └── ...
```

## 下一步

1. ✅ 运行 `python3 start_local.py`
2. ✅ 访问 http://localhost:5173
3. ✅ 开始开发
4. ✅ 修改代码并查看实时更新
5. ✅ 使用浏览器开发者工具调试

## 获取帮助

### 查看 API 文档
http://localhost:8000/docs

### 查看数据库日志
```bash
docker-compose logs -f db
```

### 进入容器
```bash
docker-compose exec db bash
```

---

**最后更新**: 2026-03-22
**版本**: 0.1.0
