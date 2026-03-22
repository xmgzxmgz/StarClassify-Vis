# 🚀 StarClassify-Vis 启动指南

## 前置要求

- **Docker Desktop** (已安装并运行)
- **Docker Compose** v5.0+
- **macOS/Linux/Windows** (任何支持 Docker 的系统)

## 快速启动

### 1️⃣ 启动 Docker Desktop

在 macOS 上：
```bash
# 打开 Applications 文件夹中的 Docker.app
# 或使用命令行
open /Applications/Docker.app
```

等待 Docker 完全启动（菜单栏会显示 Docker 图标）。

### 2️⃣ 启动项目

```bash
cd /Users/xiamuguizhi/code/StarClassify-Vis
docker-compose up -d
```

### 3️⃣ 等待服务启动

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

等待所有容器状态为 `Up` 且健康检查通过。

### 4️⃣ 访问应用

- 🌐 **前端**: http://localhost:8080
- 🔌 **API**: http://localhost:8000
- 📊 **数据库**: localhost:5433 (postgres/postgres)

## 常用命令

### 查看服务状态
```bash
docker-compose ps
```

### 查看实时日志
```bash
# 所有服务
docker-compose logs -f

# 特定服务
docker-compose logs -f api
docker-compose logs -f db
docker-compose logs -f web
```

### 停止服务
```bash
docker-compose down
```

### 完全清理（包括数据）
```bash
docker-compose down -v
```

### 重启服务
```bash
docker-compose restart
```

### 重建镜像
```bash
docker-compose up -d --build
```

## 测试连接

启动后运行连接测试：

```bash
python test_connection.py
```

预期输出：
```
✅ 数据库连接成功
✅ API 服务连接成功
✅ 前端服务连接成功
```

## 故障排除

### ❌ Docker 未运行

**错误信息**: `permission denied while trying to connect to the Docker daemon`

**解决方案**:
1. 打开 Docker Desktop 应用
2. 等待完全启动
3. 重新运行 `docker-compose up -d`

### ❌ 端口被占用

**错误信息**: `bind: address already in use`

**解决方案**:
```bash
# 查看占用端口的进程
lsof -i :8080
lsof -i :8000
lsof -i :5433

# 杀死进程
kill -9 <PID>

# 或修改 docker-compose.yml 中的端口映射
```

### ❌ 数据库连接失败

**错误信息**: `connection refused` 或 `FATAL: remaining connection slots are reserved`

**解决方案**:
```bash
# 查看数据库日志
docker-compose logs db

# 重启数据库
docker-compose restart db

# 等待 30 秒后重试
sleep 30
docker-compose logs api
```

### ❌ API 启动失败

**错误信息**: `ImportError` 或 `ModuleNotFoundError`

**解决方案**:
```bash
# 重建 API 镜像
docker-compose up -d --build api

# 查看详细日志
docker-compose logs api
```

### ❌ 前端无法加载

**错误信息**: 浏览器显示空白或 404

**解决方案**:
```bash
# 重建前端镜像
docker-compose up -d --build web

# 清除浏览器缓存
# 按 Ctrl+Shift+Delete (Windows/Linux) 或 Cmd+Shift+Delete (macOS)
```

## 开发工作流

### 修改后端代码

由于 `docker-compose.yml` 中配置了 volume 挂载，修改 `api/` 目录下的代码会自动重新加载：

```bash
# 修改 api/main.py 或其他文件
# Uvicorn 会自动检测并重新加载
docker-compose logs -f api
```

### 修改前端代码

前端需要重新构建：

```bash
# 方式 1: 重建镜像
docker-compose up -d --build web

# 方式 2: 进入容器重新构建
docker-compose exec web npm run build
```

### 本地开发模式

如果想在本地运行（不用 Docker）：

```bash
# 后端
cd api
pip install -r requirements.txt
python -m uvicorn main:app --reload

# 前端（新终端）
cd web
npm install
npm run dev
```

## 数据库管理

### 连接数据库

```bash
# 使用 psql
docker-compose exec db psql -U postgres -d starvis

# 或使用 GUI 工具（如 DBeaver）
# Host: localhost
# Port: 5433
# User: postgres
# Password: postgres
# Database: starvis
```

### 查看表结构

```bash
docker-compose exec db psql -U postgres -d starvis -c "\dt"
```

### 备份数据库

```bash
docker-compose exec db pg_dump -U postgres starvis > backup.sql
```

### 恢复数据库

```bash
docker-compose exec -T db psql -U postgres starvis < backup.sql
```

## 性能监控

### 查看容器资源使用

```bash
docker stats
```

### 查看数据库连接

```bash
docker-compose exec db psql -U postgres -d starvis -c "SELECT count(*) FROM pg_stat_activity;"
```

### 查看表大小

```bash
docker-compose exec db psql -U postgres -d starvis -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

## 功能说明

### 🎨 主题切换

- **位置**: 顶部导航栏右侧（Moon/Sun 图标）或设置页面
- **支持**: 浅色模式 / 深色模式
- **自动保存**: 主题偏好保存到浏览器 localStorage
- **系统偏好**: 首次访问时自动检测系统主题

### 🧪 实验台页面

1. 上传 CSV 数据集
2. 选择目标列和特征列
3. 配置高斯朴素贝叶斯参数
4. 执行训练和预测
5. 查看评估指标和混淆矩阵
6. 保存实验结果

### 📊 结果记录页面

1. 查看历史实验记录
2. 按数据集名称搜索
3. 查看单条记录详情
4. 复现之前的实验

### ⚙️ 设置页面

- **高级功能**
  - 外观主题切换（浅色/深色）
  - 主题说明和建议

## 项目结构

```
StarClassify-Vis/
├── api/                          # 后端 FastAPI 应用
│   ├── main.py                   # 应用入口
│   ├── db.py                     # 数据库配置
│   ├── models.py                 # ORM 模型
│   ├── schemas.py                # 数据验证
│   ├── ml.py                     # 机器学习逻辑
│   ├── settings.py               # 配置管理
│   ├── requirements.txt          # Python 依赖
│   ├── Dockerfile                # API 镜像配置
│   └── routers/
│       └── runs.py               # API 端点
│
├── web/                          # 前端 React 应用
│   ├── src/
│   │   ├── pages/                # 页面组件
│   │   ├── components/           # UI 组件
│   │   ├── context/              # 全局状态
│   │   ├── hooks/                # 自定义 Hook
│   │   ├── api/                  # HTTP 客户端
│   │   └── App.tsx               # 根组件
│   ├── Dockerfile                # 前端镜像配置
│   ├── nginx.conf                # Nginx 配置
│   ├── package.json              # 依赖管理
│   └── vite.config.ts            # Vite 配置
│
├── docker-compose.yml            # 容器编排配置
├── init-db.sql                   # 数据库初始化脚本
├── start.py                      # 启动脚本
├── test_connection.py            # 连接测试脚本
├── DATABASE_DESIGN.md            # 数据库设计文档
└── SYSTEM_ARCHITECTURE.md        # 系统架构文档
```

## 文档

- 📖 **DATABASE_DESIGN.md** - 数据库设计、ER 图、数据流
- 📖 **SYSTEM_ARCHITECTURE.md** - 系统架构、通信协议、部署拓扑
- 📖 **STARTUP_GUIDE.md** - 本文档

## 获取帮助

### 查看 API 文档

启动后访问: http://localhost:8000/docs

### 查看容器日志

```bash
docker-compose logs [service_name]
```

### 进入容器调试

```bash
# 进入 API 容器
docker-compose exec api bash

# 进入数据库容器
docker-compose exec db bash

# 进入前端容器
docker-compose exec web bash
```

## 下一步

1. ✅ 启动项目
2. ✅ 访问 http://localhost:8080
3. ✅ 上传示例数据集（在 `datasets/` 目录中）
4. ✅ 配置模型参数
5. ✅ 执行训练
6. ✅ 查看结果
7. ✅ 在设置页面切换主题

祝你使用愉快！🎉
