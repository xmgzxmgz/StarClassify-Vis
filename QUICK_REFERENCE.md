# 🎯 快速参考卡片

## 启动项目

```bash
# 1. 确保 Docker Desktop 已运行
open /Applications/Docker.app

# 2. 启动所有服务
cd /Users/xiamuguizhi/code/StarClassify-Vis
docker-compose up -d

# 3. 等待服务启动
docker-compose ps

# 4. 访问应用
# 前端: http://localhost:8080
# API: http://localhost:8000
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `docker-compose ps` | 查看容器状态 |
| `docker-compose logs -f` | 查看实时日志 |
| `docker-compose logs -f api` | 查看 API 日志 |
| `docker-compose restart` | 重启所有服务 |
| `docker-compose down` | 停止所有服务 |
| `docker-compose down -v` | 停止并删除数据 |
| `python test_connection.py` | 测试连接 |

## 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:8080 | React 应用 |
| API | http://localhost:8000 | FastAPI 服务 |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| 数据库 | localhost:5433 | PostgreSQL |

## 数据库连接

```
Host: localhost
Port: 5433
User: postgres
Password: postgres
Database: starvis
```

## 主题切换

### 方式 1: 导航栏
- 点击顶部导航栏右侧的 Moon/Sun 图标

### 方式 2: 设置页面
- 点击导航栏右侧的设置图标
- 进入"高级功能"部分
- 点击"外观主题"的切换开关

## 文件位置

| 文件 | 位置 | 说明 |
|------|------|------|
| 启动脚本 | `start.py` | 项目启动脚本 |
| 数据库初始化 | `init-db.sql` | 数据库 SQL 脚本 |
| 连接测试 | `test_connection.py` | 连接测试脚本 |
| 数据库设计 | `DATABASE_DESIGN.md` | 数据库文档 |
| 系统架构 | `SYSTEM_ARCHITECTURE.md` | 架构文档 |
| 启动指南 | `STARTUP_GUIDE.md` | 详细启动指南 |
| 项目总结 | `PROJECT_SUMMARY.md` | 项目完成总结 |

## 故障排除

### Docker 未运行
```bash
open /Applications/Docker.app
```

### 端口被占用
```bash
lsof -i :8080  # 查看占用 8080 的进程
kill -9 <PID>  # 杀死进程
```

### 数据库连接失败
```bash
docker-compose logs db      # 查看数据库日志
docker-compose restart db   # 重启数据库
sleep 30                    # 等待 30 秒
docker-compose logs api     # 查看 API 日志
```

### API 启动失败
```bash
docker-compose logs api           # 查看详细日志
docker-compose up -d --build api  # 重建 API 镜像
```

## 开发工作流

### 修改后端代码
```bash
# 修改 api/ 目录下的文件
# Uvicorn 会自动重新加载
docker-compose logs -f api
```

### 修改前端代码
```bash
# 重建前端镜像
docker-compose up -d --build web
```

### 本地开发（不用 Docker）
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

## 数据库操作

### 连接数据库
```bash
docker-compose exec db psql -U postgres -d starvis
```

### 查看表
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

## 项目结构

```
StarClassify-Vis/
├── api/                    # 后端 FastAPI
│   ├── main.py            # 应用入口
│   ├── db.py              # 数据库配置
│   ├── models.py          # ORM 模型
│   ├── schemas.py         # 数据验证
│   ├── ml.py              # 机器学习
│   ├── requirements.txt    # 依赖
│   ├── Dockerfile         # 镜像配置
│   └── routers/
│       └── runs.py        # API 端点
│
├── web/                    # 前端 React
│   ├── src/
│   │   ├── pages/         # 页面
│   │   ├── components/    # 组件
│   │   ├── context/       # 状态管理
│   │   ├── hooks/         # 自定义 Hook
│   │   └── App.tsx        # 根组件
│   ├── Dockerfile         # 镜像配置
│   ├── nginx.conf         # Nginx 配置
│   └── package.json       # 依赖
│
├── docker-compose.yml     # 容器编排
├── init-db.sql           # 数据库初始化
├── start.py              # 启动脚本
├── test_connection.py    # 连接测试
└── 文档/
    ├── DATABASE_DESIGN.md
    ├── SYSTEM_ARCHITECTURE.md
    ├── STARTUP_GUIDE.md
    ├── PROJECT_SUMMARY.md
    └── QUICK_REFERENCE.md (本文件)
```

## API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/runs` | 创建新实验 |
| GET | `/api/runs` | 查询实验列表 |
| GET | `/api/runs/{id}` | 获取实验详情 |
| GET | `/api/health` | 健康检查 |

## 性能指标

| 操作 | 响应时间 |
|------|---------|
| 上传小文件 | < 5s |
| 查询列表 | < 100ms |
| 查询详情 | < 50ms |
| 搜索 | < 200ms |

## 环境变量

| 变量 | 值 | 说明 |
|------|------|------|
| DATABASE_URL | postgresql+psycopg://postgres:postgres@db:5432/starvis | 数据库连接 |
| CORS_ORIGINS | http://localhost:8080,http://localhost:5173 | CORS 配置 |
| PYTHONUNBUFFERED | 1 | Python 输出缓冲 |

## 主题配置

### 浅色模式
- 背景: 白色 (#ffffff)
- 文字: 深灰色 (#1e293b)
- 强调: 靛蓝色 (#4f46e5)

### 深色模式
- 背景: 深灰色 (#0f172a)
- 文字: 白色 (#ffffff)
- 强调: 靛蓝色 (#818cf8)

## 快速链接

- 📖 [数据库设计](DATABASE_DESIGN.md)
- 📖 [系统架构](SYSTEM_ARCHITECTURE.md)
- 📖 [启动指南](STARTUP_GUIDE.md)
- 📖 [项目总结](PROJECT_SUMMARY.md)
- 🔗 [API 文档](http://localhost:8000/docs)

---

**最后更新**: 2026-03-22
**版本**: 0.1.0
