# 📊 启动脚本对比和选择指南

## 三种启动方式

### 1️⃣ 完整 Docker 启动 (`start.py`)

**用途**: 生产环境、完整测试、CI/CD 部署

**启动命令**:
```bash
python3 start.py
```

**启动内容**:
- ✅ PostgreSQL 数据库 (Docker)
- ✅ FastAPI 后端 (Docker)
- ✅ React 前端 (Docker + Nginx)

**访问地址**:
- 前端: http://localhost:8080
- API: http://localhost:8000
- 数据库: localhost:5433

**优点**:
- 🟢 环境完全一致
- 🟢 易于部署到生产环境
- 🟢 无需本地依赖
- 🟢 容器隔离

**缺点**:
- 🔴 启动较慢 (60-90 秒)
- 🔴 代码修改需要重建镜像
- 🔴 调试较困难
- 🔴 占用更多系统资源

**适用场景**:
- 完整功能测试
- 生产环境部署
- 团队协作
- CI/CD 流程

---

### 2️⃣ 本地开发启动 - Python 版 (`start_local.py`)

**用途**: 本地开发、快速迭代、调试

**启动命令**:
```bash
python3 start_local.py
```

**启动内容**:
- ✅ PostgreSQL 数据库 (Docker)
- ✅ FastAPI 后端 (本地)
- ✅ React 前端 (本地)

**访问地址**:
- 前端: http://localhost:5173
- API: http://localhost:8000
- 数据库: localhost:5433

**优点**:
- 🟢 启动快速 (10-20 秒)
- 🟢 代码修改自动重新加载
- 🟢 易于调试
- 🟢 跨平台支持
- 🟢 自动依赖检查

**缺点**:
- 🔴 需要本地 Python 环境
- 🔴 需要本地 Node.js 环境
- 🔴 环境可能不一致

**适用场景**:
- 日常开发
- 快速原型
- 功能调试
- 性能优化

---

### 3️⃣ 本地开发启动 - Bash 版 (`start_local.sh`)

**用途**: 本地开发 (macOS/Linux)

**启动命令**:
```bash
bash start_local.sh
```

**启动内容**:
- ✅ PostgreSQL 数据库 (Docker)
- ✅ FastAPI 后端 (本地)
- ✅ React 前端 (本地)

**访问地址**:
- 前端: http://localhost:5173
- API: http://localhost:8000
- 数据库: localhost:5433

**优点**:
- 🟢 轻量级脚本
- 🟢 快速启动
- 🟢 适合 macOS/Linux

**缺点**:
- 🔴 不支持 Windows
- 🔴 错误处理较少

**适用场景**:
- macOS/Linux 开发
- 快速启动

---

## 选择指南

### 我应该选择哪个脚本？

```
┌─ 你在做什么？
│
├─ 生产部署或完整测试？
│  └─ 使用 start.py (完整 Docker)
│
├─ 本地开发和调试？
│  ├─ Windows 系统？
│  │  └─ 使用 start_local.py (Python 版)
│  ├─ macOS/Linux 系统？
│  │  ├─ 喜欢 Python？
│  │  │  └─ 使用 start_local.py (Python 版)
│  │  └─ 喜欢 Bash？
│  │     └─ 使用 start_local.sh (Bash 版)
│
└─ 手动启动？
   └─ 参考 LOCAL_DEVELOPMENT.md 中的"方式 3"
```

### 快速决策表

| 场景 | 推荐脚本 | 原因 |
|------|---------|------|
| 生产部署 | `start.py` | 环境一致 |
| 完整测试 | `start.py` | 完整模拟 |
| 日常开发 | `start_local.py` | 快速迭代 |
| 快速调试 | `start_local.py` | 自动重载 |
| macOS 开发 | `start_local.py` 或 `start_local.sh` | 都可以 |
| Linux 开发 | `start_local.py` 或 `start_local.sh` | 都可以 |
| Windows 开发 | `start_local.py` | 唯一选择 |

---

## 详细对比

### 启动时间

```
start.py (完整 Docker)
├─ 构建镜像: 30-60 秒 (首次)
├─ 启动容器: 20-30 秒
└─ 总计: 60-90 秒 (首次), 20-30 秒 (后续)

start_local.py (本地开发)
├─ 启动数据库: 5-10 秒
├─ 安装依赖: 10-30 秒 (首次)
├─ 启动后端: 3-5 秒
├─ 启动前端: 3-5 秒
└─ 总计: 10-20 秒 (后续), 30-60 秒 (首次)
```

### 资源占用

```
start.py (完整 Docker)
├─ CPU: 中等 (构建时高)
├─ 内存: 800MB - 1.2GB
└─ 磁盘: 2-3GB (镜像)

start_local.py (本地开发)
├─ CPU: 低
├─ 内存: 300-500MB
└─ 磁盘: 500MB (node_modules + venv)
```

### 代码修改反应时间

```
start.py (完整 Docker)
├─ 后端修改: 需要重建镜像 (30-60 秒)
└─ 前端修改: 需要重建镜像 (30-60 秒)

start_local.py (本地开发)
├─ 后端修改: 自动重载 (1-2 秒)
└─ 前端修改: 热更新 (1-2 秒)
```

### 调试能力

```
start.py (完整 Docker)
├─ 日志查看: docker-compose logs
├─ 进入容器: docker-compose exec
├─ 设置断点: 困难
└─ 总体: ⭐⭐

start_local.py (本地开发)
├─ 日志查看: 直接在终端
├─ 进入进程: 直接调试
├─ 设置断点: 容易 (pdb, debugger)
└─ 总体: ⭐⭐⭐⭐⭐
```

---

## 工作流示例

### 场景 1: 新功能开发

```
1. 启动本地开发环境
   python3 start_local.py

2. 修改后端代码
   编辑 api/routers/runs.py
   → 自动重载 (1-2 秒)

3. 修改前端代码
   编辑 web/src/pages/Lab.tsx
   → 热更新 (1-2 秒)

4. 在浏览器中测试
   http://localhost:5173

5. 使用开发者工具调试
   F12 打开开发者工具

6. 完成后，运行完整测试
   docker-compose down
   python3 start.py
```

### 场景 2: Bug 修复

```
1. 启动本地开发环境
   python3 start_local.py

2. 查看错误日志
   查看后端终端输出
   查看浏览器控制台

3. 添加调试代码
   import pdb; pdb.set_trace()

4. 修改代码
   → 自动重载

5. 验证修复
   http://localhost:5173

6. 清理调试代码
   提交代码
```

### 场景 3: 性能优化

```
1. 启动本地开发环境
   python3 start_local.py

2. 使用浏览器性能工具
   F12 → Performance 标签

3. 修改代码优化
   → 热更新

4. 重新测试性能
   → 立即看到效果

5. 对比优化前后
   记录性能指标
```

---

## 常见问题

### Q: 我应该同时运行两个脚本吗？

**A**: 不应该。选择一个脚本运行。如果需要切换：

```bash
# 停止当前脚本
Ctrl+C

# 停止数据库
docker-compose down

# 启动新脚本
python3 start.py  # 或 start_local.py
```

### Q: 本地开发时如何测试生产环境？

**A**: 使用完整 Docker 脚本：

```bash
# 停止本地开发
Ctrl+C
docker-compose down

# 启动完整 Docker
python3 start.py

# 测试完成后
docker-compose down
```

### Q: 如何在两个脚本之间切换？

**A**: 

```bash
# 1. 停止当前脚本
Ctrl+C

# 2. 停止数据库
docker-compose down

# 3. 等待 5 秒
sleep 5

# 4. 启动新脚本
python3 start.py  # 或 start_local.py
```

### Q: 本地开发时数据库会丢失吗？

**A**: 不会。数据库在 Docker 中运行，数据持久化到 volume。只有运行 `docker-compose down -v` 才会删除数据。

### Q: 如何在本地开发时使用生产数据库？

**A**: 修改环境变量：

```bash
export DATABASE_URL="postgresql+psycopg://user:password@prod-host:5432/starvis"
python3 start_local.py
```

### Q: 本地开发时如何调试数据库查询？

**A**: 修改 `api/db.py`：

```python
engine = create_engine(
    get_database_url(),
    echo=True,  # 打印所有 SQL 查询
    pool_pre_ping=True,
)
```

---

## 迁移指南

### 从完整 Docker 迁移到本地开发

```bash
# 1. 停止完整 Docker
docker-compose down

# 2. 启动本地开发
python3 start_local.py
```

### 从本地开发迁移到完整 Docker

```bash
# 1. 停止本地开发
Ctrl+C

# 2. 停止数据库
docker-compose down

# 3. 启动完整 Docker
python3 start.py
```

---

## 最佳实践

### ✅ 推荐做法

1. **日常开发使用本地脚本**
   ```bash
   python3 start_local.py
   ```

2. **提交前使用完整 Docker 测试**
   ```bash
   python3 start.py
   ```

3. **定期清理数据库**
   ```bash
   docker-compose down -v
   ```

4. **使用虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **保持依赖最新**
   ```bash
   pip install -r api/requirements.txt --upgrade
   npm update
   ```

### ❌ 避免做法

1. ❌ 同时运行两个脚本
2. ❌ 修改 docker-compose.yml 后不重建
3. ❌ 忽视错误日志
4. ❌ 在 Docker 中修改代码
5. ❌ 使用过期的依赖版本

---

## 总结

| 脚本 | 启动时间 | 资源占用 | 调试能力 | 推荐场景 |
|------|---------|---------|---------|---------|
| `start.py` | 60-90s | 高 | ⭐⭐ | 生产部署 |
| `start_local.py` | 10-20s | 低 | ⭐⭐⭐⭐⭐ | 日常开发 |
| `start_local.sh` | 10-20s | 低 | ⭐⭐⭐⭐⭐ | macOS/Linux |

**建议**: 日常开发使用 `start_local.py`，提交前使用 `start.py` 进行完整测试。

---

**最后更新**: 2026-03-22
**版本**: 0.1.0
