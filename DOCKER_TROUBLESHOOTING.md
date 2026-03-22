# 🔧 Docker 权限问题解决方案

## 问题诊断

**错误信息**: `permission denied while trying to connect to the Docker daemon socket`

**原因**: Docker Desktop 守护进程没有正确启动或权限不足

---

## 解决方案

### 方案 1: 完全重启 Docker Desktop (推荐)

**步骤:**

1. **完全关闭 Docker**
   ```bash
   # 在终端运行
   killall Docker
   ```

2. **等待 3 秒**
   ```bash
   sleep 3
   ```

3. **重新打开 Docker Desktop**
   ```bash
   open /Applications/Docker.app
   ```

4. **等待 Docker 完全启动**
   - 看到菜单栏 Docker 图标
   - 图标稳定不闪烁
   - 大约需要 30-60 秒

5. **验证 Docker**
   ```bash
   docker ps
   ```
   应该显示容器列表（可能为空）

6. **运行启动脚本**
   ```bash
   python3 start_local.py
   ```

---

### 方案 2: 重启 Mac (如果方案 1 不行)

```bash
# 重启 Mac
sudo shutdown -r now

# 重启后运行
python3 start_local.py
```

---

### 方案 3: 重新安装 Docker Desktop

如果上述方案都不行：

1. 卸载 Docker Desktop
   ```bash
   rm -rf /Applications/Docker.app
   ```

2. 从 [Docker 官网](https://www.docker.com/products/docker-desktop) 重新下载安装

3. 重新运行启动脚本

---

## 快速检查清单

- [ ] Docker Desktop 已安装
- [ ] Docker Desktop 已启动 (菜单栏有图标)
- [ ] 运行 `docker ps` 没有错误
- [ ] 运行 `docker-compose --version` 显示版本

---

## 如果仍然有问题

**临时解决方案**: 使用本地数据库而不是 Docker

修改 `start_local.py` 中的数据库连接字符串：

```python
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/starvis"
```

然后在本地安装 PostgreSQL：

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16
```

---

## 获取帮助

如果问题仍未解决，请：

1. 查看 Docker 日志
   ```bash
   cat ~/Library/Logs/Docker.log
   ```

2. 检查 Docker 进程
   ```bash
   ps aux | grep Docker
   ```

3. 查看系统日志
   ```bash
   log stream --predicate 'process == "Docker"'
   ```

---

**建议**: 先尝试方案 1 (完全重启 Docker)，这通常能解决 90% 的问题。
