# StarClassify-Vis：恒星分类可视化实验台（全栈重建版）

本仓库已按 PRD/技术架构/页面设计文档完成“推倒重建”：

- 模型：仅支持高斯朴素贝叶斯（GaussianNB），已移除逻辑回归。
- 后端：FastAPI 提供训练/预测与结果落库/查询 API。
- 数据库：PostgreSQL 存储实验记录。
- 前端：React + Tailwind，两页（实验台 / 结果记录），统一深色风格。

## 目录结构

```
StarClassify-Vis/
├─ api/                 # FastAPI 后端
├─ web/                 # React 前端
├─ docker-compose.yml   # Docker 一键部署
├─ datasets/            # 模拟数据集
└─ requirements.txt     # 后端依赖（也可使用 api/requirements.txt）
```

## 本地启动

### 1) 启动 PostgreSQL

```bash
docker compose up -d
```

### 2) 启动后端（FastAPI）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r api/requirements.txt

export DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/starvis'
uvicorn api.main:app --reload --port 8000
```

健康检查：

```bash
curl http://localhost:8000/api/health
```

### 3) 启动前端（React）

```bash
cd web
pnpm install
pnpm run dev
```

打开：`http://localhost:5173/`

## Docker 一键部署

```bash
./start_docker.sh
```

打开：
- 前端：`http://localhost:8080/`
- 后端：`http://localhost:8000/docs`

停止：

```bash
docker compose down
```

## 使用方式

- 实验台：上传 CSV → 选择目标列与特征列 → 配置 GaussianNB（可选 var_smoothing）→ 运行训练/预测 → 结果自动保存到数据库。
- 结果记录：查看历史记录、打开详情、将配置一键加载回实验台。
