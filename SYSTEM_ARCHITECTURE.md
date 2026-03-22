# StarClassify-Vis 系统架构设计

## 1. 整体系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          用户浏览器 (Browser)                               │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP/HTTPS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Nginx 反向代理 (Port 8080)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 静态资源服务 (HTML/CSS/JS)                                          │   │
│  │ 路由转发: /api/* → http://api:8000                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │  前端应用 (React)    │        │  API 服务 (FastAPI) │
        │  - 实验台页面        │        │  - 数据验证          │
        │  - 结果记录页面      │        │  - ML 训练/预测      │
        │  - 数据可视化        │        │  - 数据库操作        │
        └──────────────────────┘        └──────────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────────┐
                                    │  PostgreSQL 数据库   │
                                    │  - runs 表           │
                                    │  - 索引和视图        │
                                    └──────────────────────┘
```

## 2. 前端架构 (React + Vite)

```
web/
├── src/
│   ├── pages/
│   │   ├── Home.tsx          # 首页
│   │   ├── Lab.tsx           # 实验台页面 (核心)
│   │   └── Runs.tsx          # 结果记录页面
│   │
│   ├── components/
│   │   ├── workspaces/
│   │   │   ├── EducatorWorkspace.tsx
│   │   │   ├── PublicWorkspace.tsx
│   │   │   └── ResearcherWorkspace.tsx
│   │   ├── LabConfigPanel.tsx      # 模型配置面板
│   │   ├── LabResultsPanel.tsx     # 结果展示面板
│   │   ├── ConfusionMatrixTable.tsx # 混淆矩阵表格
│   │   ├── ResultMetrics.tsx       # 评估指标展示
│   │   └── ...其他组件
│   │
│   ├── api/
│   │   └── http.ts           # HTTP 客户端 (axios/fetch)
│   │
│   ├── context/
│   │   └── ModeContext.tsx    # 全局状态管理
│   │
│   ├── hooks/
│   │   ├── useTheme.ts        # 主题 Hook
│   │   └── useToast.ts        # 提示 Hook
│   │
│   ├── types.ts              # TypeScript 类型定义
│   ├── App.tsx               # 根组件
│   └── main.tsx              # 入口文件
│
├── public/
│   └── datasets/             # 示例数据集
│
├── Dockerfile                # Docker 镜像配置
├── nginx.conf                # Nginx 配置
├── vite.config.ts            # Vite 配置
├── tailwind.config.js        # Tailwind CSS 配置
└── package.json              # 依赖管理
```

### 2.1 前端数据流

```
用户交互
    ↓
React 组件状态更新
    ↓
调用 API 客户端 (api/http.ts)
    ↓
发送 HTTP 请求到后端
    ↓
接收响应并更新 UI
    ↓
展示结果或错误信息
```

### 2.2 关键页面流程

#### Lab 页面 (实验台)
```
1. 数据导入
   └─ 用户上传 CSV 文件
   └─ 前端预览数据

2. 特征配置
   └─ 选择目标列
   └─ 选择特征列

3. 模型配置
   └─ 设置 GNB 参数 (var_smoothing)
   └─ 设置测试集比例

4. 执行训练
   └─ POST /api/runs (上传文件 + 配置)
   └─ 显示进度
   └─ 接收结果

5. 结果展示
   └─ 显示指标 (Accuracy/Precision/Recall/F1)
   └─ 显示混淆矩阵
   └─ 显示预测分布

6. 结果保存
   └─ 自动保存到数据库
   └─ 显示保存成功提示
```

#### Runs 页面 (结果记录)
```
1. 加载记录列表
   └─ GET /api/runs?page=1&pageSize=20
   └─ 显示分页列表

2. 搜索功能
   └─ GET /api/runs?query=xxx&page=1
   └─ 按数据集名称搜索

3. 查看详情
   └─ GET /api/runs/{run_id}
   └─ 显示完整配置和结果

4. 复现实验
   └─ 加载配置到 Lab 页面
   └─ 用户可修改后重新训练
```

## 3. 后端架构 (FastAPI)

```
api/
├── main.py                   # FastAPI 应用入口
├── settings.py               # 配置管理
├── db.py                     # 数据库连接和会话
├── models.py                 # SQLAlchemy ORM 模型
├── schemas.py                # Pydantic 数据验证模型
├── ml.py                     # 机器学习逻辑
│
├── routers/
│   ├── __init__.py
│   └── runs.py               # 实验运行相关 API 端点
│
├── requirements.txt          # Python 依赖
└── Dockerfile                # Docker 镜像配置
```

### 3.1 API 端点设计

```
POST /api/runs
├─ 请求: multipart/form-data
│  ├─ file: CSV 文件
│  └─ payload: JSON 配置
├─ 处理流程:
│  ├─ 验证文件格式
│  ├─ 解析 CSV
│  ├─ 推断列信息
│  ├─ 执行 GNB 训练
│  ├─ 计算评估指标
│  └─ 保存到数据库
└─ 响应: RunResult JSON

GET /api/runs
├─ 查询参数:
│  ├─ query: 搜索关键词 (可选)
│  ├─ page: 页码 (默认 1)
│  └─ pageSize: 每页数量 (默认 20)
├─ 处理流程:
│  ├─ 构建 SQL 查询
│  ├─ 应用搜索过滤
│  ├─ 排序和分页
│  └─ 返回结果列表
└─ 响应: RunListResponse JSON

GET /api/runs/{run_id}
├─ 路径参数: run_id (UUID)
├─ 处理流程:
│  ├─ 验证 UUID 格式
│  ├─ 查询数据库
│  └─ 返回完整记录
└─ 响应: RunResult JSON

GET /api/health
├─ 健康检查端点
└─ 响应: {"ok": true}
```

### 3.2 后端数据流

```
HTTP 请求
    ↓
FastAPI 路由处理
    ↓
Pydantic 数据验证
    ↓
业务逻辑处理
    ├─ 文件解析 (CSV)
    ├─ 数据预处理
    ├─ ML 模型训练 (GNB)
    └─ 指标计算
    ↓
SQLAlchemy ORM 操作
    ├─ 构建 SQL 语句
    ├─ 执行数据库操作
    └─ 事务管理
    ↓
HTTP 响应
    └─ JSON 序列化
```

### 3.3 关键模块说明

#### models.py - ORM 模型
```python
class Run(Base):
    __tablename__ = "runs"
    
    # 主键和时间戳
    id: UUID
    created_at: datetime
    
    # 输入参数
    dataset_name: str
    target_column: str
    feature_columns: list[str]
    test_size: float
    random_state: int | None
    
    # 模型信息
    model_type: str
    model_params: dict
    
    # 输出结果
    metrics: dict
    confusion_matrix: list[list[int]]
    labels: list[str]
```

#### schemas.py - 数据验证
```python
class RunCreateRequest:
    datasetName: str
    targetColumn: str
    featureColumns: list[str]
    testSize: float
    randomState: int | None
    modelType: Literal["gaussian_nb"]
    gnbParams: GnbParams

class RunResult:
    id: str
    createdAt: datetime
    request: RunCreateRequest
    metrics: Metrics
    confusionMatrix: list[list[int]]
    labels: list[str]
```

#### ml.py - 机器学习逻辑
```python
def train_gaussian_nb(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    test_size: float,
    random_state: int | None,
    var_smoothing: float | None
) -> TrainOutput:
    # 1. 数据分割
    # 2. 模型训练
    # 3. 预测
    # 4. 指标计算
    # 5. 返回结果
```

## 4. 数据库架构 (PostgreSQL)

```
Database: starvis
│
└─ Schema: public
   │
   ├─ Table: runs
   │  ├─ Columns: 11 个字段
   │  ├─ Indexes: 3 个索引
   │  └─ Constraints: PRIMARY KEY, NOT NULL
   │
   └─ View: runs_summary
      └─ 统计视图
```

### 4.1 数据库连接池

```
FastAPI 应用
    ↓
SQLAlchemy Engine
    ├─ 连接池 (pool_size=5, max_overflow=10)
    ├─ 连接重用
    └─ 自动重连 (pool_pre_ping=True)
    ↓
PostgreSQL 数据库
```

## 5. Docker 容器编排

```
docker-compose.yml
│
├─ Service: db (PostgreSQL)
│  ├─ Image: postgres:16-alpine
│  ├─ Port: 5433:5432
│  ├─ Volume: starvis_pgdata
│  ├─ Health Check: pg_isready
│  └─ Network: starvis-network
│
├─ Service: api (FastAPI)
│  ├─ Build: ./api/Dockerfile
│  ├─ Port: 8000:8000
│  ├─ Depends On: db (healthy)
│  ├─ Environment: DATABASE_URL, CORS_ORIGINS
│  └─ Network: starvis-network
│
└─ Service: web (React + Nginx)
   ├─ Build: ./web/Dockerfile
   ├─ Port: 8080:80
   ├─ Depends On: api
   └─ Network: starvis-network
```

### 5.1 启动顺序

```
1. PostgreSQL 启动
   └─ 等待健康检查通过

2. FastAPI 启动
   └─ 连接数据库
   └─ 创建表结构
   └─ 启动 Uvicorn 服务器

3. React + Nginx 启动
   └─ 构建前端应用
   └─ 启动 Nginx 反向代理
   └─ 配置 API 路由转发
```

## 6. 通信协议

### 6.1 前端 ↔ 后端

```
请求格式:
POST /api/runs
Content-Type: multipart/form-data

file: [CSV 文件二进制数据]
payload: {
  "datasetName": "sdss_like_balanced",
  "targetColumn": "class",
  "featureColumns": ["u_mag", "g_mag", "r_mag", "i_mag", "z_mag"],
  "testSize": 0.2,
  "randomState": 42,
  "modelType": "gaussian_nb",
  "gnbParams": {
    "varSmoothing": 1e-9
  }
}

响应格式:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "createdAt": "2024-03-21T10:30:00Z",
  "request": {...},
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94,
    "f1": 0.935
  },
  "confusionMatrix": [[450, 20, 5], [15, 480, 10], [8, 12, 495]],
  "labels": ["STAR", "GALAXY", "QSO"]
}
```

### 6.2 后端 ↔ 数据库

```
连接字符串:
postgresql+psycopg://postgres:postgres@db:5432/starvis

查询示例:
SELECT * FROM runs 
WHERE dataset_name ILIKE '%sdss%'
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;

事务管理:
BEGIN;
  INSERT INTO runs (...) VALUES (...);
  -- 如果出错自动 ROLLBACK
COMMIT;
```

## 7. 部署拓扑

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Host                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │            Docker Network (starvis-network)       │  │
│  │                                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │  │
│  │  │   Nginx      │  │   FastAPI    │  │  PG    │  │  │
│  │  │  :8080       │  │  :8000       │  │ :5432  │  │  │
│  │  └──────────────┘  └──────────────┘  └────────┘  │  │
│  │         ↑                  ↑              ↑        │  │
│  │         └──────────────────┴──────────────┘        │  │
│  │              内部通信 (DNS)                        │  │
│  └───────────────────────────────────────────────────┘  │
│         ↑                                                │
│         │ 外部访问                                      │
│    :8080, :5433                                         │
└─────────────────────────────────────────────────────────┘
```

## 8. 性能指标

### 8.1 预期性能

| 操作 | 响应时间 | 说明 |
|------|---------|------|
| 上传小文件 (< 1MB) | < 5s | 包括训练时间 |
| 查询列表 (20 条) | < 100ms | 有索引优化 |
| 查询详情 | < 50ms | 单条记录查询 |
| 搜索 (模糊匹配) | < 200ms | 全表扫描 |

### 8.2 可扩展性建议

1. **数据库优化**
   - 添加更多索引
   - 使用分区表
   - 定期清理过期数据

2. **缓存策略**
   - Redis 缓存热门查询
   - 前端本地缓存

3. **异步处理**
   - 使用 Celery 处理长时间训练
   - 后台任务队列

4. **负载均衡**
   - 多个 API 实例
   - 数据库读写分离

## 9. 安全考虑

```
┌─────────────────────────────────────────────────────────┐
│                    安全层次                             │
├─────────────────────────────────────────────────────────┤
│ 1. 网络层                                               │
│    └─ Docker 网络隔离                                   │
│    └─ 防火墙规则                                        │
│                                                         │
│ 2. 应用层                                               │
│    └─ CORS 配置                                         │
│    └─ 输入验证 (Pydantic)                               │
│    └─ 错误处理                                          │
│                                                         │
│ 3. 数据库层                                             │
│    └─ 用户认证                                          │
│    └─ SQL 注入防护 (ORM)                                │
│    └─ 数据加密 (可选)                                   │
│                                                         │
│ 4. 数据保护                                             │
│    └─ 定期备份                                          │
│    └─ 访问日志                                          │
│    └─ 数据隐私                                          │
└─────────────────────────────────────────────────────────┘
```

## 10. 监控和日志

```
日志收集:
├─ Nginx 访问日志
├─ FastAPI 应用日志
├─ PostgreSQL 查询日志
└─ Docker 容器日志

监控指标:
├─ CPU 使用率
├─ 内存使用率
├─ 磁盘 I/O
├─ 数据库连接数
├─ API 响应时间
└─ 错误率

查看日志:
docker-compose logs -f [service_name]
```
