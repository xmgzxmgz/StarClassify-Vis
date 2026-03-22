# StarClassify-Vis 数据库设计文档

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     前端 (React + Vite)                      │
│                    http://localhost:8080                     │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  后端 API (FastAPI)                          │
│                  http://localhost:8000                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ /api/runs (POST)   - 创建新的实验运行                │   │
│  │ /api/runs (GET)    - 查询实验记录列表                │   │
│  │ /api/runs/{id}     - 获取单条实验详情                │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ SQLAlchemy ORM
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL 数据库 (Port 5433)                   │
│                    Database: starvis                         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数据库 ER 图

```
┌──────────────────────────────────────────────────────────────┐
│                         runs 表                              │
├──────────────────────────────────────────────────────────────┤
│ PK │ id (UUID)                                               │
├────┼──────────────────────────────────────────────────────────┤
│    │ created_at (TIMESTAMP WITH TIME ZONE)                   │
│    │ dataset_name (VARCHAR)                                  │
│    │ target_column (VARCHAR)                                 │
│    │ feature_columns (JSONB)                                 │
│    │ test_size (FLOAT)                                       │
│    │ random_state (INTEGER, nullable)                        │
│    │ model_type (VARCHAR)                                    │
│    │ model_params (JSONB)                                    │
│    │ metrics (JSONB)                                         │
│    │ confusion_matrix (JSONB)                                │
│    │ labels (JSONB)                                          │
└──────────────────────────────────────────────────────────────┘
```

## 3. 表结构详解

### 3.1 runs 表 - 实验运行记录

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| `id` | UUID | PRIMARY KEY | 唯一标识符，自动生成 |
| `created_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, DEFAULT CURRENT_TIMESTAMP | 记录创建时间，自动设置为当前时间 |
| `dataset_name` | VARCHAR(255) | NOT NULL | 数据集名称（用户上传的文件名或自定义名称） |
| `target_column` | VARCHAR(255) | NOT NULL | 目标列名（分类标签列） |
| `feature_columns` | JSONB | NOT NULL | 特征列名列表，JSON 数组格式 |
| `test_size` | FLOAT | NOT NULL | 测试集比例，范围 0.01-0.99 |
| `random_state` | INTEGER | NULLABLE | 随机种子，用于结果复现 |
| `model_type` | VARCHAR(50) | NOT NULL | 模型类型，当前仅支持 "gaussian_nb" |
| `model_params` | JSONB | NOT NULL | 模型参数，JSON 对象格式 |
| `model_params.var_smoothing` | FLOAT | NULLABLE | 高斯朴素贝叶斯的方差平滑参数 |
| `metrics` | JSONB | NOT NULL | 评估指标，JSON 对象格式 |
| `metrics.accuracy` | FLOAT | - | 准确率 (0-1) |
| `metrics.precision` | FLOAT | - | 精确率 (0-1) |
| `metrics.recall` | FLOAT | - | 召回率 (0-1) |
| `metrics.f1` | FLOAT | - | F1 分数 (0-1) |
| `confusion_matrix` | JSONB | NOT NULL | 混淆矩阵，JSON 二维数组格式 |
| `labels` | JSONB | NOT NULL | 分类标签列表，JSON 数组格式 |

### 3.2 索引设计

```sql
-- 按创建时间倒序查询（用于列表展示）
CREATE INDEX idx_runs_created_at ON runs(created_at DESC);

-- 按数据集名称模糊查询（用于搜索功能）
CREATE INDEX idx_runs_dataset_name ON runs(dataset_name);

-- 按模型类型查询（为未来扩展预留）
CREATE INDEX idx_runs_model_type ON runs(model_type);
```

### 3.3 视图设计

```sql
-- runs_summary 视图：用于快速获取统计信息
CREATE VIEW runs_summary AS
SELECT 
    COUNT(*) as total_runs,                    -- 总实验数
    COUNT(DISTINCT dataset_name) as unique_datasets,  -- 不同数据集数
    COUNT(DISTINCT model_type) as model_types,        -- 模型类型数
    MAX(created_at) as latest_run,             -- 最新实验时间
    MIN(created_at) as earliest_run            -- 最早实验时间
FROM runs;
```

## 4. 数据流向

### 4.1 创建实验流程

```
用户上传 CSV 文件
    ↓
前端发送 POST /api/runs (multipart/form-data)
    ├─ file: CSV 文件
    └─ payload: JSON 请求体
    ↓
后端验证和处理
    ├─ 解析 CSV 文件
    ├─ 推断目标列和特征列
    ├─ 执行 GNB 训练和预测
    └─ 计算评估指标
    ↓
保存到数据库
    └─ INSERT INTO runs (...)
    ↓
返回结果给前端
    └─ RunResult JSON 对象
```

### 4.2 查询实验流程

```
用户访问结果记录页面
    ↓
前端发送 GET /api/runs?query=xxx&page=1&pageSize=20
    ↓
后端查询数据库
    ├─ SELECT * FROM runs WHERE dataset_name ILIKE '%xxx%'
    ├─ ORDER BY created_at DESC
    └─ LIMIT 20 OFFSET 0
    ↓
返回分页结果
    └─ RunListResponse JSON 对象
```

### 4.3 获取详情流程

```
用户点击某条记录
    ↓
前端发送 GET /api/runs/{run_id}
    ↓
后端查询数据库
    └─ SELECT * FROM runs WHERE id = {run_id}
    ↓
返回完整记录
    └─ RunResult JSON 对象
```

## 5. JSON 数据结构示例

### 5.1 feature_columns 示例
```json
["u_mag", "g_mag", "r_mag", "i_mag", "z_mag"]
```

### 5.2 model_params 示例
```json
{
  "var_smoothing": 1e-9
}
```

### 5.3 metrics 示例
```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.94,
  "f1": 0.935
}
```

### 5.4 confusion_matrix 示例
```json
[
  [450, 20, 5],
  [15, 480, 10],
  [8, 12, 495]
]
```

### 5.5 labels 示例
```json
["STAR", "GALAXY", "QSO"]
```

## 6. 数据库连接配置

### 6.1 本地开发环境
```
DATABASE_URL: postgresql+psycopg://postgres:postgres@localhost:5432/starvis
```

### 6.2 Docker 环境
```
DATABASE_URL: postgresql+psycopg://postgres:postgres@db:5432/starvis
```

### 6.3 连接参数
- **Host**: localhost (本地) / db (Docker)
- **Port**: 5432 (内部) / 5433 (外部映射)
- **User**: postgres
- **Password**: postgres
- **Database**: starvis

## 7. 扩展建议

### 7.1 未来可能的表扩展

如果需要支持多个模型类型，可以考虑：

```sql
-- 模型配置表（可选）
CREATE TABLE model_configs (
    id UUID PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    config_name VARCHAR(255) NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据集元数据表（可选）
CREATE TABLE datasets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    row_count INTEGER,
    column_count INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 关联关系
ALTER TABLE runs ADD COLUMN dataset_id UUID REFERENCES datasets(id);
ALTER TABLE runs ADD COLUMN model_config_id UUID REFERENCES model_configs(id);
```

### 7.2 性能优化建议

1. **分区策略**：按 `created_at` 按月分区
   ```sql
   CREATE TABLE runs_2024_03 PARTITION OF runs
   FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
   ```

2. **归档策略**：定期将旧数据归档到冷存储

3. **缓存策略**：使用 Redis 缓存热门查询结果

## 8. 数据库维护

### 8.1 备份
```bash
# 完整备份
docker exec starvis-db pg_dump -U postgres starvis > backup.sql

# 恢复
docker exec -i starvis-db psql -U postgres starvis < backup.sql
```

### 8.2 监控
```sql
-- 查看表大小
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 查看索引使用情况
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 8.3 清理
```sql
-- 清理死行
VACUUM ANALYZE runs;

-- 重建索引
REINDEX TABLE runs;
```
