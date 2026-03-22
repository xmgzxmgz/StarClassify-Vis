-- StarClassify-Vis 数据库初始化脚本
-- 创建必要的表和索引

-- 创建 runs 表（实验记录）
CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- 数据集信息
    dataset_name VARCHAR(255) NOT NULL,
    target_column VARCHAR(255) NOT NULL,
    feature_columns JSONB NOT NULL,
    
    -- 训练参数
    test_size FLOAT NOT NULL,
    random_state INTEGER,
    
    -- 模型信息
    model_type VARCHAR(50) NOT NULL,
    model_params JSONB NOT NULL,
    
    -- 结果数据
    metrics JSONB NOT NULL,
    confusion_matrix JSONB NOT NULL,
    labels JSONB NOT NULL
);

-- 创建索引以优化查询性能
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_dataset_name ON runs(dataset_name);
CREATE INDEX IF NOT EXISTS idx_runs_model_type ON runs(model_type);

-- 创建注释以文档化表结构
COMMENT ON TABLE runs IS '实验运行记录表 - 存储每次 GNB 模型训练的完整信息';
COMMENT ON COLUMN runs.id IS '唯一标识符 (UUID)';
COMMENT ON COLUMN runs.created_at IS '记录创建时间';
COMMENT ON COLUMN runs.dataset_name IS '数据集名称';
COMMENT ON COLUMN runs.target_column IS '目标列名';
COMMENT ON COLUMN runs.feature_columns IS 'JSON 数组 - 特征列名列表';
COMMENT ON COLUMN runs.test_size IS '测试集比例 (0.01-0.99)';
COMMENT ON COLUMN runs.random_state IS '随机种子 (可选)';
COMMENT ON COLUMN runs.model_type IS '模型类型 (gaussian_nb)';
COMMENT ON COLUMN runs.model_params IS 'JSON 对象 - 模型参数 (如 var_smoothing)';
COMMENT ON COLUMN runs.metrics IS 'JSON 对象 - 评估指标 (accuracy, precision, recall, f1)';
COMMENT ON COLUMN runs.confusion_matrix IS 'JSON 二维数组 - 混淆矩阵';
COMMENT ON COLUMN runs.labels IS 'JSON 数组 - 分类标签列表';

-- 创建视图用于快速统计
CREATE OR REPLACE VIEW runs_summary AS
SELECT 
    COUNT(*) as total_runs,
    COUNT(DISTINCT dataset_name) as unique_datasets,
    COUNT(DISTINCT model_type) as model_types,
    MAX(created_at) as latest_run,
    MIN(created_at) as earliest_run
FROM runs;

COMMENT ON VIEW runs_summary IS '实验运行统计视图';
