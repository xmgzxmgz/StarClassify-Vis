# StarClassify-Vis：基于 SDSS 的恒星分类与科普可视化平台

> 轻量化集成模型（逻辑回归 + 朴素贝叶斯软投票）、多页面交互式可视化（科研/科普）、一键端到端测试与研究包打包分发。

## 目录
- [项目介绍](#项目介绍)
- [使用指南](#使用指南)
- [API 文档与示例](#api-文档与示例)
- [常见问题与故障排除](#常见问题与故障排除)
- [未来规划](#未来规划)
- [社区参与与贡献](#社区参与与贡献)
- [许可证](#许可证)
- [联系方式与支持](#联系方式与支持)
- [相关资源与参考链接](#相关资源与参考链接)

---

## 项目介绍

### 核心功能与价值定位
- 基于 SDSS 数据的恒星分类：通过颜色指数、空间与物理参数构建特征，训练轻量化集成模型进行分类与解释。
- 科研/科普双模式可视化：提供“科研筛选工具”“赫罗图科普互动”“典型案例演示”等页面，兼顾专业分析与教学演示。
- 一键端到端验证与研究包分发：内置测试中心与批量操作功能，支持模型、数据、报告的 ZIP 打包与版本化管理。
- 高级能力集成：支持超参数自动调优、SMOTE 类不平衡处理、SHAP 特征重要性分析、训练过程监控、PDF 报告生成。

### 适用用户与使用场景
- 天文数据科研人员：快速筛选高置信度样本，生成评估报告，管理模型版本与研究包。
- 大学教师与学生：通过赫罗图互动和典型案例，理解恒星演化与分类要点，开展课程实验。
- 数据科学从业者：将轻量化集成模型应用于结构化科学数据，探索特征工程与可解释性路径。

### 技术栈与主要依赖
- 应用框架：`streamlit`
- 数据/科学计算：`pandas`、`numpy`、`scipy`
- 机器学习：`scikit-learn`、`scikit-optimize`（可选：贝叶斯优化）、`imbalanced-learn`（SMOTE）、`shap`
- 可视化：`plotly`、`matplotlib`、`seaborn`
- 报告与文件：`reportlab`、`joblib`、`zipfile36`
- 其他：`openpyxl`、`psutil`、`requests`、`pydantic`、`python-dotenv`
- 完整依赖列表见仓库根目录 `requirements.txt`。

项目结构概览：
```
StarClassify-Vis/
├─ app.py                  # Streamlit 主入口
├─ requirements.txt        # 依赖清单
├─ data/                   # 示例/缓存数据
└─ starvis/                # 业务模块
   ├─ data_loader.py       # 数据加载与模拟 SDSS
   ├─ preprocessing.py     # 缺失值/异常值/标准化/标签编码
   ├─ features.py          # 特征工程与相关性筛选
   ├─ model.py             # 集成模型训练与预测概率
   ├─ evaluation.py        # 指标计算与混淆矩阵/特征重要性
   ├─ utils.py             # 绘图、案例、规则分类、目录工具
   ├─ testing.py           # 模块测试与端到端验证
   ├─ advanced_features.py # 调优、SMOTE、SHAP、监控、报告
   ├─ batch_operations.py  # 科研包打包与管理
   ├─ model_management.py  # 模型版本/保存/加载/回滚
   ├─ i18n.py              # 国际化与多语言文案
   └─ themes.py            # 主题管理与样式
```

---

## 使用指南

### 环境要求
- 操作系统：macOS
- Python：建议 `>= 3.10`

### 安装步骤
1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/<your-org>/StarClassify-Vis.git
   cd StarClassify-Vis
   ```
2. 创建并启用虚拟环境（示例以 `venv`）：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   # 可选：安装 parquet 缓存支持
   pip install pyarrow
   ```

### 启动应用
- 启动 Streamlit 前端：
  ```bash
  streamlit run app.py
  ```
- 默认在 `http://localhost:8501` 打开；如端口被占用可使用：
  ```bash
  streamlit run app.py --server.port 8502
  ```

### 页面导航与功能速览
- 数据概览：上传 CSV 或使用模拟 SDSS，预览数据与天球分布；自动生成基础特征。
- 模型训练报告：划分训练/测试、训练软投票模型、输出指标、混淆矩阵与特征重要性。
- 科研筛选工具：对新 CSV 批量预测，导出包含类别与置信度的结果。
- 赫罗图科普互动：调节温度/金属丰度，查看规则分类与图上高亮点位与解释。
- 典型案例演示：太阳、天狼星、参宿四等案例的一键分类与分析。
- 测试中心：运行模块级测试与端到端验证，确保系统可用性。
- 批量操作：一键生成包含数据/模型/报告的科研包 ZIP；支持上传解析。
- 模型管理：保存/加载/回滚模型，查看版本与性能指标与大小。
- 数据增强、自动调优、特征分析、训练监控、报告生成：进阶研究工具套件。

---

## API 文档与示例

> 下述 API 来自 `starvis` 包；在 Python 代码中按需导入并使用。

### 数据加载：`starvis.data_loader.DataLoader`
- `load_csv_or_mock(csv_path) -> DataFrame`
- `load_csv(csv_path) -> DataFrame`（含轻量 parquet 缓存，需 `pyarrow`）
- `load_from_buffer(buffer) -> DataFrame`
- `get_info(df) -> Dict[str, float>`
- `generate_mock_sdss(n=10000) -> DataFrame`
示例：
```python
from starvis.data_loader import DataLoader
loader = DataLoader()
df = loader.load_csv_or_mock('data/sdss_mock.csv')
info = loader.get_info(df)
```

### 特征工程：`starvis.features.FeatureEngineer`
- `build_features(df) -> (features_df, feature_names)`
- `plot_corr_heatmap(features_df) -> Figure`
示例：
```python
from starvis.features import FeatureEngineer
engineer = FeatureEngineer(corr_threshold=0.05)
features_df, names = engineer.build_features(df)
```

### 预处理：`starvis.preprocessing.Preprocessor`
- `fill_missing(df) -> DataFrame`
- `remove_outliers_3sigma(df, cols=("redshift","temp")) -> DataFrame`
- `prepare_xy(features_df, raw_df) -> (X, y, label_encoder)`
- `train_test_split(X, y, train_ratio=0.8) -> X_train, X_test, y_train, y_test`
- `transform_features(features_df) -> X_new`
示例：
```python
from starvis.preprocessing import Preprocessor
pre = Preprocessor()
X, y, le = pre.prepare_xy(features_df, df)
X_train, X_test, y_train, y_test = pre.train_test_split(X, y, train_ratio=0.8)
```

### 模型训练：`starvis.model.ModelTrainer`
- `train_voting_classifier(X_train, y_train, weights=(0.6,0.4)) -> model`
- `predict_proba(model, X_new) -> np.ndarray`
示例：
```python
from starvis.model import ModelTrainer
trainer = ModelTrainer()
model = trainer.train_voting_classifier(X_train, y_train)
y_proba = trainer.predict_proba(model, X_test)
```

### 评估分析：`starvis.evaluation.Evaluator`
- `evaluate(model, X_test, y_test) -> (metrics, fig_cm)`
- `feature_importance(model, X_test) -> Figure | None`
示例：
```python
from starvis.evaluation import Evaluator
evalr = Evaluator()
metrics, fig_cm = evalr.evaluate(model, X_test, y_test)
```

### 通用工具与科普：`starvis.utils`
- `ensure_data_dir(path='data') -> None`
- `plot_sky_distribution(df) -> PlotlyFigure`
- `plot_hr_diagram(df, features_df, highlight_point=False, temp=6000) -> PlotlyFigure`
- `classify_by_rules(temp, feh) -> str`
- `explain_star_class(cls) -> (explain_text, img_url)`
- `build_case_dataset() -> DataFrame`

### 测试与端到端验证：`starvis.testing`
- `test_*_api(...)`：模块级测试入口
- `run_pipeline_quick(loader, engineer, preprocessor, trainer, evaluator) -> dict`

### 高级功能集：`starvis.advanced_features.AdvancedFeatures`
- `hyperparameter_tuning(model, X, y, param_grid, method='grid'|'random'|'bayesian', cv=5) -> dict`
- `apply_smote(X, y, sampling_strategy='auto', k_neighbors=5) -> (X_res, y_res, stats)`
- `shap_analysis(model, X, feature_names=None, sample_size=100) -> dict`
- `advanced_filtering(df, conditions, save_scheme=False, scheme_name=None) -> (filtered_df, stats)`
- `training_monitor(model, X_train, y_train, X_val, y_val, epochs=100, monitor_interval=1.0) -> dict`
- `generate_academic_report(title, author, abstract, content, template='academic_poster') -> bytes`

UI 辅助：
- `render_advanced_filter_ui(df) -> Optional[DataFrame]`

### 科研包与模型管理
- `starvis.batch_operations.BatchOperations`：`create_research_package`、`extract_research_package`、`list_packages`、`delete_package`
- 便捷函数：`handle_package_upload()`、`handle_package_download(...)`
- `starvis.model_management.ModelManager`：`save_model`、`load_model`、`rollback_model`、`delete_model`、`list_models`

### 端到端使用示例（脚本）
```python
from starvis.data_loader import DataLoader
from starvis.features import FeatureEngineer
from starvis.preprocessing import Preprocessor
from starvis.model import ModelTrainer
from starvis.evaluation import Evaluator

loader = DataLoader()
df = loader.load_csv_or_mock('data/sdss_mock.csv')
engineer = FeatureEngineer()
features_df, names = engineer.build_features(df)
pre = Preprocessor()
X, y, le = pre.prepare_xy(features_df, df)
X_train, X_test, y_train, y_test = pre.train_test_split(X, y, train_ratio=0.8)
trainer = ModelTrainer()
model = trainer.train_voting_classifier(X_train, y_train)
from sklearn.metrics import classification_report
import numpy as np
print(classification_report(y_test, model.predict(X_test)))
```

---

## 常见问题与故障排除

- Streamlit 无法启动或端口占用：
  - 解决：使用 `--server.port` 指定新端口；或关闭占用进程。
- `parquet` 缓存报错：
  - 现象：`to_parquet`/`read_parquet` 报缺少依赖。
  - 解决：安装 `pyarrow`；或忽略缓存（模块已自动降级为 CSV 读取）。
- 安装 `shap`/`imbalanced-learn` 失败：
  - 解决：升级 `pip` 与 `setuptools`，或使用 Conda 环境；若无需该功能可暂时跳过。
- 图形后端问题（macOS）：
  - 现象：`matplotlib` 显示异常。
  - 解决：确保在虚拟环境中运行 Streamlit；避免混用旧版后端。
- 内存占用高或运行慢：
  - 解决：减少样本量（改用采样数据）、下调 `n_iter` 或 `cv`、关闭不必要的页面图表。

---

## 未来规划

### 短期（1–2 个月）
- 增强特征重要性解释与可视化联动（Plotly/SHAP 联动）。
- 完善科研包格式与元数据协议，增加签名校验。
- 扩展国际化词库与主题系统，支持自定义皮肤。
- 增加更系统化的单元测试与 CI。

### 中长期（3–6 个月）
- 引入更强模型（如梯度提升/浅层深度模型）与可解释接口。
- 支持 SDSS 在线接口/更多天文数据源、数据版本管理与审计。
- 提供 Web Service/API 模式（后端服务化，支持批量任务队列）。
- 插件化架构（筛选器/特征器/评估器可插拔）。

---

## 社区参与与贡献

- 分支与提交流程：
  - Fork 仓库 → 创建主题分支 → 提交 PR。
- 代码规范与约定：
  - Python 代码保持函数级 Docstring；遵循现有模块风格与依赖选择；避免提交密钥与数据隐私。
- 测试与验证：
  - 变更请至少运行“测试中心”页面的模块测试；建议附带最小可复现实验。
- 国际化与文案：
  - UI 文案请同步维护 `starvis/i18n.py` 的多语言键值。

---

## 许可证

- 建议许可证：MIT License（可自由使用、修改与分发，需保留原作者声明）。
- 如需变更许可证，请在仓库根目录添加/更新 `LICENSE` 文件并同步本章节说明。

---

## 联系方式与支持

- 问题反馈：建议通过仓库 Issue 提交。
- 业务合作与咨询：请在 Issue 中说明需求，维护者将跟进。

---

## 相关资源与参考链接
- SDSS 官方站点与文档：https://www.sdss.org/
- Streamlit 文档：https://docs.streamlit.io/
- scikit-learn 文档：https://scikit-learn.org/stable/
- imbalanced-learn 文档：https://imbalanced-learn.org/
- SHAP 文档：https://shap.readthedocs.io/
- Plotly 文档：https://plotly.com/python/
- ReportLab 文档：https://www.reportlab.com/documentation/

---

> 致谢：本项目以教学示例为目标，数据生成与模型选择均为近似与轻量化设计，便于快速入门与演示。如用于正式科研，请结合真实数据与更严格的实验流程进行验证与扩展。