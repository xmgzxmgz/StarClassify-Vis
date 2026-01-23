# 页面设计文档（桌面优先）

## 0. 全局规范（适用于全部页面）
### Layout
- 桌面优先：以 1200px 内容容器为基准（`max-width: 1200px; margin: 0 auto; padding: 24px`）。
- 主布局：顶部导航 + 内容区（单列为主，局部用 CSS Grid 进行两栏/多卡片布局）。
- 响应式：
  - ≥1024px：两栏（配置/结果）或卡片网格。
  - <1024px：改为单列堆叠；图表与表格可横向滚动。

### Meta Information（默认）
- title：星体分类可视化实验台
- description：基于高斯朴素贝叶斯的训练/预测与结果可视化，并可将实验记录存储到数据库。
- Open Graph：`og:title`、`og:description` 与 title/description 一致；`og:type=website`。

### Global Styles（设计令牌建议）
- 背景：`#0B1220`（深色底） / 内容卡片：`#111A2E`
- 主色：`#3B82F6`（按钮/高亮）
- 成功/警告/错误：`#22C55E / #F59E0B / #EF4444`
- 字体：系统字体栈；标题 20/24/32，正文 14/16
- 按钮：
  - Primary：主色填充，hover 加深；disabled 降低不透明度并禁用交互
  - Secondary：描边按钮
- 交互：所有可点击元素 hover 提示（颜色/阴影/下划线）；加载态统一用 skeleton 或 spinner。

### 组件化约束（用于“风格统一并确保可用”）
- 统一组件：AppHeader、Card、FormRow、PrimaryButton、SecondaryButton、Toast、EmptyState、ErrorBanner。
- 表单校验错误：字段下方红色提示 + 顶部 ErrorBanner 汇总。

---

## 1. 实验台页面（/）
### Meta
- title：实验台｜星体分类可视化实验台
- description：上传数据并使用高斯朴素贝叶斯进行训练/预测与可视化。

### Page Structure
- 桌面端两栏：左侧“配置区”，右侧“结果区”；顶部固定导航。
- Grid：`grid-template-columns: 420px 1fr; gap: 20px;`

### Sections & Components
1) 顶部导航（AppHeader）
- 左：产品名
- 右：导航链接（实验台、结果记录）

2) 配置区（Card Stack）
- Card：数据导入
  - FileUploader：CSV 上传
  - 数据概览：字段数量、行数、缺失值提示（最小可用）
- Card：字段选择
  - Select：目标列
  - MultiSelect：特征列
- Card：模型配置（仅 GaussianNB）
  - NumberInput：`var_smoothing`
  - 文案提示：本版本已移除逻辑回归
- Card：执行区
  - Slider/NumberInput：test_size
  - NumberInput：random_state（可选）
  - PrimaryButton：开始训练/预测
  - SecondaryButton：保存结果（仅在成功计算后可用）

3) 结果区（Card）
- 状态展示：
  - 初始：EmptyState（提示先上传并配置）
  - 运行中：Loading（禁用提交，显示进度文案）
  - 失败：ErrorBanner（展示可读错误）
- 成功后：
  - 指标卡片（4个小卡）：Accuracy/Precision/Recall/F1
  - 混淆矩阵：表格或热力图（二选一即可，优先表格保证可用）
  - 类别分布：简单柱状图或表格（以可用为先）
  - 保存成功后：Toast + “去结果记录查看”链接

---

## 2. 结果记录页面（/runs）
### Meta
- title：结果记录｜星体分类可视化实验台
- description：浏览历史实验记录并查看详情、复现实验配置。

### Page Structure
- 单列为主，上方筛选，下面列表；点击行/按钮打开详情（右侧抽屉或弹窗）。

### Sections & Components
1) 顶部导航（AppHeader）
- 与实验台一致，保证一致性。

2) 筛选区（Card）
- Input：按数据集名关键字筛选
- SecondaryButton：重置

3) 列表区（Card）
- Table：列包含时间、数据集名、目标列、Accuracy、F1、操作
- Pagination：上一页/下一页
- 空状态：EmptyState

4) 详情查看（Drawer/Modal）
- 概览：核心指标、参数摘要
- 混淆矩阵：表格
- PrimaryButton：加载到实验台（将字段选择+参数带回）
- SecondaryButton：关闭
