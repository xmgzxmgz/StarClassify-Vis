"""
国际化支持模块（Internationalization）

职责：
- 提供中英文双语支持
- 动态语言切换
- 统一的翻译管理
- 支持扩展其他语言

物理意义：
通过国际化支持，让应用能够服务于全球用户，提升可用性和专业性。
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any, Optional


class I18nManager:
    """国际化管理器，负责语言切换和翻译管理。"""
    
    # 翻译字典
    TRANSLATIONS = {
        "zh": {
            # 通用
            "app_title": "StarClassify-Vis：基于 SDSS 的恒星分类与科普可视化",
            "app_subtitle": "轻量化集成模型（逻辑回归 + 朴素贝叶斯软投票），双模式可视化（科研/科普）",
            "loading": "加载中...",
            "success": "成功",
            "error": "错误",
            "warning": "警告",
            "info": "信息",
            "confirm": "确认",
            "cancel": "取消",
            "save": "保存",
            "download": "下载",
            "upload": "上传",
            "delete": "删除",
            "edit": "编辑",
            "view": "查看",
            "search": "搜索",
            "filter": "筛选",
            "reset": "重置",
            "apply": "应用",
            "run": "运行",
            "stop": "停止",
            "refresh": "刷新",
            "export": "导出",
            "import": "导入",
            "settings": "设置",
            "help": "帮助",
            "about": "关于",
            "close": "关闭",
            "back": "返回",
            "next": "下一步",
            "previous": "上一步",
            "finish": "完成",
            "start": "开始",
            "pause": "暂停",
            "resume": "继续",
            "retry": "重试",
            "skip": "跳过",
            "yes": "是",
            "no": "否",
            "ok": "确定",
            
            # 导航
            "nav_data_overview": "数据概览",
            "nav_model_training": "模型训练报告",
            "nav_research_filter": "科研筛选工具",
            "nav_hr_interactive": "赫罗图科普互动",
            "nav_case_studies": "典型案例演示",
            "nav_testing_center": "测试中心",
            "nav_batch_operations": "批量操作",
            "nav_theme_settings": "主题设置",
            "nav_language_settings": "语言设置",
            "nav_model_management": "模型管理",
            "nav_data_augmentation": "数据增强",
            "nav_advanced_filter": "高级筛选",
            "nav_auto_tuning": "自动调优",
            "nav_feature_analysis": "特征分析",
            "nav_training_monitor": "训练监控",
            "nav_report_generator": "报告生成",
            
            # 数据概览
            "data_overview_title": "数据概览",
            "data_upload_placeholder": "上传 SDSS CSV（可选）",
            "load_data_button": "加载 SDSS 数据",
            "data_load_success": "成功加载数据：{rows} 行、{cols} 列，内存约 {memory_mb:.2f} MB",
            "data_preview": "前 10 行预览：",
            "sky_distribution": "天球空间分布（采样）",
            "features_generated": "已生成颜色指数与空间特征，用于后续训练与分析。",
            "data_load_error": "数据加载失败：{error}",
            
            # 模型训练
            "model_training_title": "模型训练报告",
            "train_ratio_config": "训练集占比",
            "start_training_button": "开始训练",
            "training_progress": "训练进度",
            "training_complete": "训练完成！准确率 {accuracy:.2%}，精确率 {precision:.2%}，召回率 {recall:.2%}，F1 {f1:.2%}",
            "feature_importance": "特征重要性（逻辑回归系数/排列重要性）",
            "feature_importance_info": "从结果可见：红移与 g-r 颜色等物理参数对分类贡献显著。",
            "training_error": "训练过程出现错误：{error}",
            "load_data_first": "请先在'数据概览'页面加载数据并生成基础特征。",
            
            # 科研筛选
            "research_filter_title": "科研筛选工具",
            "upload_for_prediction": "上传待分类 CSV",
            "batch_predict_button": "批量预测",
            "prediction_complete": "批量预测完成。",
            "download_predictions": "下载预测结果 CSV",
            "high_confidence": "高置信度",
            "needs_review": "需人工复核",
            "prediction_error": "批量预测失败：{error}",
            "model_not_trained": "请先在'模型训练报告'页面完成训练。",
            
            # 赫罗图互动
            "hr_interactive_title": "赫罗图科普互动",
            "effective_temperature": "有效温度 (K)",
            "metallicity": "金属丰度 [Fe/H]",
            "star_classification": "判定类型：{classification}",
            "red_giant": "红巨星",
            "white_dwarf": "白矮星",
            "main_sequence": "主序星",
            "hr_diagram": "赫罗图（颜色指数 vs r 星等，采样）",
            
            # 案例演示
            "case_studies_title": "典型案例演示",
            "load_cases_button": "一键加载案例",
            "case_results": "示例结果表明：红移与 g-r 颜色是分类关键要素之一。",
            "case_error": "案例演示失败：{error}",
            
            # 测试中心
            "testing_center_title": "测试中心：一键验证各模块与端到端流程",
            "run_all_tests": "运行所有模块测试",
            "run_e2e_test": "一键端到端验证",
            "data_loader_test": "数据加载模块测试",
            "feature_engineer_test": "特征工程模块测试",
            "preprocessor_test": "预处理模块测试",
            "model_trainer_test": "模型训练模块测试",
            "evaluator_test": "评估模块测试",
            "all_tests_passed": "模块级测试全部通过 ✅",
            "e2e_test_passed": "端到端流程验证通过 ✅",
            "test_failed": "测试运行失败：{error}",
            "e2e_test_failed": "端到端流程失败：{error}",
            
            # 主题设置
            "theme_settings_title": "主题设置",
            "select_theme": "选择主题",
            "theme_applied": "主题已应用：{theme_name}",
            
            # 语言设置
            "language_settings_title": "语言设置",
            "select_language": "选择语言",
            "language_applied": "语言已切换至：{language_name}",
            
            # 批量操作
            "batch_operations_title": "批量操作",
            "upload_zip": "上传 ZIP 科研包",
            "download_zip": "下载 ZIP 科研包",
            "package_contents": "科研包内容",
            "data_files": "数据文件",
            "model_files": "模型文件",
            "report_files": "报告文件",
            "package_created": "科研包创建成功！",
            "package_error": "科研包处理失败：{error}",
            
            # 模型管理
            "model_management_title": "模型管理",
            "save_model": "保存模型",
            "load_model": "加载模型",
            "model_list": "模型列表",
            "model_version": "版本",
            "model_created": "创建时间",
            "model_accuracy": "准确率",
            "model_size": "大小",
            "model_status": "状态",
            "model_actions": "操作",
            "rollback_model": "回滚",
            "delete_model": "删除",
            "model_saved": "模型保存成功！",
            "model_loaded": "模型加载成功！",
            "model_deleted": "模型删除成功！",
            "model_rollback": "模型回滚成功！",
            
            # 数据增强
            "data_augmentation_title": "数据增强",
            "smote_config": "SMOTE 配置",
            "apply_smote": "应用 SMOTE",
            "original_distribution": "原始类别分布",
            "augmented_distribution": "增强后类别分布",
            "smote_applied": "SMOTE 应用成功！",
            "smote_error": "SMOTE 应用失败：{error}",
            
            # 高级筛选
            "advanced_filter_title": "高级筛选",
            "filter_conditions": "筛选条件",
            "add_condition": "添加条件",
            "save_filter_scheme": "保存筛选方案",
            "load_filter_scheme": "加载筛选方案",
            "apply_filter": "应用筛选",
            "clear_filter": "清除筛选",
            "filter_scheme_name": "方案名称",
            "filter_applied": "筛选应用成功！",
            "filter_saved": "筛选方案保存成功！",
            "filter_loaded": "筛选方案加载成功！",
            "filter_error": "筛选应用失败：{error}",
            
            # 自动调优
            "auto_tuning_title": "超参数自动调优",
            "tuning_method": "调优方法",
            "grid_search": "网格搜索",
            "random_search": "随机搜索",
            "bayesian_optimization": "贝叶斯优化",
            "param_grid": "参数网格",
            "cv_folds": "交叉验证折数",
            "n_jobs": "并行作业数",
            "start_tuning": "开始调优",
            "tuning_progress": "调优进度",
            "best_params": "最优参数",
            "best_score": "最优得分",
            "tuning_complete": "调优完成！",
            "tuning_error": "调优失败：{error}",
            
            # 特征分析
            "feature_analysis_title": "特征重要性分析",
            "shap_analysis": "SHAP 分析",
            "feature_importance_plot": "特征重要性图",
            "shap_summary": "SHAP 摘要图",
            "shap_waterfall": "SHAP 瀑布图",
            "generate_shap": "生成 SHAP 分析",
            "shap_generated": "SHAP 分析生成成功！",
            "shap_error": "SHAP 分析失败：{error}",
            
            # 训练监控
            "training_monitor_title": "训练过程监控",
            "real_time_progress": "实时进度",
            "training_logs": "训练日志",
            "epoch_progress": "轮次进度",
            "loss_curve": "损失曲线",
            "accuracy_curve": "准确率曲线",
            "start_monitoring": "开始监控",
            "stop_monitoring": "停止监控",
            "monitoring_started": "训练监控已启动！",
            "monitoring_stopped": "训练监控已停止！",
            
            # 报告生成
            "report_generator_title": "学术报告生成",
            "report_template": "报告模板",
            "academic_poster": "学术海报",
            "technical_report": "技术报告",
            "custom_report": "自定义报告",
            "report_title": "报告标题",
            "report_author": "作者",
            "report_abstract": "摘要",
            "generate_report": "生成报告",
            "download_pdf": "下载 PDF",
            "report_generated": "报告生成成功！",
            "report_error": "报告生成失败：{error}",
            
            # 状态消息
            "status_ready": "就绪",
            "status_running": "运行中",
            "status_completed": "已完成",
            "status_failed": "失败",
            "status_warning": "警告",
            "status_info": "信息",
            
            # 单位
            "unit_temperature": "K",
            "unit_metallicity": "[Fe/H]",
            "unit_degrees": "度",
            "unit_magnitude": "星等",
            "unit_percentage": "%",
            "unit_mb": "MB",
            "unit_gb": "GB",
            "unit_seconds": "秒",
            "unit_minutes": "分钟",
            
            # 错误信息
            "error_file_not_found": "文件未找到：{filename}",
            "error_invalid_format": "无效的文件格式",
            "error_data_empty": "数据为空",
            "error_model_not_found": "模型未找到",
            "error_invalid_parameter": "无效参数：{param}",
            "error_system_error": "系统错误：{error}",
            "error_network_error": "网络错误：{error}",
            "error_permission_denied": "权限被拒绝：{error}",
            
            # 成功信息
            "success_operation": "操作成功完成",
            "success_save": "保存成功",
            "success_load": "加载成功",
            "success_delete": "删除成功",
            "success_update": "更新成功",
            "success_create": "创建成功",
            
            # 警告信息
            "warning_confirm_delete": "确认删除？此操作不可撤销。",
            "warning_large_file": "文件较大，处理可能需要一些时间。",
            "warning_unsaved_changes": "有未保存的更改，是否继续？",
            "warning_overwrite": "文件已存在，是否覆盖？",
            
            # 帮助信息
            "help_upload_format": "支持的格式：CSV, JSON, ZIP",
            "help_model_training": "训练过程可能需要几分钟，请耐心等待。",
            "help_feature_selection": "选择相关性高的特征可以提升模型性能。",
            "help_data_preprocessing": "良好的数据预处理是模型成功的关键。",
            "help_hyperparameter_tuning": "调优过程会自动搜索最佳参数组合。",
            
            # 工具提示
            "tooltip_save_model": "保存当前模型到本地",
            "tooltip_load_model": "从本地加载已保存的模型",
            "tooltip_delete_model": "删除选中的模型",
            "tooltip_rollback_model": "回滚到选中的模型版本",
            "tooltip_apply_smote": "使用SMOTE算法处理类别不平衡",
            "tooltip_generate_shap": "生成SHAP值用于模型解释",
            "tooltip_start_monitoring": "开始实时监控训练过程",
            "tooltip_generate_report": "生成PDF格式的学术报告",
            "tooltip_download_zip": "下载包含数据和模型的ZIP包",
            "tooltip_upload_zip": "上传ZIP格式的科研数据包"
        },
        "en": {
            # Common
            "app_title": "StarClassify-Vis: SDSS-based Star Classification & Visualization",
            "app_subtitle": "Lightweight ensemble model (Logistic Regression + Naive Bayes soft voting), dual-mode visualization (research/popular science)",
            "loading": "Loading...",
            "success": "Success",
            "error": "Error",
            "warning": "Warning",
            "info": "Information",
            "confirm": "Confirm",
            "cancel": "Cancel",
            "save": "Save",
            "download": "Download",
            "upload": "Upload",
            "delete": "Delete",
            "edit": "Edit",
            "view": "View",
            "search": "Search",
            "filter": "Filter",
            "reset": "Reset",
            "apply": "Apply",
            "run": "Run",
            "stop": "Stop",
            "refresh": "Refresh",
            "export": "Export",
            "import": "Import",
            "settings": "Settings",
            "help": "Help",
            "about": "About",
            "close": "Close",
            "back": "Back",
            "next": "Next",
            "previous": "Previous",
            "finish": "Finish",
            "start": "Start",
            "pause": "Pause",
            "resume": "Resume",
            "retry": "Retry",
            "skip": "Skip",
            "yes": "Yes",
            "no": "No",
            "ok": "OK",
            
            # Navigation
            "nav_data_overview": "Data Overview",
            "nav_model_training": "Model Training Report",
            "nav_research_filter": "Research Filter Tool",
            "nav_hr_interactive": "HR Diagram Interactive",
            "nav_case_studies": "Case Studies",
            "nav_testing_center": "Testing Center",
            "nav_batch_operations": "Batch Operations",
            "nav_theme_settings": "Theme Settings",
            "nav_language_settings": "Language Settings",
            "nav_model_management": "Model Management",
            "nav_data_augmentation": "Data Augmentation",
            "nav_advanced_filter": "Advanced Filter",
            "nav_auto_tuning": "Auto Tuning",
            "nav_feature_analysis": "Feature Analysis",
            "nav_training_monitor": "Training Monitor",
            "nav_report_generator": "Report Generator",
            
            # Data Overview
            "data_overview_title": "Data Overview",
            "data_upload_placeholder": "Upload SDSS CSV (optional)",
            "load_data_button": "Load SDSS Data",
            "data_load_success": "Data loaded successfully: {rows} rows, {cols} columns, approximately {memory_mb:.2f} MB",
            "data_preview": "First 10 rows preview:",
            "sky_distribution": "Sky Distribution (Sample)",
            "features_generated": "Color indices and spatial features generated for subsequent training and analysis.",
            "data_load_error": "Data loading failed: {error}",
            
            # Model Training
            "model_training_title": "Model Training Report",
            "train_ratio_config": "Training Set Ratio",
            "start_training_button": "Start Training",
            "training_progress": "Training Progress",
            "training_complete": "Training completed! Accuracy {accuracy:.2%}, Precision {precision:.2%}, Recall {recall:.2%}, F1 {f1:.2%}",
            "feature_importance": "Feature Importance (Logistic Regression Coefficients/Permutation Importance)",
            "feature_importance_info": "Results show: redshift and g-r color are significant contributors to classification.",
            "training_error": "Training error: {error}",
            "load_data_first": "Please load data and generate basic features in the 'Data Overview' page first.",
            
            # Research Filter
            "research_filter_title": "Research Filter Tool",
            "upload_for_prediction": "Upload CSV for Classification",
            "batch_predict_button": "Batch Predict",
            "prediction_complete": "Batch prediction completed.",
            "download_predictions": "Download Prediction Results CSV",
            "high_confidence": "High Confidence",
            "needs_review": "Manual Review Required",
            "prediction_error": "Batch prediction failed: {error}",
            "model_not_trained": "Please complete model training in the 'Model Training Report' page first.",
            
            # HR Interactive
            "hr_interactive_title": "HR Diagram Interactive",
            "effective_temperature": "Effective Temperature (K)",
            "metallicity": "Metallicity [Fe/H]",
            "star_classification": "Classification: {classification}",
            "red_giant": "Red Giant",
            "white_dwarf": "White Dwarf",
            "main_sequence": "Main Sequence",
            "hr_diagram": "HR Diagram (Color Index vs r Magnitude, Sample)",
            
            # Case Studies
            "case_studies_title": "Case Studies",
            "load_cases_button": "Load Sample Cases",
            "case_results": "Results show: redshift and g-r color are key classification features.",
            "case_error": "Case study failed: {error}",
            
            # Testing Center
            "testing_center_title": "Testing Center: One-click Verification of Modules and End-to-end Process",
            "run_all_tests": "Run All Module Tests",
            "run_e2e_test": "One-click End-to-end Verification",
            "data_loader_test": "Data Loader Module Test",
            "feature_engineer_test": "Feature Engineer Module Test",
            "preprocessor_test": "Preprocessor Module Test",
            "model_trainer_test": "Model Trainer Module Test",
            "evaluator_test": "Evaluator Module Test",
            "all_tests_passed": "All module tests passed ✅",
            "e2e_test_passed": "End-to-end process verification passed ✅",
            "test_failed": "Test execution failed: {error}",
            "e2e_test_failed": "End-to-end process failed: {error}",
            
            # Theme Settings
            "theme_settings_title": "Theme Settings",
            "select_theme": "Select Theme",
            "theme_applied": "Theme applied: {theme_name}",
            
            # Language Settings
            "language_settings_title": "Language Settings",
            "select_language": "Select Language",
            "language_applied": "Language switched to: {language_name}",
            
            # Batch Operations
            "batch_operations_title": "Batch Operations",
            "upload_zip": "Upload Research Package ZIP",
            "download_zip": "Download Research Package ZIP",
            "package_contents": "Package Contents",
            "data_files": "Data Files",
            "model_files": "Model Files",
            "report_files": "Report Files",
            "package_created": "Research package created successfully!",
            "package_error": "Research package processing failed: {error}",
            
            # Model Management
            "model_management_title": "Model Management",
            "save_model": "Save Model",
            "load_model": "Load Model",
            "model_list": "Model List",
            "model_version": "Version",
            "model_created": "Created",
            "model_accuracy": "Accuracy",
            "model_size": "Size",
            "model_status": "Status",
            "model_actions": "Actions",
            "rollback_model": "Rollback",
            "delete_model": "Delete",
            "model_saved": "Model saved successfully!",
            "model_loaded": "Model loaded successfully!",
            "model_deleted": "Model deleted successfully!",
            "model_rollback": "Model rollback successful!",
            
            # Data Augmentation
            "data_augmentation_title": "Data Augmentation",
            "smote_config": "SMOTE Configuration",
            "apply_smote": "Apply SMOTE",
            "original_distribution": "Original Class Distribution",
            "augmented_distribution": "Augmented Class Distribution",
            "smote_applied": "SMOTE applied successfully!",
            "smote_error": "SMOTE application failed: {error}",
            
            # Advanced Filter
            "advanced_filter_title": "Advanced Filter",
            "filter_conditions": "Filter Conditions",
            "add_condition": "Add Condition",
            "save_filter_scheme": "Save Filter Scheme",
            "load_filter_scheme": "Load Filter Scheme",
            "apply_filter": "Apply Filter",
            "clear_filter": "Clear Filter",
            "filter_scheme_name": "Scheme Name",
            "filter_applied": "Filter applied successfully!",
            "filter_saved": "Filter scheme saved successfully!",
            "filter_loaded": "Filter scheme loaded successfully!",
            "filter_error": "Filter application failed: {error}",
            
            # Auto Tuning
            "auto_tuning_title": "Hyperparameter Auto Tuning",
            "tuning_method": "Tuning Method",
            "grid_search": "Grid Search",
            "random_search": "Random Search",
            "bayesian_optimization": "Bayesian Optimization",
            "param_grid": "Parameter Grid",
            "cv_folds": "Cross-validation Folds",
            "n_jobs": "Parallel Jobs",
            "start_tuning": "Start Tuning",
            "tuning_progress": "Tuning Progress",
            "best_params": "Best Parameters",
            "best_score": "Best Score",
            "tuning_complete": "Tuning completed!",
            "tuning_error": "Tuning failed: {error}",
            
            # Feature Analysis
            "feature_analysis_title": "Feature Importance Analysis",
            "shap_analysis": "SHAP Analysis",
            "feature_importance_plot": "Feature Importance Plot",
            "shap_summary": "SHAP Summary Plot",
            "shap_waterfall": "SHAP Waterfall Plot",
            "generate_shap": "Generate SHAP Analysis",
            "shap_generated": "SHAP analysis generated successfully!",
            "shap_error": "SHAP analysis failed: {error}",
            
            # Training Monitor
            "training_monitor_title": "Training Process Monitor",
            "real_time_progress": "Real-time Progress",
            "training_logs": "Training Logs",
            "epoch_progress": "Epoch Progress",
            "loss_curve": "Loss Curve",
            "accuracy_curve": "Accuracy Curve",
            "start_monitoring": "Start Monitoring",
            "stop_monitoring": "Stop Monitoring",
            "monitoring_started": "Training monitoring started!",
            "monitoring_stopped": "Training monitoring stopped!",
            
            # Report Generator
            "report_generator_title": "Academic Report Generator",
            "report_template": "Report Template",
            "academic_poster": "Academic Poster",
            "technical_report": "Technical Report",
            "custom_report": "Custom Report",
            "report_title": "Report Title",
            "report_author": "Author",
            "report_abstract": "Abstract",
            "generate_report": "Generate Report",
            "download_pdf": "Download PDF",
            "report_generated": "Report generated successfully!",
            "report_error": "Report generation failed: {error}",
            
            # Status
            "status_ready": "Ready",
            "status_running": "Running",
            "status_completed": "Completed",
            "status_failed": "Failed",
            "status_warning": "Warning",
            "status_info": "Information",
            
            # Units
            "unit_temperature": "K",
            "unit_metallicity": "[Fe/H]",
            "unit_degrees": "degrees",
            "unit_magnitude": "magnitude",
            "unit_percentage": "%",
            "unit_mb": "MB",
            "unit_gb": "GB",
            "unit_seconds": "seconds",
            "unit_minutes": "minutes",
            
            # Error messages
            "error_file_not_found": "File not found: {filename}",
            "error_invalid_format": "Invalid file format",
            "error_data_empty": "Data is empty",
            "error_model_not_found": "Model not found",
            "error_invalid_parameter": "Invalid parameter: {param}",
            "error_system_error": "System error: {error}",
            "error_network_error": "Network error: {error}",
            "error_permission_denied": "Permission denied: {error}",
            
            # Success messages
            "success_operation": "Operation completed successfully",
            "success_save": "Save successful",
            "success_load": "Load successful",
            "success_delete": "Delete successful",
            "success_update": "Update successful",
            "success_create": "Create successful",
            
            # Warning messages
            "warning_confirm_delete": "Confirm deletion? This action cannot be undone.",
            "warning_large_file": "Large file, processing may take some time.",
            "warning_unsaved_changes": "There are unsaved changes, continue?",
            "warning_overwrite": "File already exists, overwrite?",
            
            # Help information
            "help_upload_format": "Supported formats: CSV, JSON, ZIP",
            "help_model_training": "Training process may take a few minutes, please be patient.",
            "help_feature_selection": "Selecting highly correlated features can improve model performance.",
            "help_data_preprocessing": "Good data preprocessing is key to model success.",
            "help_hyperparameter_tuning": "Tuning process will automatically search for optimal parameter combinations.",
            
            # Tooltips
            "tooltip_save_model": "Save current model to local",
            "tooltip_load_model": "Load saved model from local",
            "tooltip_delete_model": "Delete selected model",
            "tooltip_rollback_model": "Rollback to selected model version",
            "tooltip_apply_smote": "Use SMOTE algorithm to handle class imbalance",
            "tooltip_generate_shap": "Generate SHAP values for model interpretation",
            "tooltip_start_monitoring": "Start real-time training monitoring",
            "tooltip_generate_report": "Generate PDF format academic report",
            "tooltip_download_zip": "Download ZIP package containing data and model",
            "tooltip_upload_zip": "Upload ZIP format research data package"
        }
    }
    
    def __init__(self, default_language: str = "zh"):
        """初始化国际化管理器。
        
        参数：
            default_language: 默认语言 ('zh' 或 'en')
        """
        self.current_language = default_language
        self._validate_language(default_language)
    
    def set_language(self, language: str) -> None:
        """设置当前语言。
        
        参数：
            language: 语言代码 ('zh' 或 'en')
        """
        self._validate_language(language)
        self.current_language = language
        
        # 保存到session state
        if "i18n_manager" not in st.session_state:
            st.session_state.i18n_manager = self
    
    def get_text(self, key: str, **kwargs) -> str:
        """获取翻译文本。
        
        参数：
            key: 翻译键
            **kwargs: 格式化参数
        
        返回：
            str: 翻译后的文本
        """
        translation = self.TRANSLATIONS[self.current_language].get(key, key)
        
        if kwargs:
            try:
                return translation.format(**kwargs)
            except KeyError:
                # 如果格式化失败，返回原始文本
                return translation
        
        return translation
    
    def get_available_languages(self) -> Dict[str, str]:
        """获取可用语言列表。
        
        返回：
            Dict[str, str]: 语言代码和名称的映射
        """
        return {
            "zh": "中文 (Chinese)",
            "en": "English"
        }
    
    def get_current_language_name(self) -> str:
        """获取当前语言的显示名称。
        
        返回：
            str: 当前语言的显示名称
        """
        return self.get_available_languages()[self.current_language]
    
    def _validate_language(self, language: str) -> None:
        """验证语言代码是否有效。
        
        参数：
            language: 语言代码
        
        抛出：
            ValueError: 如果语言代码无效
        """
        if language not in self.TRANSLATIONS:
            available = list(self.TRANSLATIONS.keys())
            raise ValueError(f"语言 '{language}' 不支持。可用语言: {available}")
    
    def add_translation(self, language: str, translations: Dict[str, str]) -> None:
        """添加或更新翻译。
        
        参数：
            language: 语言代码
            translations: 翻译字典
        """
        if language not in self.TRANSLATIONS:
            self.TRANSLATIONS[language] = {}
        
        self.TRANSLATIONS[language].update(translations)
    
    def get_missing_translations(self, base_language: str = "zh") -> Dict[str, list]:
        """获取缺失的翻译。
        
        参数：
            base_language: 基准语言
        
        返回：
            Dict[str, list]: 每种语言缺失的翻译键列表
        """
        base_keys = set(self.TRANSLATIONS[base_language].keys())
        missing = {}
        
        for language, translations in self.TRANSLATIONS.items():
            if language == base_language:
                continue
            
            language_keys = set(translations.keys())
            missing_keys = base_keys - language_keys
            
            if missing_keys:
                missing[language] = list(missing_keys)
        
        return missing


# 全局国际化管理器实例
def get_i18n_manager() -> I18nManager:
    """获取全局国际化管理器实例。
    
    返回：
        I18nManager: 国际化管理器实例
    """
    if "i18n_manager" not in st.session_state:
        st.session_state.i18n_manager = I18nManager()
    return st.session_state.i18n_manager


def _(key: str, **kwargs) -> str:
    """翻译函数简写。
    
    参数：
        key: 翻译键
        **kwargs: 格式化参数
    
    返回：
        str: 翻译后的文本
    
    使用示例：
        st.write(_("app_title"))
        st.write(_("data_load_success", rows=1000, cols=20, memory_mb=15.5))
    """
    manager = get_i18n_manager()
    return manager.get_text(key, **kwargs)


def set_language(language: str) -> None:
    """设置应用语言。
    
    参数：
        language: 语言代码 ('zh' 或 'en')
    """
    manager = get_i18n_manager()
    manager.set_language(language)


def get_current_language() -> str:
    """获取当前语言代码。
    
    返回：
        str: 当前语言代码
    """
    manager = get_i18n_manager()
    return manager.current_language


def get_language_name(language: str) -> str:
    """获取语言显示名称。
    
    参数：
        language: 语言代码
    
    返回：
        str: 语言显示名称
    """
    manager = get_i18n_manager()
    return manager.get_available_languages().get(language, language)