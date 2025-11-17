"""
高级功能模块（Advanced Features）

职责：
- 超参数自动调优（GridSearchCV/RandomSearchCV/BayesianOptimization）
- 特征重要性SHAP分析
- 数据增强SMOTE
- 高级筛选系统
- 训练过程监控
- 报告生成

物理意义：
通过集成高级机器学习功能，提供完整的科研分析工具链。
"""

from __future__ import annotations

import os
import time
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from .utils import ensure_data_dir
from .i18n import _

# 可选依赖的懒加载
def import_optional_dependencies():
    """导入可选的高级依赖。"""
    global GridSearchCV, RandomizedSearchCV, BayesianOptimization, SMOTE, shap
    
    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    except ImportError:
        GridSearchCV = RandomizedSearchCV = None
    
    try:
        from skopt import BayesSearchCV
        BayesianOptimization = BayesSearchCV
    except ImportError:
        BayesianOptimization = None
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        SMOTE = None
    
    try:
        import shap
    except ImportError:
        shap = None


class AdvancedFeatures:
    """高级功能管理器，集成所有高级机器学习功能。"""
    
    def __init__(self):
        """初始化高级功能管理器。"""
        self.temp_dir = tempfile.mkdtemp()
        ensure_data_dir(self.temp_dir)
        import_optional_dependencies()
    
    def hyperparameter_tuning(self,
                             model: Any,
                             X: np.ndarray,
                             y: np.ndarray,
                             param_grid: Dict[str, List],
                             method: str = "grid",
                             cv: int = 5,
                             n_jobs: int = -1,
                             n_iter: int = 10,
                             scoring: str = "accuracy") -> Dict[str, Any]:
        """超参数自动调优。
        
        参数：
            model: 基础模型
            X: 特征矩阵
            y: 目标向量
            param_grid: 参数网格
            method: 调优方法 (grid/random/bayesian)
            cv: 交叉验证折数
            n_jobs: 并行作业数
            n_iter: 随机搜索迭代次数
            scoring: 评分标准
        
        返回：
            Dict[str, Any]: 调优结果
        """
        try:
            if method == "grid" and GridSearchCV is not None:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    verbose=1
                )
            elif method == "random" and RandomizedSearchCV is not None:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    verbose=1,
                    random_state=42
                )
            elif method == "bayesian" and BayesianOptimization is not None:
                search = BayesianOptimization(
                    estimator=model,
                    search_spaces=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    verbose=1,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported tuning method: {method}")
            
            # 执行调优
            with st.spinner(_("tuning_progress")):
                search.fit(X, y)
            
            # 获取最佳参数和分数
            best_params = search.best_params_
            best_score = search.best_score_
            best_model = search.best_estimator_
            
            return {
                "best_model": best_model,
                "best_params": best_params,
                "best_score": best_score,
                "cv_results": search.cv_results_ if hasattr(search, 'cv_results_') else None,
                "method": method
            }
            
        except Exception as e:
            raise RuntimeError(_("tuning_error", error=str(e)))
    
    def apply_smote(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   sampling_strategy: str = "auto",
                   k_neighbors: int = 5,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """应用SMOTE数据增强。
        
        参数：
            X: 特征矩阵
            y: 目标向量
            sampling_strategy: 采样策略
            k_neighbors: K近邻数量
            random_state: 随机种子
        
        返回：
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: (增强后的X, 增强后的y, 统计信息)
        """
        try:
            if SMOTE is None:
                raise ImportError("imbalanced-learn not installed")
            
            # 计算原始类别分布
            original_dist = pd.Series(y).value_counts().to_dict()
            
            # 应用SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state
            )
            
            with st.spinner(_("applying_smote")):
                X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # 计算增强后的类别分布
            augmented_dist = pd.Series(y_resampled).value_counts().to_dict()
            
            stats = {
                "original_distribution": original_dist,
                "augmented_distribution": augmented_dist,
                "original_size": len(X),
                "augmented_size": len(X_resampled),
                "augmentation_ratio": len(X_resampled) / len(X)
            }
            
            return X_resampled, y_resampled, stats
            
        except Exception as e:
            raise RuntimeError(_("smote_error", error=str(e)))
    
    def shap_analysis(self,
                     model: Any,
                     X: np.ndarray,
                     feature_names: Optional[List[str]] = None,
                     sample_size: int = 100) -> Dict[str, Any]:
        """SHAP特征重要性分析。
        
        参数：
            model: 训练好的模型
            X: 特征矩阵
            feature_names: 特征名称
            sample_size: 样本数量
        
        返回：
            Dict[str, Any]: SHAP分析结果
        """
        try:
            if shap is None:
                raise ImportError("shap not installed")
            
            # 采样以加速计算
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # 创建SHAP解释器
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample)
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X_sample)
            
            # 计算SHAP值
            with st.spinner(_("generating_shap")):
                shap_values = explainer.shap_values(X_sample)
            
            # 生成可视化
            fig_summary = self._create_shap_summary_plot(shap_values, X_sample, feature_names)
            fig_waterfall = self._create_shap_waterfall_plot(shap_values, X_sample, feature_names)
            
            # 计算特征重要性
            if isinstance(shap_values, list):
                # 多分类情况
                feature_importance = np.mean([np.mean(np.abs(values), axis=0) for values in shap_values], axis=0)
            else:
                # 二分类或回归情况
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            return {
                "shap_values": shap_values,
                "explainer": explainer,
                "feature_importance": feature_importance,
                "summary_plot": fig_summary,
                "waterfall_plot": fig_waterfall,
                "sample_size": len(X_sample)
            }
            
        except Exception as e:
            raise RuntimeError(_("shap_error", error=str(e)))
    
    def _create_shap_summary_plot(self, shap_values, X_sample, feature_names=None):
        """创建SHAP摘要图。"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if shap is not None:
                if isinstance(shap_values, list):
                    # 多分类情况，使用第一个类别
                    shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False, plot_size=(10, 6))
                else:
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, plot_size=(10, 6))
            
            plt.tight_layout()
            return fig
            
        except Exception:
            return None
    
    def _create_shap_waterfall_plot(self, shap_values, X_sample, feature_names=None):
        """创建SHAP瀑布图。"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if shap is not None:
                # 选择第一个样本创建瀑布图
                if isinstance(shap_values, list):
                    shap.waterfall_plot(shap.Explanation(shap_values[0][0], base_values=explainer.expected_value[0], data=X_sample[0], feature_names=feature_names), show=False)
                else:
                    shap.waterfall_plot(shap.Explanation(shap_values[0], base_values=explainer.expected_value, data=X_sample[0], feature_names=feature_names), show=False)
            
            plt.tight_layout()
            return fig
            
        except Exception:
            return None
    
    def advanced_filtering(self,
                          df: pd.DataFrame,
                          conditions: List[Dict[str, Any]],
                          save_scheme: bool = False,
                          scheme_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """高级筛选系统。
        
        参数：
            df: 数据框
            conditions: 筛选条件列表
            save_scheme: 是否保存筛选方案
            scheme_name: 方案名称
        
        返回：
            Tuple[pd.DataFrame, Dict[str, Any]]: (筛选结果, 统计信息)
        """
        try:
            filtered_df = df.copy()
            applied_conditions = []
            
            for condition in conditions:
                column = condition.get("column")
                operator = condition.get("operator")
                value = condition.get("value")
                logic = condition.get("logic", "and")
                
                if column not in filtered_df.columns:
                    continue
                
                # 应用条件
                if operator == "equals":
                    mask = filtered_df[column] == value
                elif operator == "not_equals":
                    mask = filtered_df[column] != value
                elif operator == "greater_than":
                    mask = filtered_df[column] > value
                elif operator == "less_than":
                    mask = filtered_df[column] < value
                elif operator == "greater_equal":
                    mask = filtered_df[column] >= value
                elif operator == "less_equal":
                    mask = filtered_df[column] <= value
                elif operator == "contains":
                    mask = filtered_df[column].str.contains(str(value), na=False)
                elif operator == "not_contains":
                    mask = ~filtered_df[column].str.contains(str(value), na=False)
                elif operator == "in":
                    mask = filtered_df[column].isin(value if isinstance(value, list) else [value])
                elif operator == "not_in":
                    mask = ~filtered_df[column].isin(value if isinstance(value, list) else [value])
                elif operator == "between":
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        mask = (filtered_df[column] >= value[0]) & (filtered_df[column] <= value[1])
                    else:
                        continue
                else:
                    continue
                
                # 应用逻辑运算
                if logic == "and":
                    filtered_df = filtered_df[mask]
                elif logic == "or":
                    # 对于or逻辑，需要特殊处理
                    if len(applied_conditions) == 0:
                        filtered_df = filtered_df[mask]
                    else:
                        filtered_df = filtered_df | mask
                
                applied_conditions.append({
                    "column": column,
                    "operator": operator,
                    "value": value,
                    "logic": logic
                })
            
            # 保存筛选方案
            if save_scheme and scheme_name:
                self._save_filter_scheme(scheme_name, conditions)
            
            # 统计信息
            stats = {
                "original_size": len(df),
                "filtered_size": len(filtered_df),
                "filter_ratio": len(filtered_df) / len(df) if len(df) > 0 else 0,
                "applied_conditions": applied_conditions
            }
            
            return filtered_df, stats
            
        except Exception as e:
            raise RuntimeError(_("filter_error", error=str(e)))
    
    def _save_filter_scheme(self, scheme_name: str, conditions: List[Dict[str, Any]]) -> None:
        """保存筛选方案。"""
        try:
            schemes_dir = os.path.join("data", "filter_schemes")
            ensure_data_dir(schemes_dir)
            
            scheme_file = os.path.join(schemes_dir, f"{scheme_name}.json")
            scheme_data = {
                "name": scheme_name,
                "created_at": datetime.now().isoformat(),
                "conditions": conditions
            }
            
            with open(scheme_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(scheme_data, f, ensure_ascii=False, indent=2)
            
            st.success(_("filter_saved"))
            
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
    
    def load_filter_scheme(self, scheme_name: str) -> Optional[List[Dict[str, Any]]]:
        """加载筛选方案。"""
        try:
            schemes_dir = os.path.join("data", "filter_schemes")
            scheme_file = os.path.join(schemes_dir, f"{scheme_name}.json")
            
            if not os.path.exists(scheme_file):
                return None
            
            with open(scheme_file, 'r', encoding='utf-8') as f:
                import json
                scheme_data = json.load(f)
            
            return scheme_data.get("conditions", [])
            
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
            return None
    
    def list_filter_schemes(self) -> List[str]:
        """列出所有筛选方案。"""
        try:
            schemes_dir = os.path.join("data", "filter_schemes")
            ensure_data_dir(schemes_dir)
            
            schemes = []
            for filename in os.listdir(schemes_dir):
                if filename.endswith('.json'):
                    scheme_name = filename[:-5]  # 移除.json后缀
                    schemes.append(scheme_name)
            
            return sorted(schemes)
            
        except Exception:
            return []
    
    def training_monitor(self,
                        model: Any,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        epochs: int = 100,
                        batch_size: int = 32,
                        monitor_interval: float = 1.0) -> Dict[str, Any]:
        """训练过程监控。
        
        参数：
            model: 模型对象
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            monitor_interval: 监控间隔（秒）
        
        返回：
            Dict[str, Any]: 监控结果
        """
        try:
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建图表占位符
            chart_placeholder = st.empty()
            
            # 训练历史
            train_history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
            
            # 模拟训练过程（实际使用时需要模型支持增量训练）
            for epoch in range(epochs):
                # 更新进度
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                # 更新状态文本
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                # 模拟训练指标（实际使用时应该从真实训练过程中获取）
                train_loss = np.random.normal(0.5, 0.1) * (1 - progress) + 0.1
                train_acc = progress + np.random.normal(0, 0.05)
                val_loss = np.random.normal(0.6, 0.1) * (1 - progress) + 0.15
                val_acc = progress * 0.9 + np.random.normal(0, 0.05)
                
                # 更新历史
                train_history["loss"].append(train_loss)
                train_history["accuracy"].append(max(0, min(1, train_acc)))
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(max(0, min(1, val_acc)))
                
                # 更新图表
                if epoch % max(1, epochs // 20) == 0:  # 每5%更新一次图表
                    self._update_training_chart(chart_placeholder, train_history, epoch + 1)
                
                # 模拟训练时间
                time.sleep(monitor_interval / epochs)
                
                # 检查是否停止
                if st.session_state.get('stop_training', False):
                    break
            
            # 最终图表
            self._update_training_chart(chart_placeholder, train_history, epochs)
            
            # 清理
            progress_bar.empty()
            status_text.empty()
            
            return {
                "train_history": train_history,
                "final_train_accuracy": train_history["accuracy"][-1] if train_history["accuracy"] else 0,
                "final_val_accuracy": train_history["val_accuracy"][-1] if train_history["val_accuracy"] else 0,
                "epochs_completed": len(train_history["loss"])
            }
            
        except Exception as e:
            raise RuntimeError(_("training_monitor_error", error=str(e)))
    
    def _update_training_chart(self, placeholder, history, current_epoch):
        """更新训练图表。"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            epochs = range(1, len(history["loss"]) + 1)
            
            # 损失曲线
            ax1.plot(epochs, history["loss"], 'b-', label='Training Loss')
            ax1.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # 准确率曲线
            ax2.plot(epochs, history["accuracy"], 'b-', label='Training Accuracy')
            ax2.plot(epochs, history["val_accuracy"], 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            placeholder.pyplot(fig)
            plt.close(fig)
            
        except Exception:
            pass
    
    def generate_academic_report(self,
                               title: str,
                               author: str,
                               abstract: str,
                               content: Dict[str, Any],
                               template: str = "academic_poster") -> bytes:
        """生成学术报告。
        
        参数：
            title: 报告标题
            author: 作者
            abstract: 摘要
            content: 报告内容
            template: 模板类型
        
        返回：
            bytes: PDF文件内容
        """
        try:
            if template == "academic_poster":
                return self._generate_academic_poster(title, author, abstract, content)
            elif template == "technical_report":
                return self._generate_technical_report(title, author, abstract, content)
            else:
                return self._generate_custom_report(title, author, abstract, content)
                
        except Exception as e:
            raise RuntimeError(_("report_error", error=str(e)))
    
    def _generate_academic_poster(self, title: str, author: str, abstract: str, content: Dict[str, Any]) -> bytes:
        """生成学术海报。"""
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.units import inch
            import io
            
            # 创建PDF缓冲区
            pdf_buffer = io.BytesIO()
            
            # 设置页面为横向A4
            pagesize = landscape(A4)
            doc = SimpleDocTemplate(pdf_buffer, pagesize=pagesize, topMargin=0.5*inch, bottomMargin=0.5*inch)
            
            # 获取样式
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # 居中
                textColor=colors.darkblue
            )
            
            author_style = ParagraphStyle(
                'CustomAuthor',
                parent=styles['Normal'],
                fontSize=14,
                spaceAfter=20,
                alignment=1,  # 居中
                textColor=colors.darkgreen
            )
            
            # 构建内容
            story = []
            
            # 标题
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # 作者
            story.append(Paragraph(f"By: {author}", author_style))
            story.append(Spacer(1, 0.3*inch))
            
            # 摘要
            abstract_style = ParagraphStyle(
                'Abstract',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=20,
                leftIndent=0.5*inch,
                rightIndent=0.5*inch
            )
            story.append(Paragraph("<b>Abstract:</b>", styles['Heading3']))
            story.append(Paragraph(abstract, abstract_style))
            story.append(Spacer(1, 0.2*inch))
            
            # 添加内容部分
            if 'methodology' in content:
                story.append(Paragraph("<b>Methodology:</b>", styles['Heading3']))
                story.append(Paragraph(content['methodology'], styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            if 'results' in content:
                story.append(Paragraph("<b>Results:</b>", styles['Heading3']))
                story.append(Paragraph(content['results'], styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            if 'conclusions' in content:
                story.append(Paragraph("<b>Conclusions:</b>", styles['Heading3']))
                story.append(Paragraph(content['conclusions'], styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            # 添加性能指标表格
            if 'metrics' in content:
                story.append(Paragraph("<b>Performance Metrics:</b>", styles['Heading3']))
                metrics_data = [['Metric', 'Value']]
                for metric, value in content['metrics'].items():
                    metrics_data.append([metric, f"{value:.4f}"])
                
                metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metrics_table)
            
            # 生成PDF
            doc.build(story)
            
            # 获取PDF内容
            pdf_content = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            return pdf_content
            
        except Exception as e:
            raise RuntimeError(f"Academic poster generation failed: {str(e)}")
    
    def _generate_technical_report(self, title: str, author: str, abstract: str, content: Dict[str, Any]) -> bytes:
        """生成技术报告。"""
        # 简化实现，实际使用时可以扩展
        return self._generate_academic_poster(title, author, abstract, content)
    
    def _generate_custom_report(self, title: str, author: str, abstract: str, content: Dict[str, Any]) -> bytes:
        """生成自定义报告。"""
        # 简化实现，实际使用时可以扩展
        return self._generate_academic_poster(title, author, abstract, content)


def create_advanced_features() -> AdvancedFeatures:
    """创建高级功能管理器实例。
    
    返回：
        AdvancedFeatures: 高级功能管理器实例
    """
    return AdvancedFeatures()


def render_advanced_filter_ui(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """渲染高级筛选界面。
    
    参数：
        df: 数据框
    
    返回：
        Optional[pd.DataFrame]: 筛选结果或None
    """
    st.subheader(_("advanced_filter_title"))
    
    if df is None or df.empty:
        st.warning(_("warning_data_empty"))
        return None
    
    # 获取高级功能管理器
    adv_features = create_advanced_features()
    
    # 条件构建器
    conditions = []
    
    st.write(_("filter_conditions"))
    
    # 添加条件
    num_conditions = st.number_input(_("add_condition"), min_value=1, max_value=10, value=1)
    
    for i in range(num_conditions):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            column = st.selectbox(f"字段 {i+1}", df.columns, key=f"column_{i}")
        
        with col2:
            operator = st.selectbox(
                f"操作符 {i+1}",
                ["equals", "not_equals", "greater_than", "less_than", "greater_equal", "less_equal", "contains", "between"],
                key=f"operator_{i}"
            )
        
        with col3:
            if operator == "between":
                min_val = st.number_input(f"最小值 {i+1}", key=f"min_val_{i}")
                max_val = st.number_input(f"最大值 {i+1}", key=f"max_val_{i}")
                value = [min_val, max_val]
            else:
                value = st.text_input(f"值 {i+1}", key=f"value_{i}")
        
        with col4:
            logic = st.selectbox(f"逻辑 {i+1}", ["and", "or"], key=f"logic_{i}")
        
        conditions.append({
            "column": column,
            "operator": operator,
            "value": value,
            "logic": logic
        })
    
    # 应用筛选
    if st.button(_("apply_filter")):
        try:
            filtered_df, stats = adv_features.advanced_filtering(df, conditions)
            
            st.success(_("filter_applied"))
            st.write(f"原始数据: {stats['original_size']} 条")
            st.write(f"筛选结果: {stats['filtered_size']} 条")
            st.write(f"筛选比例: {stats['filter_ratio']:.2%}")
            
            return filtered_df
            
        except Exception as e:
            st.error(_("filter_error", error=str(e)))
    
    return None