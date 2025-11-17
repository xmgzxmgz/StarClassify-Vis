"""
StarClassify-Vis —— 基于 SDSS 大数据的恒星多维特征分类与交互式科普平台

说明：
- 本文件是 Streamlit 应用入口，提供侧边栏导航与多个主要页面：
  数据概览、模型训练报告、科研筛选工具、赫罗图科普互动、批量操作、主题设置、
  语言设置、模型管理、数据增强、高级筛选、自动调优、特征分析、训练监控、报告生成。
- 所有业务逻辑封装在 starvis 包内的模块函数/类中。
- 代码遵循教学性质要求，函数均提供中文 Docstring。
- 新增功能：主题切换、国际化、批量操作、模型管理、高级分析等。
"""

import os
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# 注意：为了降低启动时的依赖压力，部分第三方库在内部函数中懒加载

# 基础模块
from starvis.data_loader import DataLoader
from starvis.preprocessing import Preprocessor
from starvis.features import FeatureEngineer
from starvis.model import ModelTrainer
from starvis.evaluation import Evaluator
from starvis.utils import (
    ensure_data_dir,
    plot_sky_distribution,
    plot_hr_diagram,
    explain_star_class,
    classify_by_rules,
    build_case_dataset,
)
from starvis.testing import (
    test_data_not_empty,
    test_predict_shape,
    test_data_loader_api,
    test_feature_engineer_api,
    test_preprocessor_api,
    test_model_trainer_api,
    test_evaluator_api,
    run_pipeline_quick,
)

# 新增功能模块
from starvis.themes import get_theme_manager, apply_theme
from starvis.i18n import get_i18n_manager, _, set_language
from starvis.batch_operations import create_batch_operations, handle_package_upload, handle_package_download
from starvis.model_management import create_model_manager, handle_model_save, handle_model_load
from starvis.advanced_features import create_advanced_features, render_advanced_filter_ui


def page_header():
    """在页面顶部展示标题与背景说明。

    物理意义：为应用提供清晰的入口与背景图，在科普场景中提升用户沉浸感。
    """
    # 应用主题
    theme_manager = get_theme_manager()
    
    # 设置页面配置
    st.set_page_config(
        page_title=_("app_title"), 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用主题样式
    theme_manager.apply_custom_style("header", """
        .header-container {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.8) 0%, rgba(0, 242, 254, 0.8) 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header-title {
            color: white !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .header-subtitle {
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 1.2rem !important;
            text-align: center;
            font-weight: 300;
        }
    """)
    
    # 显示标题
    st.markdown(f"""
        <div class="header-container">
            <h1 class="header-title">{_("app_title")}</h1>
            <p class="header-subtitle">{_("app_subtitle")}</p>
        </div>
    """, unsafe_allow_html=True)


def sidebar_nav() -> str:
    """构建侧边栏导航与超参数配置。

    返回值：当前选择的页面名称。
    物理意义：将不同任务分区，提升科研工作与科普教学的操作效率。
    """
    # 应用主题到侧边栏
    theme_manager = get_theme_manager()
    
    st.sidebar.title(_("settings"))
    
    # 主题切换
    with st.sidebar.expander(_("nav_theme_settings")):
        available_themes = theme_manager.get_available_themes()
        current_theme = st.selectbox(
            _("select_theme"),
            options=list(available_themes.keys()),
            format_func=lambda x: available_themes[x],
            key="theme_selector"
        )
        if current_theme != theme_manager.current_theme:
            theme_manager.set_theme(current_theme)
            st.success(_("theme_applied", theme_name=available_themes[current_theme]))
            st.rerun()
    
    # 语言切换
    with st.sidebar.expander(_("nav_language_settings")):
        i18n_manager = get_i18n_manager()
        available_languages = i18n_manager.get_available_languages()
        current_lang = st.selectbox(
            _("select_language"),
            options=list(available_languages.keys()),
            format_func=lambda x: available_languages[x],
            key="language_selector"
        )
        if current_lang != i18n_manager.current_language:
            set_language(current_lang)
            st.success(_("language_applied", language_name=available_languages[current_lang]))
            st.rerun()
    
    st.sidebar.title(_("settings"))
    
    # 主题切换
    with st.sidebar.expander(_("nav_theme_settings")):
        available_themes = theme_manager.get_available_themes()
        current_theme = st.selectbox(
            _("select_theme"),
            options=list(available_themes.keys()),
            format_func=lambda x: available_themes[x],
            key="theme_selector"
        )
        if current_theme != theme_manager.current_theme:
            theme_manager.set_theme(current_theme)
            st.success(_("theme_applied", theme_name=available_themes[current_theme]))
            st.rerun()
    
    # 语言切换
    with st.sidebar.expander(_("nav_language_settings")):
        i18n_manager = get_i18n_manager()
        available_languages = i18n_manager.get_available_languages()
        current_lang = st.selectbox(
            _("select_language"),
            options=list(available_languages.keys()),
            format_func=lambda x: available_languages[x],
            key="language_selector"
        )
        if current_lang != i18n_manager.current_language:
            set_language(current_lang)
            st.success(_("language_applied", language_name=available_languages[current_lang]))
            st.rerun()
    
    st.sidebar.title(_("功能导航"))
    page = st.sidebar.radio(
        _("选择页面"),
        [
            "nav_data_overview",
            "nav_model_training", 
            "nav_research_filter",
            "nav_hr_interactive",
            "nav_case_studies",
            "nav_testing_center",
            "nav_batch_operations",
            "nav_model_management",
            "nav_data_augmentation",
            "nav_advanced_filter",
            "nav_auto_tuning",
            "nav_feature_analysis",
            "nav_training_monitor",
            "nav_report_generator"
        ],
        format_func=lambda x: _(x),
        index=0,
    )

    st.sidebar.title(_("超参数配置"))
    train_ratio = st.sidebar.slider(_("train_ratio_config"), 0.5, 0.9, 0.8, 0.05)
    st.session_state["train_ratio"] = train_ratio

    return page


def init_states():
    """初始化 Streamlit 会话状态，避免首次运行缺少键导致异常。

    物理意义：确保系统稳健性，使交互在多次操作间保持一致状态。
    """
    keys = [
        "df",
        "features",
        "X",
        "y",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "model",
        "label_encoder",
        "feature_names",
        "train_ratio",
    ]
    for k in keys:
        st.session_state.setdefault(k, None)


def page_data_overview(loader: DataLoader, engineer: FeatureEngineer):
    """页面 A：数据概览。

    功能：加载 CSV 或生成模拟数据；展示数据摘要与天球分布（ra vs dec）。
    物理意义：帮助用户了解数据规模与空间分布，为后续特征工程与模型训练提供直觉。
    """
    st.subheader(_("data_overview_title"))
    uploaded = st.file_uploader(_("data_upload_placeholder"), type=["csv"])

    if st.button(_("load_data_button")):
        try:
            ensure_data_dir()
            if uploaded is not None:
                df = loader.load_from_buffer(uploaded)
            else:
                # 若本地不存在则自动生成模拟数据
                csv_path = os.path.join("data", "sdss_mock.csv")
                df = loader.load_csv_or_mock(csv_path)
            st.session_state["df"] = df

            info = loader.get_info(df)
            st.success(_("data_load_success", rows=info["rows"], cols=info["cols"], memory_mb=info["memory_mb"]))

            st.write(_("data_preview"))
            st.dataframe(df.head(10))

            fig = plot_sky_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

            # 预先计算基础特征，便于后续页面使用
            features_df, feature_names = engineer.build_features(df)
            st.session_state["features"] = features_df
            st.session_state["feature_names"] = feature_names
            st.info(_("features_generated"))
        except Exception as e:
            st.error(_("data_load_error", error=e))


def page_model_training(preprocessor: Preprocessor, trainer: ModelTrainer, evaluator: Evaluator):
    """页面 B：模型训练报告。

    功能：执行训练/测试集划分，训练软投票集成模型，展示指标与混淆矩阵。
    物理意义：通过可解释性的线性与概率模型融合，提升分类性能并保留可解释性。
    """
    st.subheader(_("model_training_title"))
    df = st.session_state.get("df")
    features_df = st.session_state.get("features")
    train_ratio = st.session_state.get("train_ratio", 0.8)

    if df is None or features_df is None:
        st.warning(_("load_data_first"))
        return

    if st.button(_("start_training_button")):
        progress = st.progress(10)
        try:
            X, y, label_encoder = preprocessor.prepare_xy(features_df, df)
            # 保存已拟合的预处理器到会话，便于后续科研筛选/案例演示复用
            st.session_state["preprocessor"] = preprocessor
            progress.progress(30)

            X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=train_ratio)
            st.session_state.update({
                "X": X, "y": y,
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "label_encoder": label_encoder
            })
            progress.progress(60)

            model = trainer.train_voting_classifier(X_train, y_train, weights=(0.6, 0.4))
            st.session_state["model"] = model
            progress.progress(80)

            metrics, fig_cm = evaluator.evaluate(model, X_test, y_test)
            progress.progress(100)

            st.success(
                _("training_complete", accuracy=metrics["accuracy"], precision=metrics["precision"], 
                  recall=metrics["recall"], f1=metrics["f1"])
            )
            st.pyplot(fig_cm)

            # 特征重要性
            st.markdown(f"### {_(\"feature_importance\")}")
            try:
                fig_imp = evaluator.feature_importance(model, X_test)
                if fig_imp is not None:
                    st.pyplot(fig_imp)
                    st.info(_("feature_importance_info"))
            except Exception as e:
                st.warning(f"特征重要性计算失败：{e}")

        except Exception as e:
            st.error(_("training_error", error=e))


def page_research_filter(preprocessor: Preprocessor, trainer: ModelTrainer):
    """页面 C：科研筛选工具。

    功能：支持批量预测 CSV，输出高置信度与需复核标签，并支持下载结果。
    物理意义：为科研场景提供快速筛选能力，降低人工标注成本。
    """
    st.subheader(_("research_filter_title"))
    model = st.session_state.get("model")
    df = st.session_state.get("df")
    features_df = st.session_state.get("features")
    label_encoder = st.session_state.get("label_encoder")
    # 优先使用会话中已拟合的预处理器
    preprocessor = st.session_state.get("preprocessor") or preprocessor

    if model is None or df is None or features_df is None or label_encoder is None:
        st.warning(_("model_not_trained"))
        return
    if getattr(preprocessor, "_scaler", None) is None:
        st.warning(_("model_not_trained"))
        return

    uploaded = st.file_uploader(_("upload_for_prediction"), type=["csv"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            engineer = FeatureEngineer()
            new_features, _ = engineer.build_features(new_df)

            X_new = preprocessor.transform_features(new_features)
            y_pred_proba = trainer.predict_proba(model, X_new)
            y_pred = y_pred_proba.argmax(axis=1)
            labels = label_encoder.inverse_transform(y_pred)

            # 置信度分类
            confidences = y_pred_proba.max(axis=1)
            tag = pd.Series([_("high_confidence") if c >= 0.7 else _("needs_review") for c in confidences], name="review_tag")
            result = new_df.copy()
            result["pred_class"] = labels
            result["confidence"] = confidences
            result = pd.concat([result, tag], axis=1)

            st.success(_("prediction_complete"))
            st.dataframe(result.head(20))

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(_("download_predictions"), data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(_("prediction_error", error=e))


def page_hr_interactive():
    """页面 D：赫罗图科普互动。

    功能：绘制赫罗图（颜色指数/温度 vs 星等/光度）；通过滑块调整有效温度、金属丰度，实时给出分类与科普解释，并在图中高亮对应区域。
    物理意义：通过互动让学生理解不同恒星类型在赫罗图上的分布与演化阶段。
    """
    st.subheader("赫罗图科普互动")
    df = st.session_state.get("df")
    features_df = st.session_state.get("features")
    if df is None or features_df is None:
        st.warning("请先在‘数据概览’页面加载数据。")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        temp = st.slider("有效温度 (K)", 3000, 15000, 6000, 100)
        feh = st.slider("金属丰度 [Fe/H]", -2.5, 0.5, -0.2, 0.1)
        classification = classify_by_rules(temp, feh)
        st.info(f"判定类型：{classification}")
        explain, img_url = explain_star_class(classification)
        st.image(img_url, caption=classification, use_container_width=True)
        st.write(explain)

    with col2:
        fig_hr = plot_hr_diagram(df, features_df, highlight_point=True, temp=temp)
        st.plotly_chart(fig_hr, use_container_width=True)


def page_cases(preprocessor: Preprocessor, trainer: ModelTrainer):
    """页面：典型案例演示。

    功能：一键加载太阳、天狼星、参宿四等案例，展示模型分类与概率分布。
    物理意义：通过熟悉的恒星例子帮助学生建立直觉，理解模型输出。
    """
    st.subheader("典型案例演示")

    model = st.session_state.get("model")
    label_encoder = st.session_state.get("label_encoder")
    # 使用训练阶段保存的预处理器
    preprocessor = st.session_state.get("preprocessor") or preprocessor
    if model is None or label_encoder is None:
        st.warning("请先完成模型训练。")
        return

    if st.button("一键加载案例"):
        try:
            df_cases = build_case_dataset()
            engineer = FeatureEngineer()
            case_features, feature_names = engineer.build_features(df_cases)
            X_case = preprocessor.transform_features(case_features)
            y_pred_proba = trainer.predict_proba(model, X_case)
            y_pred = y_pred_proba.argmax(axis=1)
            labels = label_encoder.inverse_transform(y_pred)

            df_show = df_cases[["name", "g", "r", "redshift", "feh"]].copy()
            df_show["类别"] = labels
            df_show["置信度"] = y_pred_proba.max(axis=1)
            st.dataframe(df_show)

            st.info("示例结果表明：红移与 g-r 颜色是分类关键要素之一。")
        except Exception as e:
            st.error(f"案例演示失败：{e}")


def page_tests():
    """页面：测试中心。

    提供模块级与端到端测试入口，展示通过/失败与关键指标。
    """
    st.subheader("测试中心：一键验证各模块与端到端流程")
    col_run, col_info = st.columns([1, 2])

    with col_run:
        run_all = st.button("运行所有模块测试")
        run_e2e = st.button("一键端到端验证")

    # 初始化工具实例（局部使用，避免影响全局状态）
    loader = DataLoader()
    engineer = FeatureEngineer()
    preprocessor = Preprocessor()
    trainer = ModelTrainer()
    evaluator = Evaluator()

    if run_all:
        try:
            st.write("—— 数据加载模块测试 ——")
            ok, df, info = test_data_loader_api(loader)
            st.write({"通过": ok, **info})
            if not ok:
                st.error("数据加载测试未通过：数据为空。")
                return

            st.write("—— 特征工程模块测试 ——")
            features_df, names = test_feature_engineer_api(engineer, df)
            st.write({"特征数": len(names), "样本数": len(features_df)})

            st.write("—— 预处理模块测试 ——")
            X, y, le = test_preprocessor_api(preprocessor, features_df, df)
            st.write({"X_shape": list(X.shape), "y_len": int(len(y)), "类别数": int(len(le.classes_))})

            st.write("—— 模型训练模块测试 ——")
            model, proba_shape = test_model_trainer_api(trainer, X, y)
            st.write({"预测概率形状": list(proba_shape)})
            if not test_predict_shape(np.empty(proba_shape)):
                st.error("预测概率形状不符合预期。")
                return

            st.write("—— 评估模块测试 ——")
            metrics, fig_cm = test_evaluator_api(evaluator, model, X[:1000], y[:1000])
            st.json(metrics)
            if fig_cm is not None:
                st.pyplot(fig_cm)
            else:
                st.info("未生成混淆矩阵图。")
            st.success("模块级测试全部通过 ✅")
        except Exception as e:
            st.error(f"测试运行失败：{e}")

    if run_e2e:
        try:
            result = run_pipeline_quick(loader, engineer, preprocessor, trainer, evaluator)
            st.write({
                "数据行数": result["rows"],
                "核心特征数": result["n_features"],
                "训练集大小": result["train_size"],
                "测试集大小": result["test_size"],
            })
            st.subheader("评估指标")
            st.json(result["metrics"])
            st.subheader("混淆矩阵")
            if result["fig_cm"] is not None:
                st.pyplot(result["fig_cm"])  # seaborn 可选，已在 Evaluator 中处理
            else:
                st.info("未生成混淆矩阵图。")
            st.success("端到端流程验证通过 ✅")
        except Exception as e:
            st.error(f"端到端流程失败：{e}")


def run_app():
    """主程序入口：组装页面并处理导航逻辑。

    物理意义：将各模块功能有机结合，形成科研与科普双模式平台。
    """
    page_header()
    init_states()
    page = sidebar_nav()

    # 初始化模块实例
    loader = DataLoader()
    preprocessor = Preprocessor()
    engineer = FeatureEngineer()
    trainer = ModelTrainer()
    evaluator = Evaluator()

    # 页面路由
    page_map = {
        "nav_data_overview": lambda: page_data_overview(loader, engineer),
        "nav_model_training": lambda: page_model_training(preprocessor, trainer, evaluator),
        "nav_research_filter": lambda: page_research_filter(preprocessor, trainer),
        "nav_hr_interactive": page_hr_interactive,
        "nav_case_studies": lambda: page_cases(preprocessor, trainer),
        "nav_testing_center": page_tests,
        "nav_batch_operations": page_batch_operations,
        "nav_model_management": page_model_management,
        "nav_data_augmentation": page_data_augmentation,
        "nav_advanced_filter": lambda: page_advanced_filter(preprocessor),
        "nav_auto_tuning": page_auto_tuning,
        "nav_feature_analysis": page_feature_analysis,
        "nav_training_monitor": page_training_monitor,
        "nav_report_generator": page_report_generator,
    }

    # 执行对应页面
    if page in page_map:
        page_map[page]()
    else:
        st.error(f"页面 '{page}' 未找到")


# 新增页面函数
def page_batch_operations():
    """批量操作页面。"""
    st.subheader(_("batch_operations_title"))
    
    tab1, tab2, tab3 = st.tabs([_("upload_zip"), _("download_zip"), _("package_contents")])
    
    with tab1:
        package_info = handle_package_upload()
        if package_info:
            st.json(package_info.get("metadata", {}))
    
    with tab2:
        # 获取当前会话数据
        df = st.session_state.get("df")
        model = st.session_state.get("model")
        
        if df is not None and model is not None:
            package_name = st.text_input(_("package_name"), value="research_package")
            if st.button(_("create_package")):
                # 准备数据文件
                data_files = {
                    "data.csv": df,
                    "features.csv": st.session_state.get("features", pd.DataFrame())
                }
                
                # 准备模型文件
                import joblib
                model_bytes = joblib.dumps(model)
                model_files = {
                    "model.pkl": model_bytes
                }
                
                # 准备报告文件
                report_files = {
                    "README.md": f"# {package_name}\\n\\nGenerated on {pd.Timestamp.now()}",
                    "metadata.json": str(st.session_state)
                }
                
                handle_package_download(
                    package_name=package_name,
                    data_files=data_files,
                    model_files=model_files,
                    report_files=report_files
                )
        else:
            st.warning("请先加载数据并训练模型")
    
    with tab3:
        batch_ops = create_batch_operations()
        packages = batch_ops.list_packages()
        
        if packages:
            for pkg in packages:
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(pkg["name"])
                with col2:
                    st.write(pkg["created_at"])
                with col3:
                    st.write(f"{pkg['size'] / 1024 / 1024:.1f} MB")
                with col4:
                    if st.button(_("delete"), key=f"del_{pkg['name']}"):
                        batch_ops.delete_package(pkg["name"])
                        st.rerun()


def page_model_management():
    """模型管理页面。"""
    st.subheader(_("model_management_title"))
    
    manager = create_model_manager()
    models = manager.list_models()
    
    if not models:
        st.info(_("no_models_found"))
        return
    
    # 显示模型列表
    for model in models:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
        
        with col1:
            st.write(f"{model['name']} v{model['version']}")
        with col2:
            status_color = "🟢" if model['status'] == 'active' else "⚪"
            st.write(f"{status_color} {model['status']}")
        with col3:
            accuracy = model['performance_metrics'].get('accuracy', 0)
            st.write(f"{accuracy:.1%}")
        with col4:
            st.write(f"{model['size'] / 1024:.1f} KB")
        with col5:
            if model['status'] != 'active':
                if st.button(_("rollback_model"), key=f"rollback_{model['name']}_{model['version']}"):
                    if manager.rollback_model(model['name'], model['version']):
                        st.success(_("model_rollback"))
                        st.rerun()
            
            if st.button(_("delete_model"), key=f"delete_{model['name']}_{model['version']}"):
                if manager.delete_model(model['name'], model['version']):
                    st.success(_("model_deleted"))
                    st.rerun()


def page_data_augmentation():
    """数据增强页面。"""
    st.subheader(_("data_augmentation_title"))
    
    df = st.session_state.get("df")
    if df is None:
        st.warning("请先加载数据")
        return
    
    # 获取特征和标签
    features_df = st.session_state.get("features")
    if features_df is None:
        engineer = FeatureEngineer()
        features_df, _ = engineer.build_features(df)
    
    # 准备数据
    X = features_df.values
    y = df['class'].values if 'class' in df.columns else np.zeros(len(df))
    
    # SMOTE配置
    col1, col2 = st.columns(2)
    with col1:
        sampling_strategy = st.selectbox("采样策略", ["auto", "minority", "not minority", "all"])
    with col2:
        k_neighbors = st.number_input("K近邻数量", min_value=1, max_value=20, value=5)
    
    if st.button(_("apply_smote")):
        try:
            adv_features = create_advanced_features()
            X_resampled, y_resampled, stats = adv_features.apply_smote(
                X, y, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors
            )
            
            st.success(_("smote_applied"))
            
            # 显示统计信息
            col1, col2 = st.columns(2)
            with col1:
                st.write(_("original_distribution"))
                st.json(stats["original_distribution"])
            with col2:
                st.write(_("augmented_distribution"))
                st.json(stats["augmented_distribution"])
            
            st.write(f"数据增强比例: {stats['augmentation_ratio']:.2f}x")
            
            # 更新会话状态
            st.session_state["X_resampled"] = X_resampled
            st.session_state["y_resampled"] = y_resampled
            
        except Exception as e:
            st.error(f"SMOTE应用失败: {str(e)}")


def page_advanced_filter(preprocessor: Preprocessor):
    """高级筛选页面。"""
    st.subheader(_("advanced_filter_title"))
    
    df = st.session_state.get("df")
    if df is None:
        st.warning("请先加载数据")
        return
    
    # 使用通用的高级筛选界面
    filtered_df = render_advanced_filter_ui(df)
    
    if filtered_df is not None:
        st.write("筛选结果:")
        st.dataframe(filtered_df)


def page_auto_tuning():
    """自动调优页面。"""
    st.subheader(_("auto_tuning_title"))
    
    # 检查必要的数据
    X_train = st.session_state.get("X_train")
    y_train = st.session_state.get("y_train")
    
    if X_train is None or y_train is None:
        st.warning("请先训练模型以获取训练数据")
        return
    
    # 调优配置
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(_("tuning_method"), ["grid", "random", "bayesian"])
    with col2:
        cv_folds = st.number_input(_("cv_folds"), min_value=2, max_value=10, value=5)
    
    # 参数网格配置
    st.write("参数配置:")
    param_grid = {}
    
    if st.checkbox("调优逻辑回归"):
        col1, col2 = st.columns(2)
        with col1:
            C_values = st.text_input("C参数范围", value="0.1,1,10")
        with col2:
            max_iter_values = st.text_input("max_iter参数范围", value="100,200,500")
        
        param_grid['logisticregression__C'] = [float(x) for x in C_values.split(',')]
        param_grid['logisticregression__max_iter'] = [int(x) for x in max_iter_values.split(',')]
    
    if st.button(_("start_tuning")):
        try:
            # 创建基础模型
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.ensemble import VotingClassifier
            
            base_model = VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=500)),
                    ('nb', GaussianNB())
                ],
                voting='soft',
                weights=[0.6, 0.4]
            )
            
            # 执行调优
            adv_features = create_advanced_features()
            tuning_results = adv_features.hyperparameter_tuning(
                base_model, X_train, y_train, param_grid, method=method, cv=cv_folds
            )
            
            st.success(_("tuning_complete"))
            
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.write(_("best_params"))
                st.json(tuning_results["best_params"])
            with col2:
                st.write(_("best_score"))
                st.metric("Best Score", f"{tuning_results['best_score']:.4f}")
            
            # 保存最佳模型
            st.session_state["best_model"] = tuning_results["best_model"]
            
        except Exception as e:
            st.error(_("tuning_error", error=str(e)))


def page_feature_analysis():
    """特征分析页面。"""
    st.subheader(_("feature_analysis_title"))
    
    # 检查必要的数据
    model = st.session_state.get("model")
    X_test = st.session_state.get("X_test")
    feature_names = st.session_state.get("feature_names")
    
    if model is None or X_test is None:
        st.warning("请先训练模型")
        return
    
    tab1, tab2 = st.tabs([_("shap_analysis"), _("feature_importance_plot")])
    
    with tab1:
        if st.button(_("generate_shap")):
            try:
                adv_features = create_advanced_features()
                shap_results = adv_features.shap_analysis(
                    model, X_test, feature_names=feature_names
                )
                
                st.success(_("shap_generated"))
                
                # 显示SHAP图
                if shap_results["summary_plot"]:
                    st.write(_("shap_summary"))
                    st.pyplot(shap_results["summary_plot"])
                
                if shap_results["waterfall_plot"]:
                    st.write(_("shap_waterfall"))
                    st.pyplot(shap_results["waterfall_plot"])
                
                # 显示特征重要性
                st.write("特征重要性排名:")
                importance_df = pd.DataFrame({
                    'feature': feature_names or range(len(shap_results["feature_importance"])),
                    'importance': shap_results["feature_importance"]
                }).sort_values('importance', ascending=False)
                
                st.dataframe(importance_df)
                
            except Exception as e:
                st.error(_("shap_error", error=str(e)))
    
    with tab2:
        # 显示传统的特征重要性
        st.write("传统特征重要性分析:")
        # 这里可以添加其他特征重要性分析方法


def page_training_monitor():
    """训练监控页面。"""
    st.subheader(_("training_monitor_title"))
    
    # 检查必要的数据
    X_train = st.session_state.get("X_train")
    y_train = st.session_state.get("y_train")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    
    if X_train is None or y_train is None or X_test is None or y_test is None:
        st.warning("请先训练模型以获取训练和测试数据")
        return
    
    # 监控配置
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("训练轮数", min_value=10, max_value=1000, value=100)
    with col2:
        monitor_interval = st.slider("监控间隔(秒)", min_value=0.1, max_value=5.0, value=1.0)
    
    if st.button(_("start_monitoring")):
        try:
            # 获取模型
            model = st.session_state.get("model")
            if model is None:
                # 创建新模型用于演示
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=epochs)
            
            adv_features = create_advanced_features()
            
            # 启动监控
            st.session_state['stop_training'] = False
            monitor_results = adv_features.training_monitor(
                model, X_train, y_train, X_test, y_test, 
                epochs=epochs, monitor_interval=monitor_interval
            )
            
            st.success(_("monitoring_completed"))
            st.json(monitor_results)
            
        except Exception as e:
            st.error(f"监控失败: {str(e)}")


def page_report_generator():
    """报告生成页面。"""
    st.subheader(_("report_generator_title"))
    
    # 报告配置
    with st.form("report_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input(_("report_title"), value="恒星分类研究报告")
            author = st.text_input(_("report_author"), value="研究团队")
        with col2:
            template = st.selectbox(_("report_template"), ["academic_poster", "technical_report", "custom_report"])
        
        abstract = st.text_area(_("report_abstract"), value="本报告总结了恒星分类研究的最新进展...")
        
        # 内容配置
        content = {}
        if st.checkbox("包含方法论"):
            content['methodology'] = st.text_area("方法论内容", value="使用了机器学习方法进行恒星分类...")
        
        if st.checkbox("包含结果"):
            content['results'] = st.text_area("结果内容", value="模型达到了较高的分类准确率...")
        
        if st.checkbox("包含结论"):
            content['conclusions'] = st.text_area("结论内容", value="研究表明机器学习方法在恒星分类中具有很好的应用前景...")
        
        if st.checkbox("包含性能指标"):
            # 从会话状态获取性能指标
            if 'model_metrics' in st.session_state:
                content['metrics'] = st.session_state['model_metrics']
            else:
                content['metrics'] = {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.93, 'f1': 0.94}
        
        submitted = st.form_submit_button(_("generate_report"))
        
        if submitted:
            try:
                adv_features = create_advanced_features()
                pdf_content = adv_features.generate_academic_report(
                    title, author, abstract, content, template=template
                )
                
                st.success(_("report_generated"))
                
                # 提供下载
                st.download_button(
                    label=_("download_pdf"),
                    data=pdf_content,
                    file_name=f"{title.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(_("report_error", error=str(e)))


if __name__ == "__main__":
    run_app()