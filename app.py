"""
StarClassify-Vis —— 基于 SDSS 大数据的恒星多维特征分类与交互式科普平台

说明：
- 本文件是 Streamlit 应用入口，提供侧边栏导航与四个主要页面：
  数据概览、模型训练报告、科研筛选工具、赫罗图科普互动。
- 所有业务逻辑封装在 starvis 包内的模块函数/类中。
- 代码遵循教学性质要求，函数均提供中文 Docstring。
"""

import os
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# 注意：为了降低启动时的依赖压力，部分第三方库在内部函数中懒加载

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


def page_header():
    """在页面顶部展示标题与背景说明。

    物理意义：为应用提供清晰的入口与背景图，在科普场景中提升用户沉浸感。
    """
    st.set_page_config(page_title="StarClassify-Vis", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1454789548928-9efd52dc4031?q=80&w=1600&auto=format&fit=crop');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .transbox {background: rgba(0, 0, 0, 0.45); padding: 0.8rem 1.2rem; border-radius: 8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="transbox">
        <h1 style="color:#fff;">StarClassify-Vis：基于 SDSS 的恒星分类与科普可视化</h1>
        <p style="color:#eee;">轻量化集成模型（逻辑回归 + 朴素贝叶斯软投票），双模式可视化（科研/科普）。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav() -> str:
    """构建侧边栏导航与超参数配置。

    返回值：当前选择的页面名称。
    物理意义：将不同任务分区，提升科研工作与科普教学的操作效率。
    """
    st.sidebar.title("功能导航")
    page = st.sidebar.radio(
        "选择页面",
        ["数据概览", "模型训练报告", "科研筛选工具", "赫罗图科普互动", "加载典型案例", "测试中心"],
        index=0,
    )

    st.sidebar.title("超参数配置")
    train_ratio = st.sidebar.slider("训练集占比", 0.5, 0.9, 0.8, 0.05)
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
    st.subheader("数据概览")
    uploaded = st.file_uploader("上传 SDSS CSV（可选）", type=["csv"])

    if st.button("加载 SDSS 数据"):
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
            st.success(f"成功加载数据：{info['rows']} 行、{info['cols']} 列，内存约 {info['memory_mb']:.2f} MB")

            st.write("前 10 行预览：")
            st.dataframe(df.head(10))

            fig = plot_sky_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

            # 预先计算基础特征，便于后续页面使用
            features_df, feature_names = engineer.build_features(df)
            st.session_state["features"] = features_df
            st.session_state["feature_names"] = feature_names
            st.info("已生成颜色指数与空间特征，用于后续训练与分析。")
        except Exception as e:
            st.error(f"数据加载失败：{e}")


def page_model_training(preprocessor: Preprocessor, trainer: ModelTrainer, evaluator: Evaluator):
    """页面 B：模型训练报告。

    功能：执行训练/测试集划分，训练软投票集成模型，展示指标与混淆矩阵。
    物理意义：通过可解释性的线性与概率模型融合，提升分类性能并保留可解释性。
    """
    st.subheader("模型训练报告")
    df = st.session_state.get("df")
    features_df = st.session_state.get("features")
    train_ratio = st.session_state.get("train_ratio", 0.8)

    if df is None or features_df is None:
        st.warning("请先在‘数据概览’页面加载数据并生成基础特征。")
        return

    if st.button("开始训练"):
        progress = st.progress(10)
        try:
            X, y, label_encoder = preprocessor.prepare_xy(features_df, df)
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
                f"训练完成！准确率 {metrics['accuracy']:.2%}，精确率 {metrics['precision']:.2%}，召回率 {metrics['recall']:.2%}，F1 {metrics['f1']:.2%}"
            )
            st.pyplot(fig_cm)

            # 特征重要性
            st.markdown("### 特征重要性（逻辑回归系数/排列重要性）")
            try:
                fig_imp = evaluator.feature_importance(model, X_test)
                if fig_imp is not None:
                    st.pyplot(fig_imp)
                    st.info("从结果可见：红移与 g-r 颜色等物理参数对分类贡献显著。")
            except Exception as e:
                st.warning(f"特征重要性计算失败：{e}")

        except Exception as e:
            st.error(f"训练过程出现错误：{e}")


def page_research_filter(preprocessor: Preprocessor, trainer: ModelTrainer):
    """页面 C：科研筛选工具。

    功能：支持批量预测 CSV，输出高置信度与需复核标签，并支持下载结果。
    物理意义：为科研场景提供快速筛选能力，降低人工标注成本。
    """
    st.subheader("科研筛选工具")
    model = st.session_state.get("model")
    df = st.session_state.get("df")
    features_df = st.session_state.get("features")
    label_encoder = st.session_state.get("label_encoder")

    if model is None or df is None or features_df is None or label_encoder is None:
        st.warning("请先在‘模型训练报告’页面完成训练。")
        return

    uploaded = st.file_uploader("上传待分类 CSV", type=["csv"])
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
            tag = pd.Series(["高置信度" if c >= 0.7 else "需人工复核" for c in confidences], name="review_tag")
            result = new_df.copy()
            result["pred_class"] = labels
            result["confidence"] = confidences
            result = pd.concat([result, tag], axis=1)

            st.success("批量预测完成。")
            st.dataframe(result.head(20))

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button("下载预测结果 CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"批量预测失败：{e}")


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
        st.image(img_url, caption=classification, use_column_width=True)
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

    if page == "数据概览":
        page_data_overview(loader, engineer)
    elif page == "模型训练报告":
        page_model_training(preprocessor, trainer, evaluator)
    elif page == "科研筛选工具":
        page_research_filter(preprocessor, trainer)
    elif page == "赫罗图科普互动":
        page_hr_interactive()
    elif page == "加载典型案例":
        page_cases(preprocessor, trainer)
    elif page == "测试中心":
        page_tests()


if __name__ == "__main__":
    run_app()