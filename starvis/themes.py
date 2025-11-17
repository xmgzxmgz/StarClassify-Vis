"""
统一主题系统模块（Theme System）

职责：
- 提供多种预设主题：暗色、亮色、星空渐变
- 统一管理颜色、字体、组件样式
- 支持动态主题切换
- 确保界面风格一致性

物理意义：
通过统一的主题系统提升用户体验，确保视觉一致性和专业性。
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any


class ThemeManager:
    """主题管理器，负责应用主题切换和样式管理。"""
    
    THEMES = {
        "dark": {
            "name": "暗色主题",
            "name_en": "Dark Theme",
            "colors": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "error": "#d62728",
                "warning": "#ff7f0e",
                "background": "#0e1117",
                "surface": "#262730",
                "text": "#fafafa",
                "text_secondary": "#a0a0a0",
                "border": "#404040"
            },
            "fonts": {
                "heading": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "body": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "code": "'Courier New', monospace"
            },
            "background": {
                "type": "gradient",
                "gradient": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
                "image": None
            }
        },
        "light": {
            "name": "亮色主题",
            "name_en": "Light Theme",
            "colors": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "error": "#d62728",
                "warning": "#ff7f0e",
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "text": "#262730",
                "text_secondary": "#6c757d",
                "border": "#dee2e6"
            },
            "fonts": {
                "heading": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "body": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "code": "'Courier New', monospace"
            },
            "background": {
                "type": "solid",
                "gradient": None,
                "image": None
            }
        },
        "galaxy": {
            "name": "星空渐变",
            "name_en": "Galaxy Gradient",
            "colors": {
                "primary": "#4facfe",
                "secondary": "#00f2fe",
                "success": "#43e97b",
                "error": "#fa709a",
                "warning": "#fee140",
                "background": "#0a0a0a",
                "surface": "rgba(255, 255, 255, 0.1)",
                "text": "#ffffff",
                "text_secondary": "#b0b0b0",
                "border": "rgba(255, 255, 255, 0.2)"
            },
            "fonts": {
                "heading": "'Orbitron', 'Arial Black', sans-serif",
                "body": "'Exo 2', 'Segoe UI', sans-serif",
                "code": "'Share Tech Mono', 'Courier New', monospace"
            },
            "background": {
                "type": "galaxy",
                "gradient": "radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%)",
                "image": "https://images.unsplash.com/photo-1454789548928-9efd52dc4031?q=80&w=1600&auto=format&fit=crop"
            }
        }
    }
    
    def __init__(self):
        """初始化主题管理器。"""
        self.current_theme = "galaxy"  # 默认主题
        self._apply_theme(self.current_theme)
    
    def get_available_themes(self) -> Dict[str, str]:
        """获取可用主题列表。
        
        返回：
            Dict[str, str]: 主题键值对 {theme_key: theme_name}
        """
        return {key: theme["name"] for key, theme in self.THEMES.items()}
    
    def get_current_theme(self) -> Dict[str, Any]:
        """获取当前主题配置。
        
        返回：
            Dict[str, Any]: 当前主题的完整配置
        """
        return self.THEMES[self.current_theme].copy()
    
    def set_theme(self, theme_key: str) -> None:
        """设置应用主题。
        
        参数：
            theme_key: 主题键名
        """
        if theme_key not in self.THEMES:
            raise ValueError(f"主题 '{theme_key}' 不存在。可用主题: {list(self.THEMES.keys())}")
        
        self.current_theme = theme_key
        self._apply_theme(theme_key)
        
        # 保存到session state
        if "theme_manager" not in st.session_state:
            st.session_state.theme_manager = self
    
    def _apply_theme(self, theme_key: str) -> None:
        """应用主题样式到Streamlit。"""
        theme = self.THEMES[theme_key]
        colors = theme["colors"]
        fonts = theme["fonts"]
        background = theme["background"]
        
        # 生成CSS样式
        css = self._generate_css(theme)
        
        # 应用背景
        if background["type"] == "galaxy":
            st.markdown(f"""
                <style>
                .stApp {{
                    background: {background["gradient"]};
                    background-image: url('{background["image"]}');
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    background-blend-mode: overlay;
                }}
                </style>
            """, unsafe_allow_html=True)
        elif background["type"] == "gradient":
            st.markdown(f"""
                <style>
                .stApp {{
                    background: {background["gradient"]};
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                </style>
            """, unsafe_allow_html=True)
        
        # 应用组件样式
        st.markdown(css, unsafe_allow_html=True)
    
    def _generate_css(self, theme: Dict[str, Any]) -> str:
        """生成CSS样式字符串。"""
        colors = theme["colors"]
        fonts = theme["fonts"]
        
        return f"""
        <style>
        /* 全局样式 */
        .stApp {{
            color: {colors["text"]};
            font-family: {fonts["body"]};
        }}
        
        /* 标题样式 */
        h1, h2, h3, h4, h5, h6 {{
            color: {colors["text"]} !important;
            font-family: {fonts["heading"]} !important;
            font-weight: 600 !important;
        }}
        
        /* 按钮样式 */
        .stButton > button {{
            background-color: {colors["primary"]};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {colors["secondary"]};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        
        /* 侧边栏样式 */
        .css-1d391kg {{
            background-color: {colors["surface"]};
        }}
        
        /* 数据框样式 */
        .dataframe {{
            background-color: {colors["surface"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
        }}
        
        /* 信息框样式 */
        .stAlert {{
            background-color: {colors["surface"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
        }}
        
        /* 滑块样式 */
        .stSlider > div > div {{
            background-color: {colors["primary"]};
        }}
        
        /* 文件上传样式 */
        .stFileUploader {{
            background-color: {colors["surface"]};
            border: 2px dashed {colors["border"]};
            border-radius: 8px;
        }}
        
        /* 成功/错误/警告样式 */
        .success {{
            color: {colors["success"]} !important;
        }}
        
        .error {{
            color: {colors["error"]} !important;
        }}
        
        .warning {{
            color: {colors["warning"]} !important;
        }}
        
        /* 自定义容器 */
        .custom-container {{
            background-color: {colors["surface"]};
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid {colors["border"]};
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }}
        
        /* 代码块样式 */
        code {{
            background-color: {colors["surface"]};
            color: {colors["text"]};
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: {fonts["code"]};
        }}
        
        /* 链接样式 */
        a {{
            color: {colors["primary"]};
            text-decoration: none;
            transition: color 0.3s ease;
        }}
        
        a:hover {{
            color: {colors["secondary"]};
        }}
        </style>
        """
    
    def get_theme_component(self, component: str) -> Any:
        """获取主题组件配置。
        
        参数：
            component: 组件名称 (colors, fonts, background)
        返回：
            组件配置字典
        """
        theme = self.get_current_theme()
        return theme.get(component, {})
    
    def apply_custom_style(self, style_name: str, custom_css: str) -> None:
        """应用自定义样式。
        
        参数：
            style_name: 样式名称
            custom_css: 自定义CSS字符串
        """
        st.markdown(f"""
            <style>
            /* 自定义样式: {style_name} */
            {custom_css}
            </style>
        """, unsafe_allow_html=True)


# 全局主题管理器实例
def get_theme_manager() -> ThemeManager:
    """获取全局主题管理器实例。
    
    返回：
        ThemeManager: 主题管理器实例
    """
    if "theme_manager" not in st.session_state:
        st.session_state.theme_manager = ThemeManager()
    return st.session_state.theme_manager


def apply_theme(theme_key: str = "galaxy") -> ThemeManager:
    """应用指定主题。
    
    参数：
        theme_key: 主题键名
    
    返回：
        ThemeManager: 主题管理器实例
    """
    manager = get_theme_manager()
    manager.set_theme(theme_key)
    return manager