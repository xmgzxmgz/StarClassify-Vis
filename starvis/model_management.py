"""
模型管理模块（Model Management）

职责：
- 模型版本管理
- 模型保存和加载
- 模型回滚功能
- 模型性能追踪
- 模型元数据管理

物理意义：
通过完善的模型生命周期管理，确保实验的可重复性和模型的可追溯性。
"""

from __future__ import annotations

import os
import json
import joblib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd

from .utils import ensure_data_dir
from .i18n import _


class ModelManager:
    """模型管理器，负责模型的版本控制和生命周期管理。"""
    
    def __init__(self, models_dir: str = "models"):
        """初始化模型管理器。
        
        参数：
            models_dir: 模型存储目录
        """
        self.models_dir = models_dir
        ensure_data_dir(models_dir)
        self.metadata_file = os.path.join(models_dir, "models_metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载模型元数据。"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"models": {}, "current_model": None, "model_history": []}
        return {"models": {}, "current_model": None, "model_history": []}
    
    def _save_metadata(self) -> None:
        """保存模型元数据。"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
    
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   performance_metrics: Optional[Dict[str, float]] = None,
                   training_metadata: Optional[Dict[str, Any]] = None,
                   feature_names: Optional[List[str]] = None,
                   model_params: Optional[Dict[str, Any]] = None) -> str:
        """保存模型。
        
        参数：
            model: 模型对象
            model_name: 模型名称
            performance_metrics: 性能指标
            training_metadata: 训练元数据
            feature_names: 特征名称列表
            model_params: 模型参数
        
        返回：
            str: 模型文件路径
        """
        try:
            # 生成版本号
            version = self._generate_version(model_name)
            versioned_name = f"{model_name}_v{version}"
            
            # 保存模型文件
            model_path = os.path.join(self.models_dir, f"{versioned_name}.pkl")
            joblib.dump(model, model_path)
            
            # 获取模型大小
            model_size = os.path.getsize(model_path)
            
            # 创建模型元数据
            model_info = {
                "name": model_name,
                "version": version,
                "versioned_name": versioned_name,
                "path": model_path,
                "created_at": datetime.now().isoformat(),
                "size": model_size,
                "performance_metrics": performance_metrics or {},
                "training_metadata": training_metadata or {},
                "feature_names": feature_names or [],
                "model_params": model_params or {},
                "status": "active"
            }
            
            # 更新元数据
            if model_name not in self.metadata["models"]:
                self.metadata["models"][model_name] = []
            
            self.metadata["models"][model_name].append(model_info)
            
            # 更新当前模型
            self.metadata["current_model"] = versioned_name
            
            # 添加到历史记录
            self.metadata["model_history"].append({
                "action": "save",
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat()
            })
            
            self._save_metadata()
            
            return model_path
            
        except Exception as e:
            raise RuntimeError(_("error_system_error", error=str(e)))
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """加载模型。
        
        参数：
            model_name: 模型名称
            version: 版本号（可选，默认加载最新版本）
        
        返回：
            Tuple[Any, Dict[str, Any]]: (模型对象, 模型信息)
        """
        try:
            if model_name not in self.metadata["models"]:
                raise ValueError(_("error_model_not_found"))
            
            model_versions = self.metadata["models"][model_name]
            
            if version is None:
                # 加载最新版本
                model_info = max(model_versions, key=lambda x: x["created_at"])
            else:
                # 加载指定版本
                model_info = next(
                    (info for info in model_versions if info["version"] == version),
                    None
                )
                if model_info is None:
                    raise ValueError(f"Model version {version} not found")
            
            # 加载模型
            model = joblib.load(model_info["path"])
            
            # 更新当前模型
            self.metadata["current_model"] = model_info["versioned_name"]
            
            # 添加到历史记录
            self.metadata["model_history"].append({
                "action": "load",
                "model_name": model_name,
                "version": model_info["version"],
                "timestamp": datetime.now().isoformat()
            })
            
            self._save_metadata()
            
            return model, model_info
            
        except Exception as e:
            raise RuntimeError(_("error_system_error", error=str(e)))
    
    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """回滚模型到指定版本。
        
        参数：
            model_name: 模型名称
            target_version: 目标版本
        
        返回：
            bool: 是否成功
        """
        try:
            if model_name not in self.metadata["models"]:
                return False
            
            model_versions = self.metadata["models"][model_name]
            
            # 查找目标版本
            target_info = next(
                (info for info in model_versions if info["version"] == target_version),
                None
            )
            
            if target_info is None:
                return False
            
            # 更新所有版本状态
            for info in model_versions:
                info["status"] = "archived"
            
            # 设置目标版本为活跃
            target_info["status"] = "active"
            
            # 更新当前模型
            self.metadata["current_model"] = target_info["versioned_name"]
            
            # 添加到历史记录
            self.metadata["model_history"].append({
                "action": "rollback",
                "model_name": model_name,
                "version": target_version,
                "timestamp": datetime.now().isoformat()
            })
            
            self._save_metadata()
            
            return True
            
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
            return False
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """删除模型。
        
        参数：
            model_name: 模型名称
            version: 版本号（可选，默认删除所有版本）
        
        返回：
            bool: 是否成功
        """
        try:
            if model_name not in self.metadata["models"]:
                return False
            
            if version is None:
                # 删除所有版本
                for info in self.metadata["models"][model_name]:
                    if os.path.exists(info["path"]):
                        os.remove(info["path"])
                
                del self.metadata["models"][model_name]
            else:
                # 删除指定版本
                model_versions = self.metadata["models"][model_name]
                version_info = next(
                    (info for info in model_versions if info["version"] == version),
                    None
                )
                
                if version_info is None:
                    return False
                
                # 删除文件
                if os.path.exists(version_info["path"]):
                    os.remove(version_info["path"])
                
                # 从列表中移除
                self.metadata["models"][model_name] = [
                    info for info in model_versions 
                    if info["version"] != version
                ]
                
                # 如果这是最后一个版本，删除模型记录
                if not self.metadata["models"][model_name]:
                    del self.metadata["models"][model_name]
            
            # 添加到历史记录
            self.metadata["model_history"].append({
                "action": "delete",
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now().isoformat()
            })
            
            self._save_metadata()
            
            return True
            
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型。
        
        返回：
            List[Dict[str, Any]]: 模型列表
        """
        models = []
        
        for model_name, versions in self.metadata["models"].items():
            for version_info in versions:
                model_info = {
                    "name": model_name,
                    "version": version_info["version"],
                    "versioned_name": version_info["versioned_name"],
                    "created_at": version_info["created_at"],
                    "size": version_info["size"],
                    "performance_metrics": version_info["performance_metrics"],
                    "status": version_info["status"],
                    "path": version_info["path"]
                }
                models.append(model_info)
        
        return sorted(models, key=lambda x: x["created_at"], reverse=True)
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取模型信息。
        
        参数：
            model_name: 模型名称
            version: 版本号（可选）
        
        返回：
            Optional[Dict[str, Any]]: 模型信息
        """
        try:
            if model_name not in self.metadata["models"]:
                return None
            
            model_versions = self.metadata["models"][model_name]
            
            if version is None:
                # 返回最新版本
                return max(model_versions, key=lambda x: x["created_at"])
            else:
                # 返回指定版本
                return next(
                    (info for info in model_versions if info["version"] == version),
                    None
                )
        except Exception:
            return None
    
    def get_current_model(self) -> Optional[str]:
        """获取当前模型。
        
        返回：
            Optional[str]: 当前模型版本化名称
        """
        return self.metadata.get("current_model")
    
    def get_model_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取模型操作历史。
        
        参数：
            limit: 限制返回的历史记录数量
        
        返回：
            List[Dict[str, Any]]: 历史记录列表
        """
        history = self.metadata.get("model_history", [])
        return history[-limit:] if limit > 0 else history
    
    def _generate_version(self, model_name: str) -> str:
        """生成版本号。
        
        参数：
            model_name: 模型名称
        
        返回：
            str: 版本号
        """
        if model_name not in self.metadata["models"]:
            return "1.0.0"
        
        existing_versions = [info["version"] for info in self.metadata["models"][model_name]]
        
        # 简单的版本号生成策略
        max_version = "1.0.0"
        for version in existing_versions:
            try:
                parts = version.split('.')
                if len(parts) == 3:
                    major, minor, patch = map(int, parts)
                    current_version = f"{major}.{minor}.{patch}"
                    if version > max_version:
                        max_version = version
            except ValueError:
                continue
        
        # 增加补丁版本号
        parts = max_version.split('.')
        major, minor, patch = map(int, parts)
        return f"{major}.{minor}.{patch + 1}"
    
    def export_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """导出模型元数据。
        
        参数：
            model_name: 模型名称
            version: 版本号（可选）
        
        返回：
            Dict[str, Any]: 模型元数据
        """
        model_info = self.get_model_info(model_name, version)
        if model_info is None:
            return {}
        
        return {
            "model_info": model_info,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0"
        }


def create_model_manager() -> ModelManager:
    """创建模型管理器实例。
    
    返回：
        ModelManager: 模型管理器实例
    """
    return ModelManager()


def handle_model_save(
    model: Any,
    model_name: str,
    performance_metrics: Optional[Dict[str, float]] = None,
    training_metadata: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> None:
    """处理模型保存。
    
    参数：
        model: 模型对象
        model_name: 模型名称
        performance_metrics: 性能指标
        training_metadata: 训练元数据
        feature_names: 特征名称列表
        model_params: 模型参数
    """
    try:
        manager = create_model_manager()
        model_path = manager.save_model(
            model=model,
            model_name=model_name,
            performance_metrics=performance_metrics,
            training_metadata=training_metadata,
            feature_names=feature_names,
            model_params=model_params
        )
        
        st.success(_("model_saved"))
        
    except Exception as e:
        st.error(_("error_system_error", error=str(e)))


def handle_model_load(model_name: str, version: Optional[str] = None) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """处理模型加载。
    
    参数：
        model_name: 模型名称
        version: 版本号（可选）
    
    返回：
        Optional[Tuple[Any, Dict[str, Any]]]: (模型对象, 模型信息) 或 None
    """
    try:
        manager = create_model_manager()
        model, model_info = manager.load_model(model_name, version)
        
        st.success(_("model_loaded"))
        return model, model_info
        
    except Exception as e:
        st.error(_("error_system_error", error=str(e)))
        return None