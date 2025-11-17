"""
批量操作模块（Batch Operations）

职责：
- 科研数据包的上传和下载
- 支持ZIP格式压缩
- 包含数据、模型、报告等文件
- 提供文件管理和版本控制

物理意义：
通过标准化的科研包格式，方便研究人员共享和复现实验结果。
"""

from __future__ import annotations

import os
import zipfile
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd

from .utils import ensure_data_dir
from .i18n import _


class BatchOperations:
    """批量操作管理器，处理科研包的上传下载。"""
    
    def __init__(self, package_dir: str = "packages"):
        """初始化批量操作管理器。
        
        参数：
            package_dir: 科研包存储目录
        """
        self.package_dir = package_dir
        ensure_data_dir(package_dir)
        self.metadata_file = os.path.join(package_dir, "packages_metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载包元数据。"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"packages": {}}
        return {"packages": {}}
    
    def _save_metadata(self) -> None:
        """保存包元数据。"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
    
    def create_research_package(self, 
                              package_name: str,
                              data_files: Dict[str, pd.DataFrame],
                              model_files: Dict[str, bytes],
                              report_files: Dict[str, str],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建科研包。
        
        参数：
            package_name: 包名称
            data_files: 数据文件字典 {filename: DataFrame}
            model_files: 模型文件字典 {filename: bytes}
            report_files: 报告文件字典 {filename: content}
            metadata: 额外元数据
        
        返回：
            str: 包文件路径
        """
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                package_dir = Path(temp_dir) / package_name
                package_dir.mkdir(exist_ok=True)
                
                # 创建数据目录
                data_dir = package_dir / "data"
                data_dir.mkdir(exist_ok=True)
                
                # 保存数据文件
                for filename, df in data_files.items():
                    file_path = data_dir / filename
                    if filename.endswith('.csv'):
                        df.to_csv(file_path, index=False, encoding='utf-8')
                    elif filename.endswith('.json'):
                        df.to_json(file_path, orient='records', force_ascii=False)
                    elif filename.endswith('.xlsx'):
                        df.to_excel(file_path, index=False)
                
                # 创建模型目录
                model_dir = package_dir / "models"
                model_dir.mkdir(exist_ok=True)
                
                # 保存模型文件
                for filename, content in model_files.items():
                    file_path = model_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(content)
                
                # 创建报告目录
                report_dir = package_dir / "reports"
                report_dir.mkdir(exist_ok=True)
                
                # 保存报告文件
                for filename, content in report_files.items():
                    file_path = report_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # 创建元数据文件
                package_metadata = {
                    "name": package_name,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "data_files": list(data_files.keys()),
                    "model_files": list(model_files.keys()),
                    "report_files": list(report_files.keys()),
                    "custom_metadata": metadata or {}
                }
                
                metadata_path = package_dir / "package_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(package_metadata, f, ensure_ascii=False, indent=2)
                
                # 创建README文件
                readme_content = self._generate_readme(package_metadata)
                readme_path = package_dir / "README.md"
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                # 创建ZIP文件
                zip_path = os.path.join(self.package_dir, f"{package_name}.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in package_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(package_dir)
                            zipf.write(file_path, arcname)
                
                # 更新元数据
                self.metadata["packages"][package_name] = {
                    "path": zip_path,
                    "created_at": package_metadata["created_at"],
                    "version": package_metadata["version"],
                    "size": os.path.getsize(zip_path),
                    "metadata": package_metadata
                }
                self._save_metadata()
                
                return zip_path
                
        except Exception as e:
            raise RuntimeError(_("package_error", error=str(e)))
    
    def extract_research_package(self, zip_file: str, extract_to: Optional[str] = None) -> Dict[str, Any]:
        """提取科研包。
        
        参数：
            zip_file: ZIP文件路径
            extract_to: 提取目录（可选）
        
        返回：
            Dict[str, Any]: 包信息
        """
        try:
            if not extract_to:
                extract_to = tempfile.mkdtemp()
            
            with zipfile.ZipFile(zip_file, 'r') as zipf:
                zipf.extractall(extract_to)
            
            # 读取元数据
            metadata_path = os.path.join(extract_to, "package_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # 加载数据文件
            data_files = {}
            data_dir = os.path.join(extract_to, "data")
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    file_path = os.path.join(data_dir, filename)
                    if filename.endswith('.csv'):
                        data_files[filename] = pd.read_csv(file_path)
                    elif filename.endswith('.json'):
                        data_files[filename] = pd.read_json(file_path)
                    elif filename.endswith('.xlsx'):
                        data_files[filename] = pd.read_excel(file_path)
            
            # 加载模型文件
            model_files = {}
            model_dir = os.path.join(extract_to, "models")
            if os.path.exists(model_dir):
                for filename in os.listdir(model_dir):
                    file_path = os.path.join(model_dir, filename)
                    with open(file_path, 'rb') as f:
                        model_files[filename] = f.read()
            
            # 加载报告文件
            report_files = {}
            report_dir = os.path.join(extract_to, "reports")
            if os.path.exists(report_dir):
                for filename in os.listdir(report_dir):
                    file_path = os.path.join(report_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        report_files[filename] = f.read()
            
            return {
                "metadata": metadata,
                "data_files": data_files,
                "model_files": model_files,
                "report_files": report_files,
                "extract_path": extract_to
            }
            
        except Exception as e:
            raise RuntimeError(_("package_error", error=str(e)))
    
    def list_packages(self) -> List[Dict[str, Any]]:
        """列出所有科研包。
        
        返回：
            List[Dict[str, Any]]: 包列表
        """
        packages = []
        for name, info in self.metadata["packages"].items():
            package_info = {
                "name": name,
                "created_at": info["created_at"],
                "version": info["version"],
                "size": info["size"],
                "path": info["path"]
            }
            packages.append(package_info)
        
        return sorted(packages, key=lambda x: x["created_at"], reverse=True)
    
    def delete_package(self, package_name: str) -> bool:
        """删除科研包。
        
        参数：
            package_name: 包名称
        
        返回：
            bool: 是否成功
        """
        try:
            if package_name in self.metadata["packages"]:
                package_info = self.metadata["packages"][package_name]
                if os.path.exists(package_info["path"]):
                    os.remove(package_info["path"])
                
                del self.metadata["packages"][package_name]
                self._save_metadata()
                return True
            return False
        except Exception as e:
            st.error(_("error_system_error", error=str(e)))
            return False
    
    def _generate_readme(self, metadata: Dict[str, Any]) -> str:
        """生成README文件内容。
        
        参数：
            metadata: 包元数据
        
        返回：
            str: README内容
        """
        return f"""# {metadata['name']} - Research Package

## 概览 (Overview)

这是一个自动生成的科研数据包，包含恒星分类研究的相关数据、模型和报告。

This is an automatically generated research package containing data, models, and reports for star classification research.

## 创建信息 (Creation Info)
- **名称 (Name)**: {metadata['name']}
- **版本 (Version)**: {metadata['version']}
- **创建时间 (Created)**: {metadata['created_at']}

## 文件内容 (Contents)

### 数据文件 (Data Files)
{chr(10).join(f"- {filename}" for filename in metadata['data_files'])}

### 模型文件 (Model Files)
{chr(10).join(f"- {filename}" for filename in metadata['model_files'])}

### 报告文件 (Report Files)
{chr(10).join(f"- {filename}" for filename in metadata['report_files'])}

## 使用说明 (Usage Instructions)

### 数据加载 (Data Loading)
```python
import pandas as pd

# 加载CSV数据
# Load CSV data
df = pd.read_csv('data/your_data.csv')

# 加载JSON数据
# Load JSON data
df = pd.read_json('data/your_data.json')

# 加载Excel数据
# Load Excel data
df = pd.read_excel('data/your_data.xlsx')
```

### 模型加载 (Model Loading)
```python
import joblib

# 加载模型
# Load model
model = joblib.load('models/your_model.pkl')
```

## 元数据 (Metadata)
```json
{json.dumps(metadata, ensure_ascii=False, indent=2)}
```

## 注意事项 (Notes)
- 请确保在使用前检查数据质量
- 模型可能需要重新训练以适应新的数据
- 报告中的结论基于特定的实验设置

Please ensure to check data quality before use. Models may need retraining for new data. Report conclusions are based on specific experimental settings.
"""


def create_batch_operations() -> BatchOperations:
    """创建批量操作管理器实例。
    
    返回：
        BatchOperations: 批量操作管理器实例
    """
    return BatchOperations()


def handle_package_upload() -> Optional[Dict[str, Any]]:
    """处理科研包上传。
    
    返回：
        Optional[Dict[str, Any]]: 包信息或None
    """
    uploaded_file = st.file_uploader(
        _("upload_zip"),
        type=['zip'],
        help=_("help_upload_format")
    )
    
    if uploaded_file is not None:
        try:
            # 保存上传的文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 提取包
            batch_ops = create_batch_operations()
            package_info = batch_ops.extract_research_package(tmp_path)
            
            # 清理临时文件
            os.unlink(tmp_path)
            
            st.success(_("success_load"))
            return package_info
            
        except Exception as e:
            st.error(_("package_error", error=str(e)))
            return None
    
    return None


def handle_package_download(
    package_name: str,
    data_files: Dict[str, pd.DataFrame],
    model_files: Dict[str, bytes],
    report_files: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """处理科研包下载。
    
    参数：
        package_name: 包名称
        data_files: 数据文件
        model_files: 模型文件
        report_files: 报告文件
        metadata: 元数据
    """
    try:
        batch_ops = create_batch_operations()
        zip_path = batch_ops.create_research_package(
            package_name=package_name,
            data_files=data_files,
            model_files=model_files,
            report_files=report_files,
            metadata=metadata
        )
        
        # 提供下载
        with open(zip_path, 'rb') as f:
            zip_bytes = f.read()
        
        st.download_button(
            label=_("download_zip"),
            data=zip_bytes,
            file_name=f"{package_name}.zip",
            mime="application/zip",
            help=_("tooltip_download_zip")
        )
        
        st.success(_("package_created"))
        
    except Exception as e:
        st.error(_("package_error", error=str(e)))