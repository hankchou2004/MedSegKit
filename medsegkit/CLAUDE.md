# medsegkit/ — Claude Code 上下文

## 用途 / Purpose

MedSegKit Python 套件的根目錄。`__init__.py` 只 re-export 最常用的公開 API。

Root of the MedSegKit Python package. `__init__.py` re-exports the most commonly used public API.

## 套件結構 / Package Structure

```
medsegkit/
├── __init__.py        # 公開 API：build_model, list_models, register_model
├── models/            # 模型 Registry + 所有模型實作
├── data/              # DataModule + Transform pipeline
├── engine/            # Lightning training/KD module
├── losses/            # 分割損失 + 知識蒸餾損失
└── evaluation/        # 多模型指標比較
```

## 公開 API / Public API

```python
import medsegkit
medsegkit.build_model("dynunet", in_channels=1, out_channels=14)
medsegkit.list_models()
medsegkit.register_model("my_model")(MyModelClass)
```

## 安裝 / Installation

```bash
cd /home/hank/medical_segmention/MedSegKit
pip install -e .                    # 基本安裝
pip install -e ".[sam]"             # 含 MedSAM（segment-anything）
pip install -e ".[umamba]"          # 含 U-Mamba（mamba-ssm）
pip install -e ".[all]"             # 全部
```

## 重要慣例 / Key Conventions

- 新增模組後必須在對應的 `__init__.py` import，否則 `@register_model` 不會被執行
- `medsegkit/models/__init__.py` 是模型 Registry 的初始化點，新增模型時必須在此 import
- 不要在 `medsegkit/__init__.py` 直接 import 模型類別，只走 `build_model` / `list_models`
