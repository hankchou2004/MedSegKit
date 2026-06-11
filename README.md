# BrainSegKit

> 腦部 MRI 分割研究框架 — 整合 MONAI 與 PyTorch Lightning，專為多模型架構比較與知識蒸餾實驗設計。

---

## 框架定位

```
BrainSegKit（你的研究框架）
      ↓ 使用
MONAI 1.5.2          — 模型架構、資料讀取、評估指標
PyTorch Lightning 2.6 — 訓練迴圈、Trainer、checkpoint
      ↓ 運行於
PyTorch 2.11 + CUDA 12.8 + RTX 5070
```

BrainSegKit **不複製或修改** MONAI / Lightning 原始碼，而是將兩者作為底層 library，在其上建立：
- 統一模型切換介面（Model Registry）
- FreeSurfer 41 類腦區 aseg 專用資料流（OASIS-1）
- 知識蒸餾訓練框架（4 種 KD 模式）
- 多模型比較評估流程

---

## 支援的模型

| 名稱 | 呼叫 key | 來源 | 說明 |
|------|---------|------|------|
| UNet | `"unet"` | MONAI | 標準 3D UNet，residual units |
| UNet++ | `"unet_pp"` | MONAI | Nested UNet，密集 skip connections |
| Attention UNet | `"attention_unet"` | MONAI | 加入 attention gate 的 UNet |
| DynUNet | `"dynunet"` | MONAI | nnUNet 架構，支援 deep supervision |
| SwinUNETR | `"swin_unetr"` | MONAI | Swin Transformer encoder + UNet decoder |
| MedNeXt | `"mednext"` | MONAI | ConvNeXt-style 醫學影像分割 |
| SegResNet | `"segresnet"` | MONAI | ResNet-style encoder-decoder |
| UNETR | `"unetr"` | MONAI | ViT encoder + UNet decoder |
| **U-Mamba** | `"umamba"` | 自製 | Mamba SSM bottleneck，需安裝 `mamba-ssm` |

---

## 知識蒸餾（KD）框架

支援 4 種 KD 模式，透過 config YAML 的 `kd.type` 切換：

```
Teacher（大模型，凍結）
    │
    ├── response    → KL divergence（soft label + temperature scaling）
    ├── feature     → L2 / L1 / Attention Transfer（中間層特徵對齊）
    ├── contrastive → InfoNCE / CRD（teacher/student 特徵對比）
    └── combined    → response + feature 同時啟用
    ↓
Student（小模型，訓練）
```

---

## 資料集：OASIS-1 FreeSurfer 41 類分割

### 原始資料結構

```
brain_data/
└── oasis1_freesurfer/
    └── {subject}/mri/
        ├── T1.mgz      ← T1 加權影像
        └── aseg.mgz    ← FreeSurfer 自動腦區分割
```

### 轉換後的資料集結構（nnUNet 格式）

```
dataset/
├── splits.json          ← train 255 / val 85 / test 85
└── freesurfer/
    ├── imagesTr/        ← 340 個 T1 .nii.gz（train + val）
    ├── labelsTr/        ← 340 個 aseg .nii.gz（train + val）
    ├── imagesTs/        ←  85 個 T1 .nii.gz（test）
    └── labelsTs/        ←  85 個 aseg .nii.gz（test）
```

### FreeSurfer aseg 標籤（41 類）

原始 FreeSurfer 標籤值（非連續整數）在訓練前由 `transforms.py` 自動重映射至 0–40：

| 重映射後 | 原始值 | 結構名稱 |
|---------|--------|---------|
| 0 | 0, 255 | Background / Unknown |
| 1 | 2 | Left Cerebral White Matter |
| 2 | 3 | Left Cerebral Cortex |
| 3 | 4 | Left Lateral Ventricle |
| 4 | 5 | Left Inf Lateral Ventricle |
| 5 | 7 | Left Cerebellum White Matter |
| ... | ... | ... |
| 20 | 41 | Right Cerebral White Matter |
| 21 | 42 | Right Cerebral Cortex |
| ... | ... | ... |
| 40 | 85 | Optic Chiasm |

完整對照見 `brainsegkit/data/transforms.py` 的 `FS_LABEL_SRC / FS_LABEL_DST`。

---

## 快速開始

### 1. 安裝環境

```bash
conda create -n brain_segmention python=3.10 -y
conda activate brain_segmention

# PyTorch 2.11 + CUDA 12.8（RTX 5070 / sm_120）
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install "monai[all]==1.5.2" "lightning==2.6.5"
pip install nibabel SimpleITK scipy scikit-learn einops timm wandb

# 安裝 BrainSegKit（editable）
pip install -e /path/to/BrainSegKit

# 可選：U-Mamba
pip install mamba-ssm causal-conv1d
```

### 2. 轉換資料集

```bash
conda activate brain_segmention
python scripts/convert_dataset.py
# 執行後 brain_data/ 自動刪除，保留 dataset/freesurfer/
```

### 3. 訓練

```bash
cd /path/to/BrainSegKit

# 一般分割訓練
python experiments/train.py --config configs/unet.yaml
python experiments/train.py --config configs/dynunet.yaml
python experiments/train.py --config configs/mednext.yaml

# 知識蒸餾
python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
```

### 4. 多模型比較評估

```bash
python experiments/evaluate.py \
    --config configs/unet.yaml \
    --ckpts unet:logs/unet_oasis1/best_model.ckpt \
            dynunet:logs/dynunet_oasis1/best_model.ckpt \
            mednext:logs/mednext_oasis1/best_model.ckpt
```

### 5. 資料集視覺化

```bash
# 互動視窗（需本機 display / WSLg）
python scripts/brain_viewer.py --split train

# SSH 無頭模式：存 PNG（不需要 display）
python scripts/brain_viewer.py --save --case 0 --split train --save-dir ./viewer_out
python scripts/brain_viewer.py --save --case 5 --split val   --save-dir ./viewer_out
```

視窗快捷鍵：

| 按鍵 | 功能 |
|------|------|
| `← / →` | 切換 case |
| `A / N` | 全部 labels 開 / 關 |
| `H` | 切換 Hippocampus（左右） |
| `V` | 切換 Ventricles |
| `C` | 切換 Cortex（左右） |
| `1` | 只顯示 White Matter |
| `2` | 只顯示 Gray Matter |
| `3` | 只顯示 CSF / Ventricles |
| `Scroll` | 縮放 |
| `Click`（panel） | 切換單一 label |

### Python API

```python
import brainsegkit

print(brainsegkit.list_models())
# ['attention_unet', 'dynunet', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# out_channels=41 for FreeSurfer aseg
model = brainsegkit.build_model("dynunet", in_channels=1, out_channels=41)
model = brainsegkit.build_model("mednext", in_channels=1, out_channels=41)
```

---

## 專案結構

```
BrainSegKit/
├── brainsegkit/
│   ├── models/
│   │   ├── registry.py           # @register_model 裝飾器
│   │   ├── wrappers.py           # MONAI 模型統一包裝（8 個）
│   │   └── umamba/umamba.py      # U-Mamba 自製實作
│   ├── losses/
│   │   ├── seg_losses.py         # 分割損失工廠
│   │   └── kd/
│   │       ├── response_kd.py    # Hinton soft-label KD
│   │       ├── feature_kd.py     # 特徵層對齊 KD
│   │       └── contrastive_kd.py # InfoNCE / CRD
│   ├── data/
│   │   ├── transforms.py         # MONAI 前處理 + FreeSurfer label remapping
│   │   └── oasis_module.py       # OASIS-1 LightningDataModule
│   ├── engine/
│   │   ├── seg_module.py         # 一般分割 LightningModule
│   │   └── kd_module.py          # 知識蒸餾 LightningModule
│   └── evaluation/
│       └── metrics.py            # Dice / HD95 / NSD 多模型比較
├── configs/
│   ├── unet.yaml
│   ├── unet_pp.yaml
│   ├── dynunet.yaml
│   ├── mednext.yaml
│   └── kd/dynunet_to_unet.yaml
├── experiments/
│   ├── train.py                  # 訓練入口
│   └── evaluate.py               # 評估入口
├── scripts/
│   ├── convert_dataset.py        # .mgz → nnUNet .nii.gz 轉換
│   └── brain_viewer.py           # 腦部影像互動檢視器
└── pyproject.toml
```

---

## 依賴版本

| 套件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| MONAI | 1.5.2 |
| Lightning | 2.6.5 |
| nibabel | ≥5.0 |
| einops | ≥0.6 |
| timm | ≥0.9 |
| wandb | latest |

---

## 參考文獻

- **UNet**: Ronneberger et al., MICCAI 2015
- **UNet++**: Zhou et al., MICCAI Workshop 2018
- **nnUNet / DynUNet**: Isensee et al., Nature Methods 2021
- **SwinUNETR**: Tang et al., CVPR 2022
- **MedNeXt**: Roy et al., MICCAI 2023
- **U-Mamba**: Ma et al., ArXiv 2024
- **KD (Response)**: Hinton et al., NeurIPS Workshop 2015
- **KD (Feature / FitNets)**: Romero et al., ICLR 2015
- **KD (Contrastive / CRD)**: Tian et al., ICLR 2020
- **OASIS-1**: Marcus et al., Journal of Cognitive Neuroscience 2007
