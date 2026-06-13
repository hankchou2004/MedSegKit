# medsegkit/losses/ — Claude Code 上下文

## 用途 / Purpose

分割損失函數工廠與知識蒸餾損失。`seg_losses.py` 統一包裝 MONAI 損失；`kd/` 子目錄包含各 KD 損失實作。

Segmentation loss factory and KD losses. `seg_losses.py` wraps MONAI losses uniformly; `kd/` subdirectory contains KD loss implementations.

## 檔案說明 / Files

| 檔案 | 用途 |
|------|------|
| `seg_losses.py` | `build_seg_loss(name, **kw)` 工廠函數 |
| `kd/response_kd.py` | KL divergence soft-label KD（Hinton 2015） |
| `kd/feature_kd.py` | L2 / Attention Transfer 特徵層對齊 |
| `kd/contrastive_kd.py` | InfoNCE / CRD 對比學習 |

## 分割損失 / Seg Loss Options

| name | 類別 | 適合情境 |
|------|------|---------|
| `"dice"` | `DiceLoss` | 類別不平衡，標準選擇 |
| `"dice_ce"` | `DiceCELoss` | 最常用，Dice + CrossEntropy |
| `"focal"` | `FocalLoss` | 嚴重類別不平衡（如小器官） |
| `"tversky"` | `TverskyLoss` | 調整 FP/FN 權重 |

## 慣例 / Conventions

- 所有損失函數接受 `(pred, label)` 輸入，`pred` 為 logits（未 softmax）
- `label` 形狀：`(B, 1, H, W, D)`（整數類別）
- 不要直接 import MONAI 損失類別用於訓練，統一走 `build_seg_loss()`
