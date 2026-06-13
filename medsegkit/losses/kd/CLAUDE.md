# medsegkit/losses/kd/ — Claude Code 上下文

## 用途 / Purpose

知識蒸餾損失實作。每個檔案對應一種 KD 策略，由 `medsegkit/engine/kd_module.py` 依 `kd_type` 選擇。

KD loss implementations. Each file corresponds to one KD strategy, selected by `kd_module.py` based on `kd_type`.

## 檔案說明 / Files

| 檔案 | kd_type | 損失方式 |
|------|---------|---------|
| `response_kd.py` | `"response"` | KL divergence，teacher soft label 蒸餾 |
| `feature_kd.py` | `"feature"` | L2 或 Attention Transfer（中間層特徵） |
| `contrastive_kd.py` | `"contrastive"` | InfoNCE / CRD |

## Response KD 細節 / Response KD Details

- Temperature scaling：T>1 軟化 softmax，讓 student 學習 teacher 的相對置信度
- 推薦 T=4（原始 Hinton 論文設定）
- 損失 = `KL(softmax(s/T) || softmax(t/T)) * T²`（T² 補償梯度縮放）

## Feature KD 細節 / Feature KD Details

- `mode="l2"`：直接 L2 距離（teacher/student 特徵必須同尺寸）
- `mode="attention"`：Attention Transfer，先將特徵 sum over channel，正規化後 L2

## Contrastive KD 細節 / Contrastive KD Details

- InfoNCE：student 特徵靠近 teacher（正例），遠離其他 batch 樣本（負例）
- Temperature 較低（0.07）使對比更 sharp
