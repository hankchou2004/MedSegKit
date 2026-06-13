# configs/kd/ — Claude Code 上下文

## 用途 / Purpose

存放知識蒸餾（Knowledge Distillation）訓練設定。需搭配 `--mode kd` 參數執行 `experiments/train.py`。

Stores Knowledge Distillation training configs. Must be run with `--mode kd` flag.

## KD 設定結構 / Config Structure

```yaml
experiment:
  name: kd_dynunet_to_unet
  log_dir: ./logs

kd:
  teacher:
    name: dynunet             # Registry key（已訓練好的大模型）
    ckpt: /path/to/teacher.ckpt
  student:
    name: unet                # Registry key（要訓練的小模型）
    in_channels: 1
    out_channels: 41
  type: response              # response | feature | contrastive | combined
  kd_weight: 0.5              # KD loss 佔比（1 - kd_weight = seg loss 佔比）
  temperature: 4.0            # soft label 溫度（response KD 用）

data:
  dataset: oasis1             # oasis1 | btcv
  ...（同一般訓練設定）

training:
  max_epochs: 300
  lr: 1.0e-4
  seg_loss: dice_ce
```

## KD 模式說明 / KD Type Reference

| type | 損失 | 適合情境 |
|------|------|---------|
| `response` | KL divergence（soft label） | 最通用，計算快 |
| `feature` | L2 / Attention Transfer（中間層） | teacher/student 架構接近時效果好 |
| `contrastive` | InfoNCE / CRD | 特徵空間差異大時 |
| `combined` | response + feature 同時 | 最強但最慢 |
