# configs/kd/ — 技能 / Skills

## 知識蒸餾訓練 / KD Training

```bash
python experiments/train.py \
    --config configs/kd/dynunet_to_unet.yaml \
    --mode kd
```

## 四種蒸餾模式 / Four KD Modes

### Response KD（推薦入門）
- Teacher soft label → Student，溫度 T=4 softmax
- 損失：`KL(student_soft || teacher_soft) + ce(student, label)`

### Feature KD
- 對齊中間層特徵（L2 / Attention Transfer）
- 要求 teacher / student 特徵圖空間尺寸相容

### Contrastive KD
- InfoNCE 對比學習，teacher 特徵為正例
- 特別適合 teacher/student 架構差異大時

### Combined KD
- response + feature 同時啟用
- `kd_weight` 平衡 KD vs seg loss

## 常見組合 / Common Pairings

| Teacher | Student | 用途 |
|---------|---------|------|
| DynUNet（20M） | UNet（19M） | 速度優化 |
| SwinUNETR（62M） | UNet（19M） | 精度→速度 |
| MedNeXt（18M） | SegResNet（5M） | 輕量化部署 |
