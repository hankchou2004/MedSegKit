# medsegkit/losses/ — 技能 / Skills

## 分割損失 / Segmentation Losses

```python
from medsegkit.losses.seg_losses import build_seg_loss

loss_fn = build_seg_loss("dice_ce")   # 最常用
loss_fn = build_seg_loss("dice")
loss_fn = build_seg_loss("focal")
loss_fn = build_seg_loss("tversky")

# 使用
logits = model(image)          # (B, C, H, W, D)
label  = batch["label"]        # (B, 1, H, W, D)  整數
loss   = loss_fn(logits, label)
```

## 知識蒸餾損失 / KD Losses

```python
from medsegkit.losses.kd.response_kd    import ResponseKDLoss
from medsegkit.losses.kd.feature_kd     import FeatureKDLoss
from medsegkit.losses.kd.contrastive_kd import ContrastiveKDLoss

# Response KD（soft label）
kd_loss = ResponseKDLoss(temperature=4.0)
loss = kd_loss(student_logits, teacher_logits)

# Feature KD（中間層）
kd_loss = FeatureKDLoss(mode="l2")   # "l2" | "attention"
loss = kd_loss(student_feat, teacher_feat)

# Contrastive KD（InfoNCE）
kd_loss = ContrastiveKDLoss(temperature=0.07)
loss = kd_loss(student_feat, teacher_feat)
```

## KDModule 中的損失組合 / Combined Loss in KDModule

```
total_loss = (1 - kd_weight) × seg_loss(student_pred, label)
           +      kd_weight  × kd_loss(student_logits, teacher_logits)
```

`combined` 模式：`kd_loss = response_loss + feature_loss`
