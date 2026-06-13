# medsegkit/losses/kd/ — 技能 / Skills

## 各 KD 損失使用方式 / KD Loss Usage

### Response KD（最通用）

```python
from medsegkit.losses.kd.response_kd import ResponseKDLoss

loss_fn = ResponseKDLoss(temperature=4.0)
# student_logits, teacher_logits: (B, C, H, W, D)
loss = loss_fn(student_logits, teacher_logits)
```

### Feature KD

```python
from medsegkit.losses.kd.feature_kd import FeatureKDLoss

loss_fn = FeatureKDLoss(mode="l2")        # 或 "attention"
# student_feat, teacher_feat: (B, C, H, W, D)（中間層輸出）
loss = loss_fn(student_feat, teacher_feat)
```

### Contrastive KD

```python
from medsegkit.losses.kd.contrastive_kd import ContrastiveKDLoss

loss_fn = ContrastiveKDLoss(temperature=0.07)
loss = loss_fn(student_feat, teacher_feat)
```

## 各模式優缺比較 / Mode Comparison

| 模式 | 速度 | 實作複雜度 | 效果（通常） |
|------|------|-----------|------------|
| response | 快 | 低 | 穩定，推薦起點 |
| feature | 中 | 中 | 架構相近時有效 |
| contrastive | 中 | 中 | batch size 大時更好 |
| combined | 慢 | 高 | 最強，但調參複雜 |

## 在 config 中切換 / Switching in Config

```yaml
kd:
  type: response       # 修改此行切換模式
  kd_weight: 0.5
  temperature: 4.0     # response / contrastive 用
```
