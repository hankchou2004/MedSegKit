# medsegkit/ — 技能 / Skills

## 公開介面 / Public Interface

```python
import medsegkit

# 列出所有已註冊模型
medsegkit.list_models()
# ['attention_unet', 'dynunet', 'medsam', 'mednext', 'segresnet',
#  'swin_unetr', 'umamba', 'unetr', 'unet', 'unet_pp']

# 建立模型（全參數）
model = medsegkit.build_model(
    "medsam",
    in_channels=1,
    out_channels=14,
    checkpoint="/path/to/medsam_vit_b.pth",
    freeze_encoder=True,
    slice_axis=2,
    slice_batch=8,
)

# 自訂模型註冊
@medsegkit.register_model("my_model")
class MyModel(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=14, **kw):
        ...
```

## 子模組技能 / Sub-module Skills

- **models/** → `build_model()`, `list_models()`, `register_model()`
- **data/** → `OasisDataModule`, `BTCVDataModule`, `build_transforms()`, `build_btcv_transforms()`
- **engine/** → `SegModule`, `KDModule`（Lightning Module）
- **losses/** → `build_seg_loss()`（dice / dice_ce / focal / tversky）
- **losses/kd/** → `ResponseKDLoss`, `FeatureKDLoss`, `ContrastiveKDLoss`
- **evaluation/** → `evaluate_model()`, `print_comparison_table()`
