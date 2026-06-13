# medsegkit/evaluation/ — 技能 / Skills

## 多模型比較 / Multi-model Comparison

```python
from medsegkit.evaluation.metrics import evaluate_model, print_comparison_table
from medsegkit.engine.seg_module  import SegModule

# 各自載入並評估
results = {}
for name, ckpt in [("unet", "logs/unet/best_model.ckpt"),
                   ("dynunet", "logs/dynunet/best_model.ckpt")]:
    module = SegModule.load_from_checkpoint(ckpt, map_location="cuda")
    results[name] = evaluate_model(
        module.model,
        val_loader,
        device="cuda",
        patch_size=(96, 96, 96),
    )

# 印出比較表
print_comparison_table(results)
# ┌──────────┬────────┬────────┐
# │ Model    │  Dice  │  HD95  │
# ├──────────┼────────┼────────┤
# │ unet     │ 0.8123 │ 12.45  │
# │ dynunet  │ 0.8456 │  9.23  │
# └──────────┴────────┴────────┘
```

## 命令列使用 / CLI Usage

```bash
python experiments/evaluate.py \
    --config configs/btcv_unet.yaml \
    --ckpts unet:logs/unet_btcv/best_model.ckpt \
            dynunet:logs/dynunet_btcv/best_model.ckpt \
            medsam:logs/medsam_btcv/best_model.ckpt
```

## 注意事項 / Notes

- OASIS-1 評估用 **test split**（85 筆）
- BTCV 評估用 **val split**（6 筆，因 test 無標籤）
- `evaluate_model()` 回傳 `{"dice": float, "hd95": float, "nsd": float}`
