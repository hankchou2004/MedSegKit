# medsegkit/evaluation/ — Claude Code 上下文

## 用途 / Purpose

多模型指標比較工具。`evaluate_model()` 對單一模型跑完整 test/val set 並回傳指標；`print_comparison_table()` 印出比較表格。

Multi-model metrics comparison. `evaluate_model()` runs a full test/val set for one model and returns metrics; `print_comparison_table()` prints a comparison table.

## 指標 / Metrics

| 指標 | 說明 | 越高/低越好 |
|------|------|-----------|
| Dice | 體積重疊率（0–1） | 越高越好 |
| HD95 | Hausdorff Distance 95%（mm） | 越低越好 |
| NSD | Normalized Surface Distance | 越高越好 |

- 所有指標 `include_background=False`（排除 class 0 背景）
- 計算多類別平均值

## 使用限制 / Limitations

- 需要有 label 的 dataloader（BTCV test split 無法使用）
- `evaluate_model()` 使用 `SlidingWindowInferer`，patch_size 從 config 讀取
- `SegModule.load_from_checkpoint()` 會自動還原訓練時的模型參數
