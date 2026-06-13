# experiments/ — Claude Code 上下文

## 用途 / Purpose

訓練、評估、推論的主要進入點腳本。每個腳本讀取 YAML config，自動根據 `data.dataset` 選擇對應 DataModule。

Main entry-point scripts for training, evaluation, and inference. Each script reads a YAML config and auto-selects the DataModule based on `data.dataset`.

## 腳本說明 / Scripts

| 腳本 | 用途 |
|------|------|
| `train.py` | 訓練（一般分割 or KD），自動 val loop，存 best checkpoint |
| `evaluate.py` | 多模型 Dice/HD95 比較；BTCV 自動改用 val split |
| `inference.py` | 批次推論，輸出 `.nii.gz`；支援無標籤 test split |

## DataModule 路由 / DataModule Routing

`build_datamodule(cfg)` 在 `train.py`，`build_eval_loader(cfg)` 在 `evaluate.py`：

```
data.dataset == "btcv"  → BTCVDataModule
data.dataset == "oasis1" (or 未指定) → OasisDataModule
```

新增資料集時，在 `train.py` 的 `build_datamodule()` 和 `evaluate.py` 的 `build_eval_loader()` 各加一個 `elif`。

## 重要注意 / Important Notes

- `train.py` 使用 `m_cfg.get("name")` 而非 `pop`，不會就地修改 dict
- BTCV test split（20 筆）無標籤：`evaluate.py` 改用 val split 並印出提示；`inference.py` 可直接跑 test split
- checkpoint 由 `ModelCheckpoint` 存在 `logs/{exp_name}/best_model.ckpt`
- `EarlyStopping` 監控 `val/dice`，patience=50
