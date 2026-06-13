# experiments/ — 技能 / Skills

## train.py

```bash
# 一般分割訓練
python experiments/train.py --config configs/btcv_dynunet.yaml
python experiments/train.py --config configs/btcv_medsam.yaml --gpus 1

# 知識蒸餾訓練
python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
```

輸出：`logs/{experiment.name}/best_model.ckpt`（val/dice 最高時存檔）

## evaluate.py

```bash
python experiments/evaluate.py \
    --config configs/btcv_unet.yaml \
    --ckpts  unet:logs/unet_btcv/best_model.ckpt \
             dynunet:logs/dynunet_btcv/best_model.ckpt
```

- 輸出多模型比較表（Dice / HD95）
- BTCV 自動改用 val split（6 筆有標籤）

## inference.py

```bash
# Dataset split 批次推論
python experiments/inference.py \
    --config configs/btcv_dynunet.yaml \
    --ckpt   logs/dynunet_btcv/best_model.ckpt \
    --split  test \
    --out    ./predictions/btcv_test

# 單一影像
python experiments/inference.py \
    --config configs/btcv_dynunet.yaml \
    --ckpt   logs/dynunet_btcv/best_model.ckpt \
    --image  /path/to/img0061.nii.gz \
    --out    ./predictions
```

- 輸出：`{out}/{stem}_pred.nii.gz`（重採樣後空間，保留 affine）
- BTCV test split 無標籤時印出提示（不影響執行）

## 參數總覽 / Argument Summary

| 腳本 | 參數 | 說明 |
|------|------|------|
| `train.py` | `--config` | YAML 路徑（必填） |
| | `--mode` | `seg`（預設）或 `kd` |
| | `--gpus` | GPU 數量（預設 1） |
| `evaluate.py` | `--config` | YAML 路徑（必填） |
| | `--ckpts` | `name:path` 清單（必填） |
| | `--device` | `cuda`（預設）或 `cpu` |
| `inference.py` | `--config` | YAML 路徑（必填） |
| | `--ckpt` | checkpoint 路徑（必填） |
| | `--out` | 輸出目錄（必填） |
| | `--split` | `train`/`val`/`test`（預設 `test`） |
| | `--image` | 單一影像路徑（覆蓋 `--split`） |
| | `--device` | `cuda`（預設）或 `cpu` |
