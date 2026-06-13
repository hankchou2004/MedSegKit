# configs/ — Claude Code 上下文

## 用途 / Purpose

存放所有訓練設定的 YAML 檔。`experiments/train.py` 讀取這些設定，依 `data.dataset` 選擇 DataModule，依 `model.name` 從 Registry 建立模型。

Stores all training configuration YAML files. `experiments/train.py` reads these, selects DataModule based on `data.dataset`, and builds a model via Registry using `model.name`.

## 必要欄位 / Required Fields

```yaml
model:
  name: <registry key>      # 見 medsegkit/models/__init__.py 的 list_models()
  in_channels: 1
  out_channels: <N>         # OASIS-1: 41, BTCV: 14

data:
  dataset: oasis1 | btcv
  dataset_root: /home/hank/medical_segmention/dataset
  splits_json: /path/to/splits.json
  patch_size: [H, W, D]
  spacing: [x, y, z]       # mm
  batch_size: <N>
  num_workers: 4

training:
  max_epochs: <N>
  lr: 1.0e-4
  weight_decay: 1.0e-5
  loss: dice_ce             # dice | dice_ce | focal | tversky
```

## 現有設定檔 / Existing Configs

| 檔案 | 資料集 | 模型 |
|------|--------|------|
| `unet.yaml` | OASIS-1 | UNet |
| `unet_pp.yaml` | OASIS-1 | UNet++ |
| `dynunet.yaml` | OASIS-1 | DynUNet |
| `mednext.yaml` | OASIS-1 | MedNeXt |
| `btcv_unet.yaml` | BTCV | UNet |
| `btcv_dynunet.yaml` | BTCV | DynUNet |
| `btcv_medsam.yaml` | BTCV | MedSAM |
| `kd/dynunet_to_unet.yaml` | OASIS-1 | DynUNet→UNet KD |

## 新增設定檔 / Adding a New Config

複製最接近的現有設定，修改 `model.name`、`out_channels`、`patch_size`。  
MedSAM 額外需要 `model.checkpoint`、`model.freeze_encoder`、`model.slice_axis`、`model.slice_batch`。
