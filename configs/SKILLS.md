# configs/ — 技能 / Skills

## 提供的設定模板 / Available Config Templates

### OASIS-1 腦部分割（41 類）

```bash
python experiments/train.py --config configs/unet.yaml
python experiments/train.py --config configs/dynunet.yaml
python experiments/train.py --config configs/mednext.yaml
```

### BTCV 腹部 CT 分割（14 類）

```bash
python experiments/train.py --config configs/btcv_unet.yaml
python experiments/train.py --config configs/btcv_dynunet.yaml
python experiments/train.py --config configs/btcv_medsam.yaml   # 需先下載 checkpoint
```

### 知識蒸餾

```bash
python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
```

## MedSAM 專屬欄位 / MedSAM-specific Fields

```yaml
model:
  name: medsam
  checkpoint: /path/to/medsam_vit_b.pth   # 必填，否則使用隨機初始化
  freeze_encoder: true                      # 凍結 ViT-B encoder（推薦）
  slice_axis: 2                             # 0=sagittal 1=coronal 2=axial
  slice_batch: 8                            # 降低此值可減少 OOM 風險
```

## 知識蒸餾設定結構 / KD Config Structure

```yaml
kd:
  teacher:
    name: dynunet
    ckpt: /path/to/teacher.ckpt
  student:
    name: unet
    in_channels: 1
    out_channels: 41
  type: response        # response | feature | contrastive | combined
  kd_weight: 0.5
  temperature: 4.0
```
