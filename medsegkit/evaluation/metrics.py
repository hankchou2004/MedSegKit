"""Evaluation utilities — multi-model comparison table.

Computes per-class and mean Dice, HD95, NSD for one or more models
and prints a formatted comparison table.
"""

from __future__ import annotations

import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from medsegkit.data.transforms import NUM_CLASSES as OASIS1_NUM_CLASSES
from medsegkit.data.btcv_transforms import NUM_CLASSES as BTCV_NUM_CLASSES


# ── class name tables ─────────────────────────────────────────────────────────

# OASIS-1 FreeSurfer aseg: remapped indices 0–40
# Corresponds to FS_LABEL_SRC → FS_LABEL_DST in data/transforms.py
OASIS1_CLASS_NAMES = [
    "Background",                                   # 0  (src: 0, 255)
    "L-Cerebral-WM",    "L-Cerebral-Ctx",           # 1–2   (src: 2, 3)
    "L-Lat-Vent",       "L-Inf-Lat-Vent",           # 3–4   (src: 4, 5)
    "L-Cereb-WM",       "L-Cereb-Ctx",              # 5–6   (src: 7, 8)
    "L-Thalamus",       "L-Caudate",                # 7–8   (src: 10, 11)
    "L-Putamen",        "L-Pallidum",               # 9–10  (src: 12, 13)
    "3rd-Vent",         "4th-Vent",                 # 11–12 (src: 14, 15)
    "Brain-Stem",                                   # 13    (src: 16)
    "L-Hippocampus",    "L-Amygdala",               # 14–15 (src: 17, 18)
    "CSF",                                          # 16    (src: 24)
    "L-Accumbens",      "L-VentralDC",              # 17–18 (src: 26, 28)
    "L-Vessel",                                     # 19    (src: 30)
    "R-Cerebral-WM",    "R-Cerebral-Ctx",           # 20–21 (src: 41, 42)
    "R-Lat-Vent",       "R-Inf-Lat-Vent",           # 22–23 (src: 43, 44)
    "R-Cereb-WM",       "R-Cereb-Ctx",              # 24–25 (src: 46, 47)
    "R-Thalamus",       "R-Caudate",                # 26–27 (src: 49, 50)
    "R-Putamen",        "R-Pallidum",               # 28–29 (src: 51, 52)
    "R-Hippocampus",    "R-Amygdala",               # 30–31 (src: 53, 54)
    "R-Accumbens",      "R-VentralDC",              # 32–33 (src: 58, 60)
    "R-Vessel",                                     # 34    (src: 62)
    "5th-Vent",                                     # 35    (src: 72)
    "L-WM-Hypo",        "R-WM-Hypo",               # 36–37 (src: 78, 79)
    "L-nonWM-Hypo",     "R-nonWM-Hypo",            # 38–39 (src: 81, 82)
    "Optic-Chiasm",                                 # 40    (src: 85)
]  # 41 entries

# BTCV abdominal CT: labels 0–13
BTCV_CLASS_NAMES = [
    "Background",
    "Spleen", "Right Kidney", "Left Kidney", "Gallbladder",
    "Esophagus", "Liver", "Stomach", "Aorta",
    "Inf Vena Cava", "Portal Vein", "Pancreas",
    "R Adrenal Gland", "L Adrenal Gland",
]  # 14 entries


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device:      str   = "cuda",
    patch_size:  tuple = (128, 128, 128),
    num_classes: int   = OASIS1_NUM_CLASSES,
) -> dict:
    """Run sliding-window inference and return metric dict.

    Returns:
        {
            "mean_dice":      float,
            "mean_hd95":      float,
            "mean_nsd":       float,
            "per_class_dice": list[float],   # length num_classes-1 (no bg)
            "per_class_hd95": list[float],
        }
    """
    model.eval().to(device)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)

    dice_m = DiceMetric(include_background=False, reduction="none")
    hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    nsd_m  = SurfaceDistanceMetric(include_background=False, reduction="none")

    post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds  = inferer(inputs=images, network=model)

            pl = [post_pred(p)  for p in decollate_batch(preds)]
            ll = [post_label(l) for l in decollate_batch(labels)]
            dice_m(pl, ll)
            hd95_m(pl, ll)
            nsd_m(pl, ll)

    dice_pc = dice_m.aggregate().nanmean(dim=0).cpu().numpy()   # (C-1,)
    hd95_pc = hd95_m.aggregate().nanmean(dim=0).cpu().numpy()
    nsd_pc  = nsd_m.aggregate().nanmean(dim=0).cpu().numpy()

    return {
        "mean_dice":      float(np.nanmean(dice_pc)),
        "mean_hd95":      float(np.nanmean(hd95_pc)),
        "mean_nsd":       float(np.nanmean(nsd_pc)),
        "per_class_dice": dice_pc.tolist(),
        "per_class_hd95": hd95_pc.tolist(),
    }


# ── display ───────────────────────────────────────────────────────────────────

def print_comparison_table(
    results:     dict[str, dict],
    class_names: list[str] | None = None,
) -> None:
    """Print summary + optional per-class breakdown.

    Args:
        results:     {"model_name": evaluate_model(...), ...}
        class_names: If provided, also prints per-class Dice table.
                     Use OASIS1_CLASS_NAMES or BTCV_CLASS_NAMES.
    """
    # ── summary table ────────────────────────────────────────────────
    header = f"{'Model':<20} {'Mean Dice':>10} {'Mean HD95':>10} {'Mean NSD':>10}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, r in results.items():
        print(
            f"{name:<20} {r['mean_dice']:>10.4f} "
            f"{r['mean_hd95']:>10.2f} {r['mean_nsd']:>10.4f}"
        )
    print(sep)

    if class_names is None:
        return

    # ── per-class Dice breakdown ─────────────────────────────────────
    model_names = list(results.keys())
    # class_names[1:] to skip background (metrics exclude background)
    fg_names = class_names[1:]

    col_w   = 22
    val_w   = 8
    hdr = f"{'Class':<{col_w}}" + "".join(f"{n:>{val_w}}" for n in model_names)
    sep2 = "-" * len(hdr)
    print(f"\nPer-class Dice (background excluded)\n{sep2}\n{hdr}\n{sep2}")

    for i, cls in enumerate(fg_names):
        row = f"{cls:<{col_w}}"
        for name in model_names:
            pc = results[name].get("per_class_dice", [])
            val = pc[i] if i < len(pc) else float("nan")
            row += f"{val:>{val_w}.4f}"
        print(row)
    print(sep2 + "\n")
