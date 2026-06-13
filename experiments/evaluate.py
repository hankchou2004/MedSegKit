"""Multi-model comparison evaluation script.

Usage:
    python experiments/evaluate.py \
        --ckpts unet:logs/unet_oasis1/best_model.ckpt \
                dynunet:logs/dynunet_oasis1/best_model.ckpt \
        --config configs/unet.yaml

Note: BTCV test split has no labels. For BTCV, evaluation uses the val split
(6 labelled cases). OASIS-1 uses the test split (85 labelled cases) as usual.
"""

from __future__ import annotations

import argparse

import yaml

from medsegkit.data.oasis_module import OasisDataModule
from medsegkit.data.btcv_module import BTCVDataModule
from medsegkit.data.transforms import NUM_CLASSES as OASIS1_NUM_CLASSES
from medsegkit.data.btcv_transforms import NUM_CLASSES as BTCV_NUM_CLASSES
from medsegkit.engine.seg_module import SegModule
from medsegkit.evaluation.metrics import (
    evaluate_model, print_comparison_table,
    OASIS1_CLASS_NAMES, BTCV_CLASS_NAMES,
)


def build_eval_loader(cfg: dict):
    dataset = cfg["data"].get("dataset", "oasis1")
    data_kw = {k: v for k, v in cfg["data"].items() if k != "dataset"}

    if dataset == "btcv":
        dm = BTCVDataModule(**data_kw)
        dm.setup(stage="fit")
        print("[evaluate] BTCV: using val split (6 labelled cases) — test split has no labels.")
        return dm.val_dataloader()

    dm = OasisDataModule(**data_kw)
    dm.setup(stage="test")
    return dm.test_dataloader()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts", nargs="+", metavar="NAME:PATH",
        help='List of "model_name:ckpt_path" pairs',
    )
    parser.add_argument("--config",    required=True)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--per-class", action="store_true",
                        help="Print per-class Dice breakdown")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["data"].get("dataset", "oasis1")
    num_classes  = BTCV_NUM_CLASSES  if dataset == "btcv" else OASIS1_NUM_CLASSES
    class_names  = BTCV_CLASS_NAMES  if dataset == "btcv" else OASIS1_CLASS_NAMES

    loader = build_eval_loader(cfg)

    results = {}
    for entry in args.ckpts:
        name, ckpt_path = entry.split(":", 1)
        print(f"\nLoading {name} from {ckpt_path} ...")
        module = SegModule.load_from_checkpoint(ckpt_path, map_location=args.device)
        results[name] = evaluate_model(
            module.model,
            loader,
            device=args.device,
            patch_size=cfg["data"].get("patch_size", [128, 128, 128]),
            num_classes=num_classes,
        )

    print_comparison_table(
        results,
        class_names=class_names if args.per_class else None,
    )


if __name__ == "__main__":
    main()
