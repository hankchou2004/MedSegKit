"""
將 OASIS-1 資料轉換為雙版本 .nii.gz 資料集，共用同一份 splits.json。

FreeSurfer 版本（41 類）：
    T1.mgz  →  dataset/freesurfer/imagesTr/{subject}_0000.nii.gz
    aseg.mgz →  dataset/freesurfer/labelsTr/{subject}.nii.gz

FSL 版本（3 類：1=CSF 2=GM 3=WM）：
    T1.mgz        →  dataset/fsl/imagesTr/{subject}_0000.nii.gz  (同影像)
    *_fseg.hdr    →  dataset/fsl/labelsTr/{subject}.nii.gz

輸出目錄：
    dataset/
        freesurfer/
            imagesTr/  labelsTr/  imagesTs/  labelsTs/
        fsl/
            imagesTr/  labelsTr/  imagesTs/  labelsTs/

執行：
    python convert_dataset.py [--skip-freesurfer] [--skip-fsl] [--no-delete]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


BASE        = Path('/home/hank/medical_segmention')
FS_ROOT     = BASE / 'brain_data' / 'oasis1_freesurfer'
FSL_ROOT    = BASE / 'brain_data' / 'oasis1'
SPLITS_CSV  = BASE / 'dataset' / 'splits.json'
OUT_ROOT    = BASE / 'dataset'


# ── 工具函式 ────────────────────────────────────────────────────────────────

def mgz_to_nii(src: Path, dst: Path, dtype):
    img  = nib.load(str(src))
    data = np.asarray(img.dataobj, dtype=dtype)
    nib.save(nib.Nifti1Image(data, img.affine), str(dst))


def analyze_to_nii(hdr_path: Path, dst: Path, dtype):
    """ANALYZE .hdr/.img pair → .nii.gz"""
    img  = nib.load(str(hdr_path))
    data = np.asarray(img.dataobj, dtype=dtype)
    nib.save(nib.Nifti1Image(data, img.affine), str(dst))


def find_fseg(sid: str) -> Path | None:
    """找 FSL_SEG 資料夾內的 *_fseg.hdr 檔案。"""
    fseg_dir = FSL_ROOT / sid / 'FSL_SEG'
    if not fseg_dir.exists():
        return None
    candidates = list(fseg_dir.glob('*_fseg.hdr'))
    return candidates[0] if candidates else None


def out_dirs(modality: str, split: str) -> tuple[Path, Path]:
    """回傳 (image_dir, label_dir)"""
    if split in ('train', 'val'):
        return (OUT_ROOT / modality / 'imagesTr',
                OUT_ROOT / modality / 'labelsTr')
    else:
        return (OUT_ROOT / modality / 'imagesTs',
                OUT_ROOT / modality / 'labelsTs')


# ── 主程式 ──────────────────────────────────────────────────────────────────

def convert_freesurfer(splits: dict[str, str]) -> list[str]:
    print('\n── FreeSurfer 轉換中 ──────────────────────────────')
    errors = []
    total  = len(splits)

    for i, (sid, split) in enumerate(splits.items(), 1):
        t1_src   = FS_ROOT / sid / 'mri' / 'T1.mgz'
        aseg_src = FS_ROOT / sid / 'mri' / 'aseg.mgz'

        if not t1_src.exists() or not aseg_src.exists():
            errors.append(f'FreeSurfer MISSING: {sid}')
            print(f'[{i:3d}/{total}] SKIP  {sid}')
            continue

        img_dir, lbl_dir = out_dirs('freesurfer', split)
        try:
            mgz_to_nii(t1_src,   img_dir / f'{sid}_0000.nii.gz', np.float32)
            mgz_to_nii(aseg_src, lbl_dir / f'{sid}.nii.gz',       np.int16)
            print(f'[{i:3d}/{total}] OK  {sid}  ({split})')
        except Exception as e:
            errors.append(f'FreeSurfer {sid}: {e}')
            print(f'[{i:3d}/{total}] ERR {sid}: {e}')

    return errors


def convert_fsl(splits: dict[str, str]) -> list[str]:
    print('\n── FSL 轉換中 ─────────────────────────────────────')
    errors = []
    total  = len(splits)

    for i, (sid, split) in enumerate(splits.items(), 1):
        t1_src  = FS_ROOT / sid / 'mri' / 'T1.mgz'   # 共用同一張 T1
        fseg_hdr = find_fseg(sid)

        if not t1_src.exists():
            errors.append(f'FSL T1 MISSING: {sid}')
            print(f'[{i:3d}/{total}] SKIP  {sid} — T1 not found')
            continue
        if fseg_hdr is None:
            errors.append(f'FSL SEG MISSING: {sid}')
            print(f'[{i:3d}/{total}] SKIP  {sid} — fseg not found')
            continue

        img_dir, lbl_dir = out_dirs('fsl', split)
        try:
            mgz_to_nii(t1_src,    img_dir / f'{sid}_0000.nii.gz', np.float32)
            analyze_to_nii(fseg_hdr, lbl_dir / f'{sid}.nii.gz',   np.int16)
            print(f'[{i:3d}/{total}] OK  {sid}  ({split})')
        except Exception as e:
            errors.append(f'FSL {sid}: {e}')
            print(f'[{i:3d}/{total}] ERR {sid}: {e}')

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-freesurfer', action='store_true')
    parser.add_argument('--skip-fsl',        action='store_true')
    parser.add_argument('--no-delete',       action='store_true',
                        help='轉換完成後不刪除 brain_data/')
    args = parser.parse_args()

    # 讀取 splits（JSON 格式：{"splits": {"train": [...], "val": [...], "test": [...]}}）
    splits: dict[str, str] = {}
    with open(SPLITS_CSV) as f:
        data = json.load(f)
    for split_name, ids in data["splits"].items():
        for sid in ids:
            splits[sid] = split_name
    n_tr = sum(1 for s in splits.values() if s == "train")
    n_va = sum(1 for s in splits.values() if s == "val")
    n_te = sum(1 for s in splits.values() if s == "test")
    print(f'共 {len(splits)} 筆 ({n_tr} train / {n_va} val / {n_te} test)')

    # 建立輸出目錄
    for modality in (['freesurfer'] if not args.skip_freesurfer else []) + \
                    (['fsl']        if not args.skip_fsl        else []):
        for d in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
            (OUT_ROOT / modality / d).mkdir(parents=True, exist_ok=True)

    all_errors = []

    if not args.skip_freesurfer:
        all_errors += convert_freesurfer(splits)

    if not args.skip_fsl:
        all_errors += convert_fsl(splits)

    # 摘要
    print('\n' + '=' * 55)
    for modality in ['freesurfer', 'fsl']:
        d = OUT_ROOT / modality
        if not d.exists():
            continue
        print(f'\n[{modality}]')
        for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
            n = len(list((d / sub).glob('*.nii.gz')))
            print(f'  {sub:12s} {n:4d} 個檔案')

    if all_errors:
        print(f'\n錯誤 {len(all_errors)} 筆：')
        for e in all_errors:
            print(' ', e)
        print('\n→ 有錯誤，保留 brain_data/，請確認後手動刪除。')
        sys.exit(1)

    if not args.no_delete:
        print('\n刪除原始 brain_data/ ...')
        shutil.rmtree(BASE / 'brain_data')
        print('刪除完成。')

    print(f'\n✓ 完成！資料集位置：{OUT_ROOT}')


if __name__ == '__main__':
    main()
