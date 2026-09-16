"""Master subject index (CLAUDE.md W1.3) + orientation audit (W1.4).

Writes metadata/outputs/subject_index.csv, one row per subject, with the
columns CLAUDE.md Section 8 (W1.3) specifies: site/scanner metadata, voxel
spacing and volume (mm^3 lesion volumes are impossible without this —
Decision #6), orientation code (heterogeneity here is a live L/R-flip risk —
Section 11.3), FLAIR intensity percentiles (makes W2 normalisation
data-driven), and lesion voxel/volume/connected-component counts.

Stats are computed on the RAW shipped orientation (no as_closest_canonical),
because the orientation audit needs to see what was actually shipped per
subject, per scanner, before this codebase transforms anything. Connected
components and voxel counts are orientation-invariant so this doesn't bias
those columns.

Usage: python -m metadata.build_index
Output: metadata/outputs/subject_index.csv (+ .json provenance sidecar)
"""

import logging
from datetime import date

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

from metadata.config import LABEL_OTHER, LABEL_WMH, METADATA_OUTPUTS
from metadata.loader import discover_subjects, get_scanner_metadata
from metadata.provenance import write_manifest

CONNECTIVITY_26 = np.ones((3, 3, 3))


def _setup_logging() -> logging.Logger:
    METADATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    log_path = METADATA_OUTPUTS / f"build_index_{date.today().isoformat()}.log"

    logger = logging.getLogger("build_index")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def build_subject_row(subject) -> dict:
    flair_img = nib.load(subject.flair_path)
    t1_img = nib.load(subject.t1_path)
    mask_img = nib.load(subject.mask_path)

    flair_data = np.asarray(flair_img.dataobj)
    mask_data = np.asarray(mask_img.dataobj)

    spacing = tuple(float(z) for z in flair_img.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    orientation_code = "".join(nib.aff2axcodes(flair_img.affine))

    foreground = flair_data[flair_data > 0]
    if foreground.size > 0:
        p1, p50, p99 = np.percentile(foreground, [1, 50, 99])
    else:
        p1 = p50 = p99 = float("nan")

    wmh_mask = mask_data == LABEL_WMH
    n_lesion_voxels = int(wmh_mask.sum())
    lesion_volume_mm3 = n_lesion_voxels * voxel_volume_mm3

    _, n_lesions_26conn = label(wmh_mask, structure=CONNECTIVITY_26)

    n_label2_voxels = int(np.sum(mask_data == LABEL_OTHER))

    scanner_meta = get_scanner_metadata(subject.site, subject.scanner_dir)

    return {
        "subject_key": subject.subject_key,
        "subject_id": subject.subject_id,
        "split": subject.split,
        "site": subject.site,
        "scanner": scanner_meta["scanner"],
        "field_strength_t": scanner_meta["field_strength_t"],
        "vendor": scanner_meta["vendor"],
        "scanner_metadata_source": scanner_meta["source"],
        "flair_shape": "x".join(str(s) for s in flair_img.shape),
        "t1_shape": "x".join(str(s) for s in t1_img.shape),
        "voxel_spacing_x_mm": spacing[0],
        "voxel_spacing_y_mm": spacing[1],
        "voxel_spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_volume_mm3,
        "orientation_code": orientation_code,
        "flair_p1": float(p1),
        "flair_p50": float(p50),
        "flair_p99": float(p99),
        "n_lesion_voxels": n_lesion_voxels,
        "lesion_volume_mm3": lesion_volume_mm3,
        "n_lesions_26conn": int(n_lesions_26conn),
        "has_label_2": n_label2_voxels > 0,
        "n_label2_voxels": n_label2_voxels,
        "has_flair_mask_extra": subject.has_flair_mask_extra,
        "has_t1_mask_extra": subject.has_t1_mask_extra,
        "path_flair": str(subject.flair_path),
        "path_t1": str(subject.t1_path),
        "path_flair_pre": str(subject.flair_pre_path),
        "path_t1_pre": str(subject.t1_pre_path),
        "path_mask": str(subject.mask_path),
        "path_reg_t1_to_flair": str(subject.reg_t1_to_flair_path) if subject.reg_t1_to_flair_path else "",
    }


def build_index() -> pd.DataFrame:
    logger = _setup_logging()
    subjects = discover_subjects()
    logger.info(f"Building subject index for {len(subjects)} subjects.")

    rows = []
    for s in subjects:
        row = build_subject_row(s)
        rows.append(row)
        logger.info(
            f"{s.subject_key}: shape={row['flair_shape']} spacing="
            f"({row['voxel_spacing_x_mm']:.3f},{row['voxel_spacing_y_mm']:.3f},"
            f"{row['voxel_spacing_z_mm']:.3f}) orientation={row['orientation_code']} "
            f"lesion_vol={row['lesion_volume_mm3']:.1f}mm3 n_lesions={row['n_lesions_26conn']}"
        )

    df = pd.DataFrame(rows)

    orientation_counts = df["orientation_code"].value_counts()
    logger.info(f"Orientation code distribution:\n{orientation_counts.to_string()}")
    if len(orientation_counts) > 1:
        logger.warning(
            "Orientation is NOT homogeneous across subjects — "
            "downstream code must canonicalise (as_closest_canonical) rather than assume RAS/any fixed layout."
        )

    out_path = METADATA_OUTPUTS / "subject_index.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    write_manifest(
        out_path,
        generating_script="code/metadata/build_index.py",
        extra={"n_subjects": len(df), "orientation_codes": orientation_counts.to_dict()},
    )
    logger.info(f"Wrote {out_path}")

    return df


if __name__ == "__main__":
    build_index()
