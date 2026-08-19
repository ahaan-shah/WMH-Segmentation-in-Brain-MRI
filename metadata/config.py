"""Single source of project paths and parameters.

Every other script imports from here rather than hardcoding paths, the seed,
label values, or thresholds (CLAUDE.md Section 5.5). The values themselves —
and the rationale behind each — live in dataset.yaml (this directory); this
module just resolves paths and exposes both as plain Python constants/objects.

Layout: metadata/ and checks/ are tracked, top-level directories, siblings of
code/ and data/ — the parts of Week 1 the project actually depends on going
forward (subject index, splits, the phantom test suite, the vendored
scorer; checks/validate.py and checks/outputs/ are the exception, gitignored
as Week-1-specific verification). `prototypes/`, `refs/`, and `reports/`
stay organised together under the gitignored `prototyping-and-refs-week1/`
scratch folder, one level deeper — not required by the instructor yet, and
none of them are imported by anything in metadata/ or checks/. prototypes/
DOES import from metadata/ (it reuses the same subject-discovery and config
code), so each prototypes/*.py adds the true project root to sys.path itself
before that import — see the top of any prototypes/*.py file for that
handful of lines. This is the one place a path lives outside this file,
because metadata/config.py can't be imported to compute a path used to find
metadata/config.py in the first place.
"""

from pathlib import Path

import yaml

# metadata/config.py -> parents[0] is metadata/, parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "metadata" / "dataset.yaml"

DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTERIM = DATA_ROOT / "interim"
DATA_PROCESSED = DATA_ROOT / "processed"

CHECKS_OUTPUTS = PROJECT_ROOT / "checks" / "outputs"

PROTOTYPES_OUTPUTS = PROJECT_ROOT / "prototyping-and-refs-week1" / "prototypes" / "outputs"
PROTOTYPES_FIGURES = PROTOTYPES_OUTPUTS / "figures"

METADATA_OUTPUTS = PROJECT_ROOT / "metadata" / "outputs"

with open(CONFIG_PATH) as _f:
    DATASET_CONFIG = yaml.safe_load(_f)

SEED = DATASET_CONFIG["seed"]

LABEL_BACKGROUND = DATASET_CONFIG["labels"]["background"]
LABEL_WMH = DATASET_CONFIG["labels"]["wmh"]
LABEL_OTHER = DATASET_CONFIG["labels"]["other"]

PERIVENTRICULAR_THRESHOLD_MM = DATASET_CONFIG["periventricular"]["threshold_mm"]
PERIVENTRICULAR_SENSITIVITY_THRESHOLDS_MM = DATASET_CONFIG["periventricular"][
    "sensitivity_thresholds_mm"
]

# --- Week 2 pre-processing (see the `preprocessing:` block in dataset.yaml,
# which records the measured evidence behind each of these) ---
PREPROCESSING = DATASET_CONFIG["preprocessing"]

# Tolerance for assert_same_geometry(), as a world-space corner displacement in
# mm. Exact affine equality is unusable here: 6 of 170 subjects differ from
# their own FLAIR by up to 6.3e-4 mm of float32 storage rounding.
GEOMETRY_TOLERANCE_MM = PREPROCESSING["geometry_tolerance_mm"]

# Arrays are computed in canonical RAS; files are written in the raw FLAIR's
# native orientation so the vendored official scorer can read them directly.
WORKING_ORIENTATION = PREPROCESSING["working_orientation"]
STORAGE_ORIENTATION = PREPROCESSING["storage_orientation"]

N4_CONFIG = PREPROCESSING["n4"]
SKULL_STRIP_CONFIG = PREPROCESSING["skull_strip"]
TISSUE_SEGMENTATION_CONFIG = PREPROCESSING["tissue_segmentation"]
NORMALISATION_CONFIG = PREPROCESSING["normalisation"]
DENOISING_CONFIG = PREPROCESSING["denoising"]
CONTRAST_ENHANCEMENT_CONFIG = PREPROCESSING["contrast_enhancement"]
MORPHOLOGY_CONFIG = PREPROCESSING["morphology"]

TRAIN_VAL_SUBJECTS = DATASET_CONFIG["splits"]["train_val_subjects"]
TRAIN_FRACTION = DATASET_CONFIG["splits"]["train_fraction"]
VAL_FRACTION = DATASET_CONFIG["splits"]["val_fraction"]
SPLIT_STRATIFY_BY = DATASET_CONFIG["splits"]["stratify_by"]
SPLIT_SITES = DATASET_CONFIG["splits"]["sites"]

# Best-effort scanner metadata keyed by (site, scanner_subdir_or_None).
# scanner_subdir is None for sites with no intermediate scanner-name directory
# (Utrecht, Singapore) and the literal subdirectory name where one exists
# (Amsterdam: GE3T, GE1T5, "Philips_VU .PETMR_01."). The three training-site
# entries are stated in CLAUDE.md Section 3 / the Kuijf et al. 2019 challenge
# paper. The two test-only Amsterdam scanners are labelled from the directory
# name alone and are NOT cross-checked against the paper — flagged as such
# per CLAUDE.md Section 4 ("flag, don't guess") rather than asserted with
# false confidence.
SCANNER_METADATA = {
    ("Utrecht", None): {
        "scanner": "Philips Achieva",
        "field_strength_t": 3.0,
        "vendor": "Philips",
        "source": "Kuijf et al. 2019 / CLAUDE.md Section 3",
    },
    ("Singapore", None): {
        "scanner": "Siemens TrioTim",
        "field_strength_t": 3.0,
        "vendor": "Siemens",
        "source": "Kuijf et al. 2019 / CLAUDE.md Section 3",
    },
    ("Amsterdam", "GE3T"): {
        "scanner": "GE Signa HDxt",
        "field_strength_t": 3.0,
        "vendor": "GE",
        "source": "Kuijf et al. 2019 / CLAUDE.md Section 3",
    },
    ("Amsterdam", "GE1T5"): {
        "scanner": "GE Signa HDxt (1.5T variant, test-only)",
        "field_strength_t": 1.5,
        "vendor": "GE",
        "source": "directory name only — UNVERIFIED against published paper",
    },
    ("Amsterdam", "Philips_VU .PETMR_01."): {
        "scanner": "Philips (VU medical center PET/MR unit, test-only)",
        "field_strength_t": None,
        "vendor": "Philips",
        "source": "directory name only — UNVERIFIED against published paper",
    },
}
