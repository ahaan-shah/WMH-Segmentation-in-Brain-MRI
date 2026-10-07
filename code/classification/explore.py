"""Weeks 5-6 — exploratory analysis of the severity problem, before any model.

Run AFTER the decisions in `dataset.yaml: severity_classification` were
recorded, on purpose: the cut-offs, the class scheme and the six features were
fixed first, so nothing seen here can quietly steer them. What this script is
for is knowing the problem before modelling it:

1. **How big is each class, overall and per hospital?** Severity that is
   unevenly spread across sites lets a model learn "which scanner" instead of
   "how sick".
2. **How many patients sit within measurement error of a cut-off?** Those are
   the ones no classifier can be expected to place reliably — the honest
   ceiling on accuracy.
3. **Does the PATTERN of disease carry information beyond its AMOUNT?** Each
   pattern feature is tested against the class both raw and with total volume
   partialled out. Only the second number says whether a feature can add
   anything a volume threshold does not already have.
4. **Are the six features redundant with each other?** The 14 -> 6 reduction was
   by definition; this measures whether it worked.
5. **Does any feature encode the hospital?** "<=5 voxels" is a different
   physical size at each site, so `small_lesion_fraction` is the prime suspect.
6. **Why the production model's predictions cannot be the features.** In-sample
   vs held-out measurement error, side by side.

Features here are mostly REFERENCE-derived — they describe the disease itself.
Prediction-derived ones appear only in (6), and only to show why the classifier
will instead use leave-one-site-out predictions. Nothing in `eda_table.csv` is a
classifier input.

    code/.venv/bin/python -m classification.explore
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from classification.severity import FEATURE_NAMES, assign_class, derive_features
from metadata.config import (CODE_ROOT, METADATA_OUTPUTS, PROJECT_ROOT, SEED,
                             SEVERITY_CONFIG)
from metadata.loader import load_split
from metadata.provenance import get_git_commit_hash, write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "explore"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
FEATURES_CSV = CODE_ROOT / "features" / "outputs" / "features.csv"
GALLERY = PROJECT_ROOT / "outputs" / "week5-classification" / "01-exploring-the-data"

N_PERMUTATIONS = 10_000
PATTERN_FEATURES = [f for f in FEATURE_NAMES if f != "log_total_volume"]
PRETTY = {
    "log_total_volume": "log total volume",
    "periventricular_fraction": "periventricular fraction",
    "abs_laterality_index": "|laterality index|",
    "log_lesion_count": "log lesion count",
    "small_lesion_fraction": "small-lesion fraction",
    "largest_lesion_share": "largest-lesion share",
}

# --- palette: validated with the dataviz skill's validate_palette.js ---------
# Classes are ORDERED, so they take a one-hue ordinal ramp (blue 250/400/550/700;
# passes --ordinal: monotone L, every step gap >= 0.06, light end 2.06:1).
CLASS_COLOURS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Sites are NOMINAL: the first three categorical slots, which pass all-pairs.
# Aqua is under 3:1 on white, so every site also gets its own marker shape and
# a legend — identity never rests on colour alone.
SITE_STYLE = {"Amsterdam": ("#2a78d6", "o"), "Singapore": ("#eb6834", "s"),
              "Utrecht": ("#1baf7a", "^")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#1c5cab", "#f0efec", "#c23a39"])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cramers_v(a, b) -> float:
    table = pd.crosstab(a, b).to_numpy()
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def permutation_p(statistic, x, y, rng) -> float:
    """Two-sided permutation p-value for |statistic(x, y)|.

    Used instead of the asymptotic chi-square / Spearman p-values because with
    8 subjects in a class several expected cell counts fall under 5, where those
    approximations are not trustworthy.
    """
    observed = abs(statistic(x, y))
    y = np.asarray(y)
    hits = sum(abs(statistic(x, rng.permutation(y))) >= observed
               for _ in range(N_PERMUTATIONS))
    return (hits + 1) / (N_PERMUTATIONS + 1)


def spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).statistic)


def partial_spearman(x, y, control) -> float:
    """Spearman of x and y with `control` partialled out.

    Ranks everything, regresses x-ranks and y-ranks on control-ranks, and
    correlates the residuals. Answers "does x track y among subjects with the
    same total volume?" — the only question that matters for whether a pattern
    feature can add to a volume threshold.
    """
    rx, ry, rc = (stats.rankdata(v) for v in (x, y, control))
    design = np.column_stack([np.ones_like(rc), rc])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def residualise(values, control) -> np.ndarray:
    design = np.column_stack([np.ones_like(control), control])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------
def _stamp(fig, title, subtitle, commit):
    """Title block positioned in inches, not figure fractions (Week 4 lesson:
    a fixed fraction overprinted the caption on tall figures)."""
    height = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=INK,
                 y=1.0 - 0.22 / height)
    fig.text(0.5, 1.0 - 0.52 / height, subtitle, ha="center", va="top",
             fontsize=9.2, style="italic", color=INK_2, wrap=True)
    fig.text(0.995, 0.004, f"commit {commit[:8]}", ha="right", fontsize=6.5,
             color="grey")


def _reserve(fig, inches=1.15):
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=[0, 0, 1, 1.0 - inches / height])


def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name, commit, logger):
    path = GALLERY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_manifest(path, generating_script=GENERATING, extra={"commit": commit})
    logger.info("  figure: %s", path.relative_to(PROJECT_ROOT))


def _jitter(n, rng, width=0.18):
    return rng.uniform(-width, width, n)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_volumes_and_cutoffs(table, cutoffs, classes, error_band, commit, logger, rng):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _axes_style(ax)
    sites = list(SITE_STYLE)
    lo, hi = 0.5, 100.0

    # Shade the zone around each cut-off that one median held-out measurement
    # error spans: a subject inside it can be flipped by an ordinary error.
    for c in cutoffs:
        ax.axvspan(c / (1 + error_band), c * (1 + error_band), color="#f0efec", zorder=1)
        ax.axvline(c, color=INK_2, linewidth=1.0, linestyle="--", zorder=2)
        ax.text(c, -0.55, f"{c:g} mL", ha="center", va="bottom", fontsize=8.5,
                color=INK_2, zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    edges = [lo, *cutoffs, hi]
    counts = np.bincount(table["class4"], minlength=len(classes))
    for i, name in enumerate(classes):
        centre = np.sqrt(edges[i] * edges[i + 1])
        ax.text(centre, len(sites) - 0.05, f"{name}\nn = {counts[i]}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold", color=INK)

    for row, site in enumerate(sites):
        colour, marker = SITE_STYLE[site]
        part = table[table.site == site]
        ax.scatter(part["reference_volume_ml"], row + _jitter(len(part), rng),
                   s=46, color=colour, marker=marker, edgecolor="white",
                   linewidth=1.2, zorder=3, label=site)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=9.5, color=INK)
    ax.set_ylim(-0.6, len(sites) + 0.75)
    ax.set_xlabel("expert-measured WMH volume (mL, log scale)", color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=SITE_STYLE[s][0], marker=SITE_STYLE[s][1],
                              linestyle="", markersize=7, label=s) for s in sites]
              + [Patch(color="#f0efec", label=f"within ±{error_band:.0%} of a cut-off")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
              fontsize=8.5)
    _stamp(fig, "The target: 60 patients, four Fazekas-anchored classes",
           "Each dot is one patient's expert lesion volume. Dashed lines are the Joo et al. "
           "(2022) cut-offs; the grey band around each is one typical held-out measurement "
           "error wide — patients inside it can be flipped by an ordinary error.", commit)
    _reserve(fig, 0.95)
    _save(fig, "the_target_and_the_cutoffs.png", commit, logger)


def figure_classes_by_site(table, classes, commit, logger):
    sites = list(SITE_STYLE)
    rows = ["All 60", *sites]
    counts = [np.bincount(table["class4"], minlength=len(classes))]
    counts += [np.bincount(table.loc[table.site == s, "class4"], minlength=len(classes))
               for s in sites]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    _axes_style(ax)
    ax.grid(False)
    for y, row in enumerate(counts):
        share = row / row.sum()
        left = 0.0
        for k, (n, frac) in enumerate(zip(row, share)):
            if n == 0:
                continue
            # 2px surface gap between segments, per the mark spec.
            ax.barh(y, frac, left=left, height=0.62, color=CLASS_COLOURS[k],
                    edgecolor="white", linewidth=2)
            ax.text(left + frac / 2, y, str(n), ha="center", va="center", fontsize=9.5,
                    color=INK if k == 0 else "white", fontweight="bold")
            left += frac
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of that group's patients (numbers in the bars are patients)",
                  color=INK_2)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLOURS, classes)],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False,
              fontsize=9)
    _stamp(fig, "Severity is not spread evenly across the three hospitals",
           "If one hospital supplies most of the severe cases, a model can score well by "
           "recognising the scanner instead of the disease. Results will be reported per "
           "hospital for this reason.", commit)
    _reserve(fig, 0.95)
    _save(fig, "classes_by_hospital.png", commit, logger)


def figure_pattern_vs_class(table, tests, classes, commit, logger, rng):
    fig, axes = plt.subplots(1, len(PATTERN_FEATURES), figsize=(15, 5.0), sharex=True)
    for ax, feature in zip(axes, PATTERN_FEATURES):
        _axes_style(ax)
        ax.grid(axis="x", visible=False)
        for k in range(len(classes)):
            values = table.loc[table["class4"] == k, f"ref_{feature}"].to_numpy()
            ax.scatter(k + _jitter(len(values), rng), values, s=22, color="#8a8984",
                       edgecolor="white", linewidth=0.8, zorder=3)
            if len(values):
                ax.hlines(np.median(values), k - 0.3, k + 0.3, color=INK, linewidth=2.2,
                          zorder=4)
        row = tests.loc[feature]
        ax.set_title(f"{PRETTY[feature]}\n"
                     f"ρ with class  {row.rho_class:+.2f}\n"
                     f"same-volume ρ  {row.partial_rho_class:+.2f}"
                     f"{'*' if row.partial_p < 0.05 else ''}",
                     fontsize=9, color=INK, linespacing=1.35)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=8.2, rotation=25)
    _stamp(fig, "Does the PATTERN of disease say anything that its AMOUNT does not?",
           "Each dot is one patient (expert masks); black bars are class medians. "
           "'Same-volume ρ' compares patients of the same total burden — only that "
           "number shows whether a feature can beat a volume threshold.  "
           "* permutation p < 0.05, uncorrected for five tests.", commit)
    _reserve(fig, 0.95)
    _save(fig, "pattern_features_by_class.png", commit, logger)


def figure_redundancy(ref_corr, commit, logger):
    names = list(FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    image = ax.imshow(ref_corr.loc[names, names].to_numpy(), cmap=DIVERGING,
                      vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            value = ref_corr.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(value) > 0.6 else INK)
    labels = [PRETTY[n] for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Spearman ρ  (grey = unrelated)", color=INK_2, fontsize=8.5)
    bar.ax.tick_params(labelsize=8, colors=INK_2)
    _stamp(fig, "Did cutting 14 features to 6 remove the redundancy?",
           "Spearman correlation between the six features, expert masks, all 60. Before the "
           "cut, seven features sat at 0.85–0.98 with total volume.", commit)
    _reserve(fig, 1.15)
    _save(fig, "feature_redundancy.png", commit, logger)


def figure_why_honest_predictions(table, cutoffs, commit, logger):
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    _axes_style(ax)
    lo, hi = 0.5, 100.0
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, zorder=2)
    for c in cutoffs:
        ax.axvline(c, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
        ax.axhline(c, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    groups = [("training patients (n = 48) — the model learned these", False,
               "#2a78d6", "o"),
              ("validation patients (n = 12) — never trained on", True, "#eb6834", "^")]
    for label, held_out, colour, marker in groups:
        part = table[table.held_out == held_out]
        ax.scatter(part["reference_volume_ml"], part["insample_pred_volume_ml"], s=54,
                   color=colour, marker=marker, edgecolor="white", linewidth=1.3,
                   zorder=3 + held_out, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("expert volume (mL)", color=INK_2)
    ax.set_ylabel("our production model's volume (mL)", color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False,
              fontsize=8.5)
    _stamp(fig, "Why the classifier cannot use the production model's measurements",
           "On the identity line = perfect. The dashed grid is the class cut-offs. Patients "
           "the network trained on hug the line; unseen patients do not. Features will come "
           "from the leave-one-hospital-out networks instead.", commit)
    _reserve(fig, 1.15)
    _save(fig, "why_honest_predictions.png", commit, logger)


# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    rng = np.random.default_rng(SEED)
    commit = get_git_commit_hash()

    cutoffs = SEVERITY_CONFIG["cutoffs_ml"]
    classes = SEVERITY_CONFIG["classes"]
    sens_cutoffs = SEVERITY_CONFIG["sensitivity_cutoffs_ml"]
    sens_classes = SEVERITY_CONFIG["sensitivity_classes"]

    raw = pd.read_csv(FEATURES_CSV)
    reference = raw[raw.source == "reference"].set_index("subject_key")
    prediction = raw[raw.source == "prediction"].set_index("subject_key")
    if len(reference) != 60 or set(reference.index) != set(prediction.index):
        raise SystemExit(f"expected 60 subjects with both sources, got {len(reference)} "
                         f"reference / {len(prediction)} prediction")
    val = set(load_split("val"))
    if not val <= set(reference.index):
        raise SystemExit("validation split contains subjects missing from features.csv")

    # --- the analysis table --------------------------------------------------
    table = pd.DataFrame(index=reference.index)
    table["site"] = reference["site"]
    table["held_out"] = table.index.isin(val)
    table["reference_volume_ml"] = reference["total_lesion_volume_ml"]
    table["insample_pred_volume_ml"] = prediction.loc[table.index, "total_lesion_volume_ml"]
    table["class4"] = assign_class(table["reference_volume_ml"], cutoffs)
    table["class3"] = assign_class(table["reference_volume_ml"], sens_cutoffs)
    ref_feats = derive_features(reference).add_prefix("ref_")
    table = table.join(ref_feats)

    table.to_csv(OUTPUTS_DIR / "eda_table.csv")
    write_manifest(OUTPUTS_DIR / "eda_table.csv", generating_script=GENERATING,
                   extra={"note": "EDA only. NOT a classifier input — features here are "
                                  "reference-derived, and the prediction volume is "
                                  "in-sample for 48 of 60 subjects."})
    summary = {}

    # --- 1. class sizes, overall and per site --------------------------------
    counts4 = pd.crosstab(table.site, table.class4.map(dict(enumerate(classes))))
    counts4 = counts4.reindex(columns=classes, fill_value=0)
    counts3 = pd.crosstab(table.site, table.class3.map(dict(enumerate(sens_classes))))
    counts3 = counts3.reindex(columns=sens_classes, fill_value=0)
    v4 = cramers_v(table.site, table.class4)
    p4 = permutation_p(cramers_v, table.site.to_numpy(), table.class4.to_numpy(), rng)
    logger.info("1. CLASS SIZES (4-class): %s",
                dict(zip(classes, np.bincount(table.class4, minlength=4).tolist())))
    logger.info("   by site:\n%s", counts4.to_string())
    logger.info("   3-class sensitivity scheme: %s",
                dict(zip(sens_classes, np.bincount(table.class3, minlength=3).tolist())))
    logger.info("   site x class association: Cramer's V %.3f, permutation p %.4f", v4, p4)
    summary["class_counts_4"] = dict(zip(classes,
                                         np.bincount(table.class4, minlength=4).tolist()))
    summary["class_counts_3"] = dict(zip(sens_classes,
                                         np.bincount(table.class3, minlength=3).tolist()))
    summary["class_counts_4_by_site"] = counts4.to_dict(orient="index")
    summary["site_class_cramers_v"] = v4
    summary["site_class_permutation_p"] = p4

    # --- 2. how many sit within measurement error of a cut-off ---------------
    # Error measured ONLY on the 12 held-out subjects: the other 48 were
    # trained on and their error is not an honest estimate.
    held = table[table.held_out]
    ratio = held["insample_pred_volume_ml"] / held["reference_volume_ml"]
    abs_error = np.abs(ratio - 1)
    band_median = float(np.median(abs_error))
    band_p90 = float(np.quantile(abs_error, 0.9))
    volumes = table["reference_volume_ml"].to_numpy()

    def near_cutoff(band):
        return np.array([any(c / (1 + band) <= v <= c * (1 + band) for c in cutoffs)
                         for v in volumes])

    near_med, near_p90 = near_cutoff(band_median), near_cutoff(band_p90)
    logger.info("2. MEASUREMENT ERROR (held-out 12): median |pred/ref - 1| = %.1f%%, "
                "90th percentile %.1f%%", 100 * band_median, 100 * band_p90)
    logger.info("   patients within the MEDIAN error of a cut-off: %d / 60 (%s)",
                near_med.sum(), {k: int(v) for k, v in table[near_med].site.value_counts().items()})
    logger.info("   patients within the 90th-percentile error: %d / 60", near_p90.sum())
    logger.info("   => a classifier that measures volume as well as our segmentation does "
                "should be expected to misplace roughly that many, whatever the model.")
    summary.update({"heldout_median_abs_volume_error": band_median,
                    "heldout_p90_abs_volume_error": band_p90,
                    "n_within_median_error_of_cutoff": int(near_med.sum()),
                    "n_within_p90_error_of_cutoff": int(near_p90.sum())})

    # --- 3. pattern vs amount ------------------------------------------------
    rows = []
    log_volume = table["ref_log_total_volume"].to_numpy()
    for feature in FEATURE_NAMES:
        x = table[f"ref_{feature}"].to_numpy()
        y = table["class4"].to_numpy()
        row = {"feature": feature,
               "rho_class": spearman(x, y),
               "rho_volume": spearman(x, log_volume)}
        if feature == "log_total_volume":
            row.update(partial_rho_class=np.nan, partial_p=np.nan)
        else:
            row["partial_rho_class"] = partial_spearman(x, y, log_volume)
            row["partial_p"] = permutation_p(
                lambda a, b: partial_spearman(a, b, log_volume), x, y, rng)
        # Hospital effect, raw and after removing burden — a feature can differ
        # by site simply because burden does.
        groups = [x[table.site.to_numpy() == s] for s in SITE_STYLE]
        row["site_kruskal_p"] = float(stats.kruskal(*groups).pvalue)
        resid = residualise(stats.rankdata(x), stats.rankdata(log_volume))
        groups_r = [resid[table.site.to_numpy() == s] for s in SITE_STYLE]
        row["site_kruskal_p_volume_removed"] = float(stats.kruskal(*groups_r).pvalue)
        row.update({f"median_{s}": float(np.median(g)) for s, g in zip(SITE_STYLE, groups)})
        rows.append(row)
    tests = pd.DataFrame(rows).set_index("feature")
    tests.to_csv(OUTPUTS_DIR / "eda_feature_tests.csv")
    write_manifest(OUTPUTS_DIR / "eda_feature_tests.csv", generating_script=GENERATING,
                   extra={"n_permutations": N_PERMUTATIONS, "seed": SEED})
    pd.set_option("display.width", 250)
    logger.info("3/5. FEATURE TESTS (expert masks, 60 subjects):\n%s",
                tests.round(3).to_string())

    # --- 4. redundancy among the six -----------------------------------------
    ref_corr = table[[f"ref_{f}" for f in FEATURE_NAMES]].corr(method="spearman")
    ref_corr.index = ref_corr.columns = list(FEATURE_NAMES)
    off_diag = ref_corr.where(~np.eye(len(FEATURE_NAMES), dtype=bool))
    worst = off_diag.abs().stack().idxmax()
    logger.info("4. REDUNDANCY: largest |rho| between two of the six = %.2f (%s ~ %s)",
                off_diag.abs().max().max(), *worst)
    summary["max_abs_rho_between_features"] = float(off_diag.abs().max().max())
    summary["max_abs_rho_pair"] = list(worst)

    # --- 6. in-sample vs held-out --------------------------------------------
    pred_class = assign_class(table["insample_pred_volume_ml"], cutoffs)
    for held_out, name in ((False, "training 48 (in-sample)"), (True, "validation 12")):
        mask = table.held_out.to_numpy() == held_out
        err = np.abs(table.loc[mask, "insample_pred_volume_ml"]
                     / table.loc[mask, "reference_volume_ml"] - 1)
        acc = float(np.mean(pred_class[mask] == table.class4.to_numpy()[mask]))
        logger.info("6. %-24s median volume error %.1f%%, threshold-rule 4-class "
                    "accuracy %.0f%%", name, 100 * np.median(err), 100 * acc)
        summary[f"threshold_rule_acc_4class_{'heldout' if held_out else 'insample'}"] = acc

    with open(OUTPUTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_manifest(OUTPUTS_DIR / "eda_summary.json", generating_script=GENERATING)

    logger.info("figures:")
    figure_volumes_and_cutoffs(table, cutoffs, classes, band_median, commit, logger, rng)
    figure_classes_by_site(table, classes, commit, logger)
    figure_pattern_vs_class(table, tests, classes, commit, logger, rng)
    figure_redundancy(ref_corr, commit, logger)
    figure_why_honest_predictions(table, cutoffs, commit, logger)


if __name__ == "__main__":
    main()
