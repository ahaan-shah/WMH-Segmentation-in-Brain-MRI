"""Weeks 5-6 gallery — the classification figures (R10).

Continues `outputs/week5-classification/` after the EDA's `01-exploring-the-data`:

    02-volume-only/          can a model that LEARNS beat the clinical cut-offs, on volume alone?
    03-adding-the-pattern/   does WHERE and HOW the disease sits add anything to HOW MUCH?
    04-the-choice/           the pre-registered selection rule, step by step, with its numbers
    05-what-drives-it/       which features the decisions actually rest on
    06-three-patients/       the gallery's three patients: the class given beside the true one

Same rules as every earlier gallery: every figure carries its own point of
comparison (the threshold rule and the majority-class floor are drawn on every
score chart), it is generated here and never hand-edited, and it is stamped with
the commit that made it. The palette is the EDA's, validated with the dataviz
skill's checker: an ordinal one-hue ramp for the ordered classes, and three
categorical hues for the hospitals, each also carrying its own marker shape.

Drawing functions take plain tables, not paths, so each can be rendered on
synthetic numbers and looked at before the real results exist
(`checks/test_classification.py` does exactly that).

    code/.venv/bin/python -m classification.make_outputs
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from classification.explore import (CLASS_COLOURS, GRID, INK, INK_2, PRETTY, SITE_STYLE,
                                    _axes_style, _reserve, _stamp)
from metadata.config import CODE_ROOT, PROJECT_ROOT, SEVERITY_CONFIG
from metadata.provenance import get_git_commit_hash, write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "make_outputs"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
GALLERY = PROJECT_ROOT / "outputs" / "week5-classification"

# The same three patients every gallery shows: the median-burden TRAINING
# patient per hospital, the rule preprocessing/make_outputs.pick_subjects uses.
GALLERY_PATIENTS = {"Amsterdam": "training_Amsterdam_GE3T_112",
                    "Singapore": "training_Singapore_Singapore_64",
                    "Utrecht": "training_Utrecht_Utrecht_49"}

CANDIDATE_LABEL = {
    "majority_class": "majority class (floor)",
    "threshold_rule": "threshold rule (no learning)",
    "regression_then_threshold": "regression, then cut-offs",
    "ordinal_logistic": "ordinal logistic",
    "multinomial_logistic": "multinomial logistic",
    "random_forest": "random forest",
}
METRICS = [("qwk_mean", "qwk_sd", "quadratic weighted kappa", "higher is better"),
           ("mean_absolute_class_error_mean", "mean_absolute_class_error_sd",
            "mean absolute class error", "lower is better"),
           ("balanced_accuracy_mean", "balanced_accuracy_sd", "balanced accuracy",
            "higher is better")]
LEARNED_C, VOLUME_C, FULL_C = "#3987e5", "#86b6ef", "#0d366b"
BASE_C = "#8a8984"
TRUTH_C, PRED_C = "#39FF14", "#FF3B30"
# Inches kept clear above the axes for the title block. Smaller than the EDA's
# because matplotlib's tight_layout already counts the suptitle; 1.15 left a
# visible empty band under the caption (seen on the synthetic dry run).
TITLE_SPACE = 0.55


def _save(fig, relative, commit, logger, root=None):
    path = (GALLERY if root is None else root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_manifest(path, generating_script=GENERATING, extra={"commit": commit})
    if logger:
        logger.info("  figure: %s", path)
    return path


def _reference_lines(ax, summary, column, horizontal=False):
    """The two points of comparison every score chart carries."""
    rows = summary.set_index("candidate")
    for name, style in (("threshold_rule", "--"), ("majority_class", ":")):
        if name in rows.index:
            value = rows.loc[name, column]
            (ax.axhline if horizontal else ax.axvline)(value, color=INK_2, linestyle=style,
                                                       linewidth=1.1, zorder=1)


# ---------------------------------------------------------------------------
# 02 — volume only
# ---------------------------------------------------------------------------
def figure_scores(summary: pd.DataFrame, title: str, subtitle: str, colour: str, commit: str = "synthetic"):
    """Mean +- SD over repeats for each candidate, one panel per primary score."""
    summary = summary[summary.scheme == "4class"]
    order = [c for c in CANDIDATE_LABEL if c in set(summary.candidate)]
    rows = summary.set_index("candidate").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.55 * len(order) + 2.6), sharey=True)
    for ax, (mean, sd, name, direction) in zip(axes, METRICS):
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        for y, candidate in enumerate(order):
            baseline = candidate in ("majority_class", "threshold_rule")
            ax.errorbar(rows.loc[candidate, mean], y, xerr=rows.loc[candidate, sd],
                        fmt="o", markersize=8, color=BASE_C if baseline else colour,
                        ecolor=GRID if baseline else colour, elinewidth=2, capsize=0,
                        markeredgecolor="white", markeredgewidth=1.4, zorder=3)
            ax.text(rows.loc[candidate, mean], y - 0.3, f"{rows.loc[candidate, mean]:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                              edgecolor="none"))
        _reference_lines(ax, summary, mean)
        ax.set_title(f"{name}\n({direction})", fontsize=9.5, color=INK)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([CANDIDATE_LABEL[c] for c in order], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=INK_2, linestyle="--", label="threshold rule"),
                        Line2D([], [], color=INK_2, linestyle=":", label="majority-class floor"),
                        Line2D([], [], color=INK_2, marker="o", linestyle="-",
                               label="mean ± SD over 10 repeats")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    _stamp(fig, title, subtitle, commit)
    _reserve(fig, TITLE_SPACE)
    return fig


def figure_confusions(confusions: dict, ids: list, classes: list, title: str, subtitle: str, commit: str = "synthetic"):
    """Pooled confusion matrices, side by side. Rows true, columns given."""
    fig, axes = plt.subplots(1, len(ids), figsize=(4.4 * len(ids), 4.9))
    axes = np.atleast_1d(axes)
    for ax, cid in zip(axes, ids):
        matrix = np.asarray(confusions[cid], dtype=float)
        share = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blues", ["#fcfcfb", CLASS_COLOURS[2]]), vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=8.5, rotation=25)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8.5)
        ax.set_xlabel("class given", color=INK_2, fontsize=9)
        ax.set_ylabel("true class", color=INK_2, fontsize=9)
        name = cid.split("/")[-1]
        ax.set_title(CANDIDATE_LABEL.get(name, name), fontsize=10, color=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    _stamp(fig, title, subtitle, commit)
    _reserve(fig, TITLE_SPACE)
    return fig


# ---------------------------------------------------------------------------
# 03 — adding the pattern
# ---------------------------------------------------------------------------
def figure_pattern(volume: pd.DataFrame, full: pd.DataFrame, commit: str = "synthetic"):
    """Each learned candidate: volume only -> volume + pattern, QWK and per hospital."""
    volume, full = (s[(s.scheme == "4class")].set_index("candidate") for s in (volume, full))
    learned = [c for c in CANDIDATE_LABEL if c in full.index and c in volume.index
               and c not in ("majority_class", "threshold_rule")]
    sites = list(SITE_STYLE)
    fig, axes = plt.subplots(1, 1 + len(sites), figsize=(15, 0.6 * len(learned) + 2.8),
                             sharey=True)
    columns = ["qwk_mean"] + [f"qwk_{s}_mean" for s in sites]
    titles = ["all 60 patients"] + [f"{s} (20)" for s in sites]
    for ax, column, heading in zip(axes, columns, titles):
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        for y, candidate in enumerate(learned):
            a, b = volume.loc[candidate, column], full.loc[candidate, column]
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2,
                                        shrinkA=5, shrinkB=5), zorder=2)
            ax.scatter([a], [y], s=70, color=VOLUME_C, edgecolor="white", linewidth=1.4,
                       zorder=3)
            ax.scatter([b], [y], s=70, color=FULL_C, edgecolor="white", linewidth=1.4,
                       zorder=3)
        if "threshold_rule" in volume.index:
            ax.axvline(volume.loc["threshold_rule", column], color=INK_2, linestyle="--",
                       linewidth=1.1, zorder=1)
        ax.set_title(heading, fontsize=9.5, color=INK)
        ax.set_xlabel("QWK", color=INK_2, fontsize=9)
    axes[0].set_yticks(range(len(learned)))
    axes[0].set_yticklabels([CANDIDATE_LABEL[c] for c in learned], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", color=VOLUME_C, linestyle="",
                               markersize=8, label="volume only"),
                        Line2D([], [], marker="o", color=FULL_C, linestyle="",
                               markersize=8, label="volume + 5 pattern features"),
                        Line2D([], [], color=INK_2, linestyle="--", label="threshold rule")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.05))
    _stamp(fig, "Does the PATTERN of disease add anything to its AMOUNT?",
           "Each arrow runs from a model given volume alone to the same model given five "
           "pattern features too. Mean QWK over 10 repeats of 5-fold cross-validation; "
           "same folds throughout. Per-hospital panels show whether a gain is real or one "
           "hospital's.", commit)
    _reserve(fig, TITLE_SPACE)
    return fig


# ---------------------------------------------------------------------------
# 04 — the choice
# ---------------------------------------------------------------------------
def figure_choice(selection: dict, commit: str = "synthetic"):
    """Each promotion test: the QWK gain over the incumbent, with its 1-SE bar."""
    steps = selection["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 1.6 * len(steps) + 3.2),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    for y, step in enumerate(steps):
        colour = LEARNED_C if step["promoted"] else BASE_C
        ax.errorbar(step["qwk_difference"], y, xerr=step["bootstrap_se"], fmt="o",
                    markersize=9, color=colour, ecolor=colour, elinewidth=2.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        verdict = "PROMOTED" if step["promoted"] else "not promoted — selection stops"
        ax.text(step["qwk_difference"], y + 0.28,
                f"{step['qwk_difference']:+.3f} (SE {step['bootstrap_se']:.3f}) · {verdict}",
                ha="center", va="top", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([f"tier {s['tier']}: {s['challenger'].split('/')[-1]}\n"
                        f"vs {s['incumbent'].split('/')[-1]}" for s in steps],
                       fontsize=8.8, color=INK)
    ax.set_ylim(len(steps) - 0.4, -0.6)
    ax.set_xlabel("gain in mean QWK over the incumbent (bar = one paired-bootstrap SE)",
                  color=INK_2, fontsize=9)
    ax.set_title("condition 1: gain larger than one standard error", fontsize=9.5, color=INK)

    ax = axes[1]
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    sites = list(SITE_STYLE)
    for y, step in enumerate(steps):
        for k, site in enumerate(sites):
            offset = (k - 1) * 0.22
            colour, marker = SITE_STYLE[site]
            delta = step["site_qwk_challenger"][site] - step["site_qwk_incumbent"][site]
            ax.scatter(delta, y + offset, s=60, color=colour, marker=marker,
                       edgecolor="white", linewidth=1.2, zorder=3)
        wins = sum(step["wins_or_ties"].values())
        ax.text(1.0, y, f"{wins}/3 {'✓' if step['site_condition'] else '✗'}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9,
                color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([])
    ax.set_ylim(len(steps) - 0.4, -0.6)
    ax.set_xlabel("per-hospital QWK, challenger minus incumbent", color=INK_2, fontsize=9)
    ax.set_title("condition 2: win or tie on at least 2 of 3 hospitals", fontsize=9.5,
                 color=INK)
    ax.legend(handles=[Line2D([], [], color=SITE_STYLE[s][0], marker=SITE_STYLE[s][1],
                              linestyle="", markersize=7, label=s) for s in sites],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
              fontsize=8.5)
    _stamp(fig, f"The choice: {selection['selected'].split('/')[-1].replace('_', ' ')}",
           "The rule was fixed before any model ran. Each tier's best candidate must beat "
           "the simpler incumbent by more than one standard error AND on at least two of "
           "three hospitals, or selection stops.", commit)
    _reserve(fig, TITLE_SPACE)
    return fig


# ---------------------------------------------------------------------------
# 05 — what drives it
# ---------------------------------------------------------------------------
def figure_drivers(importance: pd.DataFrame, commit: str = "synthetic"):
    """Permutation importance on held-out patients, one panel per learned candidate."""
    candidates = list(dict.fromkeys(importance.candidate))
    fig, axes = plt.subplots(1, len(candidates), figsize=(5.2 * len(candidates), 4.6),
                             sharex=True, squeeze=False)
    for ax, cid in zip(axes[0], candidates):
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        part = importance[importance.candidate == cid]
        stats = part.groupby("feature").qwk_drop.agg(["mean", "std"]).sort_values("mean")
        y = np.arange(len(stats))
        ax.barh(y, stats["mean"], height=0.6, color=LEARNED_C, edgecolor="white",
                linewidth=2, zorder=3)
        ax.errorbar(stats["mean"], y, xerr=stats["std"], fmt="none", ecolor=INK_2,
                    elinewidth=1, zorder=4)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY.get(f, f) for f in stats.index], fontsize=9, color=INK)
        experiment, name = cid.split("/")
        ax.set_title(f"{CANDIDATE_LABEL.get(name, name)}\n({experiment.replace('_', ' ')})",
                     fontsize=9.5, color=INK)
        ax.set_xlabel("drop in QWK when shuffled", color=INK_2, fontsize=9)
    _stamp(fig, "What the decisions rest on",
           "Each feature shuffled among the held-out patients of every fold; the drop in "
           "QWK is how much the model was relying on it. Bars: mean over 10 repeats x 10 "
           "shuffles; whiskers: SD. Near zero = the model could do without it.", commit)
    _reserve(fig, TITLE_SPACE)
    return fig


# ---------------------------------------------------------------------------
# 06 — the three gallery patients
# ---------------------------------------------------------------------------
def figure_patients(panels: list, model_label: str, commit: str = "synthetic"):
    """Each patient: the FLAIR slice, our segmentation, and the class given vs true.

    `panels`: dicts with site, key, flair, reference, prediction (2-D slices),
    true_class, given_class, reference_ml, predicted_ml, agreement (share of
    repeats giving `given_class`).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.6), squeeze=False)
    classes = SEVERITY_CONFIG["classes"]
    for ax, panel in zip(axes[0], panels):
        ax.imshow(np.rot90(panel["flair"]), cmap="gray")
        for mask, colour, alpha in ((panel["reference"], TRUTH_C, 0.45),
                                    (panel["prediction"], PRED_C, 0.45)):
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = [*matplotlib.colors.to_rgb(colour), alpha]
            ax.imshow(np.rot90(rgba))
        ax.axis("off")
        right = panel["given_class"] == panel["true_class"]
        ax.set_title(f"{panel['site']}  ·  {panel['key'].split('_')[-1]}", fontsize=11,
                     fontweight="bold", color=INK)
        for row, (label, cls, volume) in enumerate((
                ("true", panel["true_class"], panel["reference_ml"]),
                ("given", panel["given_class"], panel["predicted_ml"]))):
            colour = CLASS_COLOURS[classes.index(cls)]
            ax.text(0.5, -0.06 - 0.085 * row, f"{label}: {cls}  ({volume:.1f} mL)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                    color="white" if classes.index(cls) > 0 else INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=colour, edgecolor="none"))
        ax.text(0.5, -0.25, ("✓ correct" if right else "✗ wrong")
                + f"   ({panel['agreement']:.0%} of repeats)", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(color=TRUTH_C, alpha=0.6, label="expert's lesions"),
                        Patch(color=PRED_C, alpha=0.6, label="our segmentation "
                              "(network that never saw this hospital)")],
               loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.12))
    _stamp(fig, "Three typical patients, graded",
           f"The median-burden patient at each hospital, as in every gallery. 'Given' is "
           f"{model_label}'s out-of-fold class — decided by a model that never saw this "
           f"patient. Overlap of green and red shows as brown.", commit)
    _reserve(fig, TITLE_SPACE + 0.3)      # two-line caption
    return fig


# ---------------------------------------------------------------------------
def _patient_panels(selection, logger):
    """Load the three gallery patients' images and their out-of-fold grading."""
    from classification.interpret import load_oof
    from metadata.derived import PRED_WMH_LOSO, load_derived_mask
    from metadata.loader import load_nifti, load_wmh_mask, subjects_by_key

    oof = load_oof(selection["selected"], "4class")
    features = pd.read_csv(CODE_ROOT / "features" / "outputs" / "features.csv")
    classes = SEVERITY_CONFIG["classes"]
    subjects = subjects_by_key()
    panels = []
    for site, key in GALLERY_PATIENTS.items():
        subject = subjects[key]
        flair = np.asarray(load_nifti(subject.flair_path).dataobj, dtype=float)
        reference = load_wmh_mask(subject.mask_path)
        prediction = load_derived_mask(key, PRED_WMH_LOSO)
        z = int(np.argmax(reference.reshape(-1, reference.shape[2]).sum(axis=0)))
        votes = oof[oof.subject_key == key].y_pred.value_counts()
        rows = features[features.subject_key == key].set_index("source")
        panels.append({
            "site": site, "key": key, "flair": flair[:, :, z],
            "reference": reference[:, :, z], "prediction": prediction[:, :, z],
            "true_class": classes[int(oof[oof.subject_key == key].y_true.iloc[0])],
            "given_class": classes[int(votes.idxmax())],
            "agreement": float(votes.max() / votes.sum()),
            "reference_ml": float(rows.loc["reference", "total_lesion_volume_ml"]),
            "predicted_ml": float(rows.loc["loso_prediction", "total_lesion_volume_ml"])})
    logger.info("gallery patients: %s", {p["key"]: (p["true_class"], p["given_class"])
                                         for p in panels})
    return panels


def main() -> None:
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    commit = get_git_commit_hash()
    classes = SEVERITY_CONFIG["classes"]

    def save(fig, relative):
        _save(fig, relative, commit, logger)

    volume = pd.read_csv(EXPERIMENTS_DIR / "summary_volume_only.csv")
    save(figure_scores(volume, "Volume alone: can a model that learns beat the cut-offs?",
                       "Every candidate given only our measured lesion volume. Mean ± SD over "
                       "10 repeats of 5-fold cross-validation, 60 patients, the same folds for "
                       "all. Dashed: the threshold rule; dotted: the majority-class floor.",
                       VOLUME_C, commit), "02-volume-only/scores.png")
    confusions = json.loads((EXPERIMENTS_DIR / "confusion_volume_only.json").read_text())
    confusions.update(json.loads((EXPERIMENTS_DIR / "confusion_baselines.json").read_text()))
    learned = volume[(volume.scheme == "4class")
                     & ~volume.candidate.isin(["majority_class", "threshold_rule"])]
    best = learned.sort_values("qwk_mean").candidate.iloc[-1]
    save(figure_confusions(confusions, ["4class/threshold_rule", f"4class/{best}"], classes,
                           "Where the errors fall: the threshold rule vs the best learned model",
                           "Share of each true class given each grade, pooled over 10 repeats. "
                           "The diagonal is correct; one step off it is a neighbouring class.", commit),
         "02-volume-only/confusion.png")

    full_path = EXPERIMENTS_DIR / "summary_full_features.csv"
    if not full_path.exists():
        logger.info("full_features not run yet — figures 03-06 wait for Week 6")
        return
    full = pd.read_csv(full_path)
    save(figure_pattern(volume, full, commit), "03-adding-the-pattern/volume_vs_pattern.png")

    selection_path = OUTPUTS_DIR / "selection.json"
    if not selection_path.exists():
        logger.info("selection.json missing — run classification.select for 04-06")
        return
    selection = json.loads(selection_path.read_text())
    save(figure_choice(selection, commit), "04-the-choice/selection_steps.png")

    importance_path = OUTPUTS_DIR / "interpretation" / "permutation_importance.csv"
    if importance_path.exists() and importance_path.stat().st_size > 1:
        save(figure_drivers(pd.read_csv(importance_path), commit),
             "05-what-drives-it/permutation_importance.png")
    else:
        logger.info("no learned candidate to read importance from (or interpret not run)")

    name = selection["selected"].split("/")[-1]
    save(figure_patients(_patient_panels(selection, logger), CANDIDATE_LABEL.get(name, name),
                         commit),
         "06-three-patients/graded.png")


if __name__ == "__main__":
    main()
