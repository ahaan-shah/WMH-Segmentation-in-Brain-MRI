# outputs/

Every result, in the order it happened, named in plain words.

**How to read these.** Every image contains its own point of comparison. The
panel with the **red title or red border is the "before"** — the untouched
original — and everything to the right of it is what we did. No image here needs
another image to make sense of it.

Start with **`week2-preprocessing/00-overview/full-pipeline-all-sites.png`** — it
shows the whole week left to right on one page.

## week2-preprocessing/

| Folder | What it shows | Requirement |
|---|---|---|
| `00-overview/` | **Start here.** The entire week, left to right, one row per hospital | — |
| `01-head-found/` | Separating head from empty air | — |
| `02-bias-corrected/` | Removing the scanner's brightness gradient | **R3** |
| `03-skull-stripped/` | Removing skull and scalp | **R1** |
| `04-tissue-sorted/` | Sorting brain into fluid / grey / white matter | — |
| `05-brightness-normalised/` | Putting all three hospitals on one scale | **R2** |
| `06-smoothing-REJECTED/` | A method we tested and chose NOT to use, with the evidence | — |
| `07-cleanup-tools/` | Tools built for Week 3, and what they cost | — |
| `quality-checks/` | Every patient checked, not just the pretty ones | — |

## The patients shown

Amsterdam 112, Singapore 64, Utrecht 49 — the **median lesion-burden patient at each hospital**, so they are
typical rather than flattering. The same three appear in every figure, so the
gallery reads as one continuous story.

## Regenerating

```
code/.venv/bin/python -m preprocessing.make_outputs
```

Every figure is produced by that one script and stamped with the git commit it
came from. Nothing here is a screenshot or hand-edited, so any figure can be
traced back to the exact code that made it.

`preprocessing/outputs/` remains the pipeline's working directory — raw QC
tables and the figures the individual stage scripts emit. This folder is the
curated version for humans.

Generated from commit `ce6a93b6`.
