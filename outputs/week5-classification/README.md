# week5-classification/

Weeks 5-6 — sorting each patient into **normal / mild / moderate / severe** WMH
burden (R10, bonus), using the Fazekas-anchored volume cut-offs of Joo et al.
(PLOS ONE, 2022): 3.4 / 9.6 / 17.1 mL. Every score is from 10 repeats of 5-fold
cross-validation on the 60 training patients, with features measured by
networks that never saw the patient's hospital.

| Folder | What it shows |
|---|---|
| `01-exploring-the-data/` | The problem before any model: class sizes, cut-offs, which features carry anything |
| `02-volume-only/` | Experiment 2 — every model given volume alone, against the plain cut-offs |
| `03-adding-the-pattern/` | Experiment 3 — does WHERE and HOW the disease sits add to HOW MUCH? |
| `04-the-choice/` | The selection rule, fixed before any model ran, applied step by step |
| `05-what-drives-it/` | Which measurements the models actually lean on |
| `06-three-patients/` | The gallery's three patients: the class given beside the true one |

**The result:** the rule selected the **threshold rule** — no learned model beat it
by more than its own uncertainty.

The same three patients as every other week: Amsterdam 112, Singapore 64,
Utrecht 49, the median lesion-burden patient at each hospital.

Regenerate with `code/.venv/bin/python -m classification.make_outputs`
(and `classification.explore` for `01`). Generated from commit `dbaa59f6`.
