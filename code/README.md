# `code/` — all source for the WMH segmentation pipeline

Five importable packages plus the one virtual environment everything runs from.

```
code/
├── .venv/           the single venv (gitignored)
├── metadata/        shared infrastructure — config, paths, loaders, provenance,
│                    the subject index and the frozen splits
├── checks/          the phantom test suite, metrics, and the vendored official
│                    challenge scorer
├── preprocessing/   Week 2 — skull stripping, bias correction, normalisation
├── segmentation/    Week 3 — the threshold baseline and the U-Net
└── features/        Week 4 — ventricles, the 10 mm split, lesion measurements
```

Everything *not* source lives one level up, at the project root: `data/`,
`outputs/` (the figure gallery), `reports/`, `deliverables/`.

## One-time setup after creating the venv

Source lives under `code/`, so Python has to be told where to find it before
`python -m segmentation.train_unet` will resolve. Run once, **from the project
root**:

```bash
echo "$(pwd)/code" > "$(code/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/wmh_project.pth"
```

That writes a `.pth` file into the venv's site-packages, which puts `code/` on
the import path for every run of that interpreter. It lives inside `.venv/`, so
it is not in git and each clone does this once.

Two things deliberately do not depend on it:

- **pytest** — `pytest.ini` sets `pythonpath = code` itself.
- **the shell scripts** in `segmentation/` — they export `PYTHONPATH` before
  running anything.

## Running things

Every command runs **from the project root**, not from inside `code/`:

```bash
code/.venv/bin/python -m pytest -q                      # 92 tests
code/.venv/bin/python -m segmentation.train_unet        # train one network
code/.venv/bin/python -m features.run_features          # R5-R9 measurements
bash code/segmentation/train_ensemble.sh 1 2 3          # queue three networks
```

## Two roots, kept distinct

`metadata/config.py` defines both, and the distinction is load-bearing:

| Constant | Points at | Holds |
|---|---|---|
| `PROJECT_ROOT` | the repository | `data/`, `outputs/`, `reports/` — things a human reads |
| `CODE_ROOT` | `code/` | the packages, and each one's `outputs/` beside its source |

So a per-package QC file goes under `CODE_ROOT`, while a gallery figure or
anything in `data/` goes under `PROJECT_ROOT`. Paths are always built from one
of these two, never from the current working directory.

## The conventions each package follows

- **Method file + driver file (+ sweep file where a choice existed).** Method
  files take arrays and return arrays — no paths, no I/O — which is what makes
  them testable against a phantom with a known answer.
- **`metadata/derived.py` is the only sanctioned way to write a derived volume.**
  It enforces raw-data immutability, FLAIR-grid geometry, storage orientation,
  provenance sidecars, and the interim/processed split.
- **`metadata/loader.py::load_wmh_mask` is the only sanctioned way to read a
  mask**, so label 2 can never be included by accident.
