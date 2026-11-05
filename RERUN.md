# Rerunning everything from scratch

The repo holds only code and results. The Python environment, the C++ build and all data are rebuilt or downloaded by one script, and anything already present is skipped.

## 1. Prerequisites

| Need | Notes |
|---|---|
| Python 3.11+ | On Windows the script uses `py -3` (the newest installed Python); it stops with a clear message if the version is too old |
| `bash` + `curl` | Git Bash on Windows |
| C++17 compiler | Linux: GCC or Clang. Windows: any Visual Studio 2019+ or Build Tools with "Desktop development with C++" (found automatically) |
| NVIDIA GPU | Optional; only M5 uses it, and it falls back to CPU |
| ~12 GB free disk | Full run (sum of the table below is ~11 GB) |

## 2. One command

```bash
git clone https://github.com/Mourya-wizard/raw2event-fx.git
cd raw2event-fx
RUN_ALL=1 bash scripts/reproduce_all.sh
```

This runs, in order: create `.venv` and install `requirements.txt` → build the C++ core → unit tests → generate synthetic scenes → M1 → M2/M3/M4 → M5 (MNIST) → M6 (CED) → M7 (Raw2Event) → `results/hero.png`.

To run less, leave out `RUN_ALL=1` (synthetic M1–M4 only) and add the pieces you want:

```bash
bash scripts/reproduce_all.sh                              # M1–M4
RUN_MNIST=1 bash scripts/reproduce_all.sh                  # + M5
RUN_CED=1 bash scripts/reproduce_all.sh                    # + M6
RUN_RAW2EVENT=1 bash scripts/reproduce_all.sh              # + M7
```

## 3. What gets downloaded or generated

Sizes are as measured on the author's machine.

| Item | Source | Location | Size |
|---|---|---|---|
| Python environment (PyTorch CUDA 12.8, NumPy, PyAV, aedat, CMake, Ninja, …) | PyPI + download.pytorch.org | `.venv/` | 5.0 GB |
| C++ build | compiled locally | `build/` | 0.5 MB |
| Synthetic scenes (M1–M4) | generated locally | `data/*.bin` | ~0.7 GB |
| MNIST (M5) | torchvision auto-download | `data/mnist/` | 64 MB |
| CED calibration bags (M6) | https://rpg.ifi.uzh.ch/datasets/CED/CED_calibration.zip | `data/ced/` | 1.0 GB zip (deleted automatically after extraction) → ~1.6 GB of bags |
| Raw2Event, 20 recordings listed in `results/m7_prefixes.txt` (M7) | https://huggingface.co/datasets/raw2event/raw2event | `data/raw2event/` | 2.0 GB |
| Experiment scratch files | written during runs | `results/raw/` | 0.7 GB |

No pretrained models are needed; the M5 classifier trains from scratch during the run.

After a run you can delete `results/raw/` to get space back; the next run recreates it.

## 4. Interrupted runs

Just rerun the same command; it picks up where it stopped:

- **Package install:** `.venv/.deps_installed` is written only after `pip install` completes, so an interrupted install resumes.
- **Downloads:** finished files are skipped, a failed download stops the script with an error, and partial files (`*.part`) are never mistaken for complete ones.
- **CED unzip:** `data/ced/.extracted` is written only after a complete unzip, so an interrupted unzip is redone.

## 5. Results get overwritten

A rerun rewrites `results/`. Synthetic accuracy (M1–M3) and real-data (M6, M7) numbers reproduced byte-identically in testing; timing (M4) and MNIST accuracy (M5) vary slightly between runs and machines (up to 3.3% and 0.003 in testing). To restore the published files:

```bash
git checkout results/
```

## 6. Freeing space again

Everything below is rebuilt by the script:

```bash
rm -rf .venv build data results/raw .cache .pytest_cache
```
