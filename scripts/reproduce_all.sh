#!/usr/bin/env bash
# Reproduce from a clean clone: venv -> build -> tests -> synthetic experiments (M1-M4).
# Optional: RUN_MNIST=1 (M5), RUN_CED=1 (M6, real events), RUN_RAW2EVENT=1 (M7, real events).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=./.cache/huggingface TORCH_HOME=./.cache/torch XDG_CACHE_HOME=./.cache MPLCONFIGDIR=./.cache/mpl
export PIP_REQUIRE_VIRTUALENV=1

if [ ! -d .venv ]; then
  if command -v py >/dev/null 2>&1; then py -3 -m venv .venv; else python3 -m venv .venv; fi
  NEW_VENV=1
fi
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; BIN=.venv/Scripts; else PY=.venv/bin/python; BIN=.venv/bin; fi
if [ "${NEW_VENV:-0}" = 1 ]; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
fi

# Build (Windows: MSVC loaded per-process by build_msvc.bat; elsewhere: g++/clang via venv cmake)
if [ "${OS:-}" = "Windows_NT" ]; then
  cmd //c ".\\scripts\\build_msvc.bat"
else
  "$BIN/cmake" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$PWD/$BIN/ninja"
  "$BIN/cmake" --build build
fi

"$PY" -m pytest -q                                   # unit tests (metrics + fixed-vs-float C++ check)
"$PY" -m raw2event.scenes                            # synthetic data -> data/*.bin (~680 MB, ~10 s)
"$PY" -m experiments.sanity_check                    # M1
"$PY" -m experiments.synthetic_benchmarks            # M2 bit-width, M3 ablation, M4 latency (run on a quiet CPU)

# M5: downstream MNIST task (GPU if available)
if [ "${RUN_MNIST:-0}" = 1 ]; then "$PY" -m experiments.mnist_task; fi

# M6: real Color-DAVIS346 events (CED calibration bags, 1.05 GB download)
if [ "${RUN_CED:-0}" = 1 ]; then
  if [ ! -f data/ced/calib_low.bag ]; then
    mkdir -p data/ced
    curl -L -o data/ced/CED_calibration.zip https://rpg.ifi.uzh.ch/datasets/CED/CED_calibration.zip
    "$PY" -c "import zipfile; zipfile.ZipFile('data/ced/CED_calibration.zip').extractall('data/ced')"
  fi
  "$PY" -m experiments.real_ced
fi

# M7: real DAVIS346 events (Raw2Event dataset, 20 recordings, ~2.14 GB download)
if [ "${RUN_RAW2EVENT:-0}" = 1 ]; then
  B=https://huggingface.co/datasets/raw2event/raw2event/resolve/main
  mkdir -p data/raw2event
  while read -r P; do
    for f in frames_raw/$P.mkv events_aedat4/$P.aedat4 meta_raw/$P.dat; do
      [ -f "data/raw2event/$(basename "$f")" ] || curl -sL -o "data/raw2event/$(basename "$f")" "$B/$f"
    done
  done < results/m7_prefixes.txt
  "$PY" -m experiments.real_raw2event results/m7_prefixes.txt
fi

"$PY" -m experiments.make_figures                    # results/hero.png dashboard from results/*.csv
