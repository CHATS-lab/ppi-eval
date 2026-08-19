#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(pwd)/.cache}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

# 1. Export the WMT score TSVs and dataset summary.
python experiments/export_datasets.py

# 2. Basic system-comparison experiments.
python experiments/basic_power.py

python experiments/confidence_intervals.py \
  --mode basic \
  --output-dir results/confidence_intervals_basic

# 3. GEMBA auto-only interval experiment.
python experiments/confidence_intervals.py \
  --mode gemba \
  --output-dir results/confidence_intervals_gemba

# 4. Parametric versus non-parametric paired tests.
python experiments/nonparam_power.py

# 5. Synthetic Type I error simulations.
python experiments/type_i_error_simulation.py

python experiments/type_i_error_metric_mean_shift_simulation.py

# 6. Paired versus unpaired variance comparisons.
python experiments/paired_vs_unpaired.py

# 7. Discriminative power of PPSR and related meta-metrics.
python experiments/ppsr_discriminative_power.py --num-workers 6

python experiments/segment_meta_metric_discriminative_power.py --num-workers 6

# 8. Ranking stability under input subsampling.
python experiments/ppsr_ranking_stability.py --num-workers 6

python experiments/segment_meta_metric_ranking_stability.py --num-workers 6

# 9. Metric score/rank tables for the paper appendix.
python experiments/ppsr_metric_rank_tables.py

# 10. Empirical annotation savings versus PPSR.
python experiments/annotation_saving_vs_ppsr.py

echo "Full reproduction completed."
