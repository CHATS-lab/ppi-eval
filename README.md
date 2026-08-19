# Prediction-Powered MT Evaluation

This repository contains the reproducibility code for the experiments in
**Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation
and Meta-Evaluation**.

The code has two goals:

- Reproduce the paper's empirical results from exported WMT metric evaluation
  score files.
- Provide reusable implementations of the paired tests, prediction-powered
  intervals, and the Prediction-Powered Saving Ratio (PPSR).

## Repository Structure

- `configs/`: dataset definitions and metric filtering rules used by the
  paper experiments.
- `datasets/`: exported WMT score TSVs and `dataset_summary.csv`. These files
  are generated from `mt-metrics-eval`.
- `experiments/`: command-line scripts for each paper experiment.
- `src/ppi_mt_eval/`: reusable library code for data loading, paired tests,
  interval estimators, PPSR/meta-metrics, plotting, and progress reporting.
- `scripts/`: convenience scripts for reproducing the full experimental
  pipeline.
- `results/`: generated CSV tables, figures, configs, and runtime logs.

## Setup

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The dataset export step requires `mt-metrics-eval`. If it is unavailable from
your package index, install it from the
[official repository](https://github.com/google-research/mt-metrics-eval)
following its documentation.

## Export Datasets

The WMT score TSVs are generated with `mt-metrics-eval`.

The paper uses six datasets:

- WMT22: `en-de`, `en-ru`, `zh-en`
- WMT23: `en-zh`, `ja-en`
- WMT24: `cs-uk`

To export the WMT score TSVs used by the paper, run:

```bash
python experiments/export_datasets.py
```

This writes files such as `datasets/wmt24.cs-uk.tsv` and creates a dataset
summary at `datasets/dataset_summary.csv`.

If you already have compatible TSV files, place them under `datasets/` using the
same naming convention:

```text
datasets/<test-set>.<language-pair>.tsv
```

For example:

```text
datasets/wmt22.en-de.tsv
datasets/wmt24.cs-uk.tsv
```

## Full Reproduction

Run all paper experiments with:

```bash
bash scripts/run_all.sh
```

The script runs the pipeline in a fixed order: dataset export, basic power
analysis, confidence-interval experiments, non-parametric power analysis, Type I
error simulations, paired-vs-unpaired variance analysis, PPSR/meta-metric
analysis, ranking-stability experiments, metric score/rank table generation, and
annotation-saving validation.

Full reproduction can be computationally expensive, especially for
permutation-based experiments and threshold searches. All outputs are written
under `results/`.

Each experiment script can also be run directly. For example:

```bash
python experiments/ppsr_discriminative_power.py --num-workers 6
python experiments/ppsr_ranking_stability.py --num-workers 6
```

Use `--help` on any experiment script to inspect its available options.

## Experiment Outputs

Output directories include:

- `results/basic_power`: empirical power for human-only, auto-only, and
  prediction-powered paired Z-tests.
- `results/confidence_intervals_basic`: confidence-interval width and coverage
  for human-only and prediction-powered intervals using MetricX/BLEU.
- `results/confidence_intervals_gemba`: GEMBA auto-only, human-only, and
  prediction-powered confidence intervals.
- `results/nonparam_power_analysis`: paired Z-test versus paired permutation
  test power comparisons.
- `results/type_i_error_simulation`: synthetic Type I error simulations for
  standard Z, oracle Z, and permutation tests.
- `results/type_i_error_metric_mean_shift_simulation`: synthetic Type I error
  simulations illustrating the role of centering metric scores in the
  prediction-powered permutation test.
- `results/paired_vs_unpaired`: paired versus unpaired variance comparisons.
- `results/ppsr_discriminative_power`: discriminative power for PPSR and
  system-level meta-metrics.
- `results/segment_meta_metric_discriminative_power`: discriminative power for
  PPSR and segment-level meta-metrics.
- `results/ppsr_ranking_stability`: ranking stability for PPSR and system-level
  meta-metrics.
- `results/segment_meta_metric_ranking_stability`: ranking stability for PPSR
  and segment-level meta-metrics.
- `results/ppsr_metric_rank_tables`: LaTeX tables of metric scores and ranks.
- `results/annotation_saving_vs_ppsr`: relationship between PPSR and empirical
  annotation savings.
