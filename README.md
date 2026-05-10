# StereACL Experiment Scaffold

This project scaffold is built around the experimental plan in `StereotypeOverview.md` and is designed to enforce:

1. Numbered experiment scripts in one folder.
2. Results organized by experiment and date.
3. A single compiled, project-wide result summary.

Experiments `01` through `06` now include executable implementations.

## Structure

```text
StereACL/
├── StereotypeOverview.md
├── README.md
├── Makefile
├── requirements.txt
├── configs/
│   └── experiment_defaults.yaml
├── experiments/
│   ├── 01_layerwise_probing.py
│   ├── 02_component_dla.py
│   ├── 03_attribution_patching.py
│   ├── 04_ablation_validation.py
│   ├── 05_cross_cultural_shift.py
│   └── 06_sae_corroboration.py
├── src/
│   └── stereacl/
│       ├── __init__.py
│       ├── registry.py
│       └── run_context.py
├── tools/
│   ├── run_experiment.py
│   └── compile_results.py
└── results/
    ├── README.md
    └── summary/
        └── README.md
```

## Result Layout

Each experiment run writes to:

```text
results/<NN_experiment_slug>/<YYYY-MM-DD>/run-XXX/
```

Each run folder contains:

- `manifest.json`: canonical run metadata (params, timing, status, metrics, artifacts).
- `tables/`: tabular outputs.
- `figures/`: figure assets or placeholders.
- `artifacts/`: caches/checkpoints/intermediate outputs.
- `logs/events.jsonl`: structured event log for the run.

## Quick Start

```bash
cd /jumbo/lisp/f004ndc/StereACL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

List experiments:

```bash
python tools/run_experiment.py --list
```

Run one scaffolded experiment:

```bash
python experiments/01_layerwise_probing.py --dry-run
```

Run all dry-run scaffolds:

```bash
make smoke
```

Compile every run into one global summary:

```bash
python tools/compile_results.py
```

Compiled outputs are written to `results/summary/`.

## Implemented Pipeline (Current)

1. `01_layerwise_probing.py`
   - Loads contrast pairs from StereoSet intrasentence, CrowS-Pairs (raw CSV), and SeeGULL global v2.
   - Filters to single-token stereotype/anti-stereotype differences under the selected model tokenizer.
   - Builds deterministic train/test split.
   - Extracts layer-wise residual activations at the prediction position.
   - Computes per-axis direction vectors and layer-wise probe AUCs.
   - Saves direction artifacts for downstream experiments.

2. `02_component_dla.py`
   - Loads the latest completed Experiment 01 directions + aligned pairs.
   - Captures attention-block and MLP-block writes via forward hooks.
   - Performs true per-head decomposition by capturing attention output projection inputs (`o_proj`/`c_proj`) and splitting projected writes by head.
   - Computes DLA-like scores for each component against token-logit difference.
   - Computes cosine with Experiment 01 direction vectors.
   - Writes unified component tables, dedicated per-head tables, and top-K component lists per axis.

3. `03_attribution_patching.py`
   - Loads top components from Experiment 02.
   - Computes gradient-attribution scores (`grad * activation`) per component.
   - Performs anti->stereo activation replacement validation for top components.
   - Reports DLA-vs-attribution rank agreement (Spearman when enough components).

4. `04_ablation_validation.py`
   - Loads held-out pairs and directions from Experiment 01.
   - Loads top components from Experiment 03.
   - Evaluates baseline vs direction-ablation vs component-ablation vs combined.
   - Reports stereotype score and margin metrics on held-out pairs.
   - Optional lightweight BBQ accuracy check (`--bbq-samples N`).

5. `05_cross_cultural_shift.py`
   - Builds culture-specific subsets (`us_english`, `latam_spanish_proxy`, `south_asia_hindi_proxy`) from StereoSet/CrowS and SeeGULL.
   - Re-runs direction extraction and component scoring per culture.
   - Computes cross-culture direction similarity and top-component Jaccard overlap.

6. `06_sae_corroboration.py`
   - Loads top localized components and aligned pairs from prior experiments.
   - Extracts component activation vectors (MLP, attention block, and attention head).
   - Fits a sparse dictionary-learning model as an SAE-style proxy when SAE checkpoints are unavailable.
   - Produces component-feature correspondence and atom-vs-direction similarity tables.

## Notes

- Default model is set to `gpt2` for practical local runs. Use `--model google/gemma-2-2b` (or another target) for your main experiments.
- Direction ablation is currently implemented as projection hooks on attention/MLP writes per layer (a practical approximation of residual-stream projection).
- Dry-run runs are intentionally lightweight and may not emit full artifact tables. Downstream experiments now select the latest run that includes required artifacts.
