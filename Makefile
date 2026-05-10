PYTHON ?= python

.PHONY: list smoke compile run-01 run-02 run-03 run-04 run-05 run-06

list:
	$(PYTHON) tools/run_experiment.py --list

run-01:
	$(PYTHON) experiments/01_layerwise_probing.py --dry-run

run-02:
	$(PYTHON) experiments/02_component_dla.py --dry-run

run-03:
	$(PYTHON) experiments/03_attribution_patching.py --dry-run

run-04:
	$(PYTHON) experiments/04_ablation_validation.py --dry-run

run-05:
	$(PYTHON) experiments/05_cross_cultural_shift.py --dry-run

run-06:
	$(PYTHON) experiments/06_sae_corroboration.py --dry-run

smoke: run-01 run-02 run-03 run-04 run-05 run-06

compile:
	$(PYTHON) tools/compile_results.py
