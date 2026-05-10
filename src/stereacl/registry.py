from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    slug: str
    title: str
    description: str
    script: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "01": ExperimentSpec(
        id="01",
        slug="layerwise_probing",
        title="Layer-wise Probing and Direction Extraction",
        description="Residual stream probing and stereotype-direction extraction.",
        script="experiments/01_layerwise_probing.py",
    ),
    "02": ExperimentSpec(
        id="02",
        slug="component_dla",
        title="Component-wise Direct Logit Attribution",
        description="Per-layer attention vs MLP direct logit attribution analysis.",
        script="experiments/02_component_dla.py",
    ),
    "03": ExperimentSpec(
        id="03",
        slug="attribution_patching",
        title="Attribution Patching Causal Validation",
        description="AtP* scoring and activation patching validation of top components.",
        script="experiments/03_attribution_patching.py",
    ),
    "04": ExperimentSpec(
        id="04",
        slug="ablation_validation",
        title="Direction and Component Ablation Validation",
        description="Held-out causal interventions with capability controls.",
        script="experiments/04_ablation_validation.py",
    ),
    "05": ExperimentSpec(
        id="05",
        slug="cross_cultural_shift",
        title="Cross-cultural Component Shift",
        description="Cross-culture direction overlap and component overlap analysis.",
        script="experiments/05_cross_cultural_shift.py",
    ),
    "06": ExperimentSpec(
        id="06",
        slug="sae_corroboration",
        title="SAE Feature Corroboration",
        description="Map implicated components to SAE feature-level evidence.",
        script="experiments/06_sae_corroboration.py",
    ),
    "07": ExperimentSpec(
        id="07",
        slug="rank_sweep",
        title="Rank-Sweep Causal Curves",
        description="Ablate rank-k stereotype subspace for k in {1,2,4,8,16,32} to produce causal effect curves.",
        script="experiments/07_rank_sweep.py",
    ),
    "08": ExperimentSpec(
        id="08",
        slug="dose_response",
        title="Signed Dose-Response Direction Injection",
        description="Sweep signed injection strengths (alpha) of normalized stereotype direction.",
        script="experiments/08_dose_response.py",
    ),
    "09": ExperimentSpec(
        id="09",
        slug="dla_atp_adjudication",
        title="DLA vs AtP Adjudication",
        description="Single-component causal ablation for union of top-k DLA and AtP rankings.",
        script="experiments/09_dla_atp_adjudication.py",
    ),
    "10": ExperimentSpec(
        id="10",
        slug="path_mediation",
        title="Layer-wise Residual Path Mediation",
        description="For each layer, replace stereo residual with anti residual to identify causal mediators.",
        script="experiments/10_path_mediation.py",
    ),
    "11": ExperimentSpec(
        id="11",
        slug="hydra_multisite",
        title="Hydra / Self-Repair Multi-site Test",
        description="Compare single-site vs multi-site synchronized ablation to detect compensation effects.",
        script="experiments/11_hydra_multisite.py",
    ),
    "12": ExperimentSpec(
        id="12",
        slug="local_atlas",
        title="Local Geometry Atlas (Principal Angles)",
        description="Compare stereotype-direction subspaces across axes and datasets via principal angles.",
        script="experiments/12_local_atlas.py",
    ),
    "13": ExperimentSpec(
        id="13",
        slug="cross_model_transfer",
        title="Cross-Model Direction and Ranking Transfer",
        description="Test whether directions and component rankings transfer between Gemma and Llama.",
        script="experiments/13_cross_model_transfer.py",
    ),
    "14": ExperimentSpec(
        id="14",
        slug="sign_reliability_audit",
        title="Sign Reliability Audit",
        description="Audit DLA/AtP signed agreement with causal direction from Exp09 ablations.",
        script="experiments/14_sign_reliability_audit.py",
    ),
    "15": ExperimentSpec(
        id="15",
        slug="cross_dataset_component_transfer",
        title="Cross-Dataset Component Transfer Matrix",
        description="2x2 matrix of source-ranked components transferred across StereoSet and CrowS evaluation sets.",
        script="experiments/15_cross_dataset_component_transfer.py",
    ),
    "16": ExperimentSpec(
        id="16",
        slug="asymmetry_matrix",
        title="Inject/Remove Asymmetry 2x2 Matrix",
        description="Evaluate remove/inject interventions on both stereotype and anti-stereotype base distributions.",
        script="experiments/16_asymmetry_matrix.py",
    ),
    "17": ExperimentSpec(
        id="17",
        slug="suppressor_contamination_audit",
        title="Causal Suppressor Contamination Audit",
        description="Estimate suppressor/promoter contamination in top-k sets using Exp09 causal labels.",
        script="experiments/17_suppressor_contamination_audit.py",
    ),
    "18": ExperimentSpec(
        id="18",
        slug="injection_controls",
        title="Injection Specificity Controls",
        description="Compare true stereotype-direction injection against norm-random and shuffled-axis injection controls.",
        script="experiments/18_injection_controls.py",
    ),
    "19": ExperimentSpec(
        id="19",
        slug="seegull_loso_sensitivity",
        title="SEEGeL Leave-One-Source-Out Sensitivity",
        description="Measure nationality ablation sensitivity on SEEGeL test pairs with and without SEEGeL in direction training.",
        script="experiments/19_seegull_loso_sensitivity.py",
    ),
    "20": ExperimentSpec(
        id="20",
        slug="same_position_validity",
        title="Same-Position vs Cross-Position Direction Validity",
        description="Compare same-position prediction-direction edits against cross-position trait-direction edits with norm-random controls.",
        script="experiments/20_same_position_validity.py",
    ),
    "21": ExperimentSpec(
        id="21",
        slug="transfer_equivalence",
        title="Transfer Equivalence Framing",
        description="Apply SESOI-based equivalence interpretation to Exp15 transfer condition summaries.",
        script="experiments/21_transfer_equivalence.py",
    ),
    "22": ExperimentSpec(
        id="22",
        slug="head_path_decomposition",
        title="Head/Path Routing Decomposition",
        description="Decompose top causal layers into residual, attention, MLP, and per-head pathway effects.",
        script="experiments/22_head_path_decomposition.py",
    ),
    "23": ExperimentSpec(
        id="23",
        slug="prospective_diagnostics_meta",
        title="Prospective Diagnostics Meta Summary",
        description="Aggregate Exp14/17/18 diagnostics under a frozen model-family multiple-testing scope.",
        script="experiments/23_prospective_diagnostics_meta.py",
    ),
    "24": ExperimentSpec(
        id="24",
        slug="backfire_stratified_test",
        title="Suppressor-Stratified Backfire Test",
        description="Construct multi-site ablation sets by suppressor fraction and test one-sided backfire hypotheses.",
        script="experiments/24_backfire_stratified_test.py",
    ),
    "25": ExperimentSpec(
        id="25",
        slug="capability_monitor_replacement",
        title="Capability Monitor Replacement",
        description="Run larger-sample paired capability deltas on above-chance benchmarks under direction ablation.",
        script="experiments/25_capability_monitor_replacement.py",
    ),
}


def normalize_experiment_id(raw: str | int) -> str:
    if isinstance(raw, int):
        return f"{raw:02d}"
    stripped = str(raw).strip()
    if not stripped:
        raise ValueError("Experiment id cannot be empty.")
    if stripped.isdigit():
        return f"{int(stripped):02d}"
    return stripped


def get_experiment(experiment_id: str | int) -> ExperimentSpec:
    normalized = normalize_experiment_id(experiment_id)
    if normalized not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment id '{normalized}'. Known ids: {known}")
    return EXPERIMENTS[normalized]


def list_experiments() -> list[ExperimentSpec]:
    return [EXPERIMENTS[key] for key in sorted(EXPERIMENTS)]
