"""Load and validate the project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TBD = Literal["TBD"]
TBD_S07 = Literal["TBD-S07"]
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataCfg(_Strict):
    raw_dir: str
    processed_dir: str
    charge_state: int = 0
    filters: list[str] = Field(default_factory=list)
    universe: str | TBD = "TBD"
    exclude_never_tested_from_per_host: bool = True


class KiyoharaProtocolCfg(_Strict):
    train: Literal["train"] = "train"
    early_stopping: Literal["val"] = "val"
    test: Literal["test"] = "test"


class SplitCfg(_Strict):
    group_key: Literal["formula"] = "formula"
    n_outer_resamples: int = Field(10, gt=0)
    test_fraction: float = Field(gt=0, lt=1)
    seeds: list[int]
    kiyohara_protocol: KiyoharaProtocolCfg


class BudgetsCfg(_Strict):
    hosts: list[int] = Field(default_factory=list)
    max: int | TBD


class CurveCfg(_Strict):
    sizes: list[int]
    n_resamples: int = Field(gt=0)
    n_test_hosts: int = Field(gt=0)


class GraphCfg(_Strict):
    structure: Literal["pristine_release_supercell"]
    vacancy_atom: Literal["kept_with_binary_flag"]
    type: Literal["radius"]
    cutoff_A: float = Field(gt=0)
    periodic: bool
    node_features: list[str]
    edge_features: Literal["gaussian_distance_expansion"]
    gaussian_centres: Literal["megnet_default"] | int
    gaussian_width: Literal["megnet_default"] | float


class ToggleCfg(_Strict):
    enabled: bool = True


class PhysicsFloorCfg(ToggleCfg):
    n_parameters: int = 2
    descriptors: list[str] = Field(min_length=2, max_length=2)
    intercept: bool = True


class NetCfg(ToggleCfg):
    backbone: Literal["megnet"] = "megnet"


class SNetCfg(NetCfg):
    backbone_package: Literal["matgl"]
    backbone_version: str
    backbone_backend: Literal["pyg"]
    readout: list[str]
    pooling: Literal["tuned"]


DescriptorClass = Literal["structural/compositional", "host-electronic DFT", "site-electronic DFT"]
InjectionVariant = Literal["D-state", "D-late"]


class DNetCfg(NetCfg):
    descriptor_classes: list[DescriptorClass] = Field(min_length=1)
    standardisation: Literal["training_set"]
    injection_variants: dict[InjectionVariant, dict[DescriptorClass, str]]
    injection_selection: Literal["lowest_mean_val_mae_over_anchors"]


class P1NetCfg(NetCfg):
    architecture: Literal["S_backbone_and_readout"]
    head: str
    loss: Literal["mse"]
    loss_weights: Literal["equal"]
    report: list[str]


class PNetCfg(NetCfg):
    composition: list[Literal["P1", "D"]]
    D_trained_on: Literal["true_descriptors"]
    true_descriptor_swap_in: Literal["D"]


class ModelsCfg(_Strict):
    rf: ToggleCfg
    physics_floor: PhysicsFloorCfg
    S: SNetCfg
    D: DNetCfg
    P1: P1NetCfg
    P: PNetCfg
    injection_mode: InjectionVariant | TBD


class SearchParamCfg(_Strict):
    dist: Literal["log_uniform", "uniform", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list[int | str] | None = None


class OptunaCfg(_Strict):
    sampler_seed: int
    pruner: Literal["none"]


class TuningCfg(_Strict):
    sampler: Literal["optuna_tpe"]
    optuna_trials_per_model_per_anchor: int = Field(gt=0)
    models: list[str]
    anchors: list[int]
    anchor_resample: int = Field(ge=0)
    objective: Literal["inner_val_mae"]
    budget_to_anchor: dict[int, int]
    search_space: dict[str, SearchParamCfg]
    trial_seed: int = 0                     # clarification (2026-10-07)
    nonfinite_objective: float = 1000.0     # clarification (2026-10-07)
    optuna: OptunaCfg = OptunaCfg(sampler_seed=0, pruner="none")


class SchedulerCfg(_Strict):
    name: Literal["reduce_lr_on_plateau"]
    factor: float = Field(gt=0, lt=1)
    patience: int = Field(gt=0)


class EarlyStoppingCfg(_Strict):
    monitor: Literal["val_mae"]
    restore_best: bool
    patience: int | TBD_S07
    min_delta: float = Field(ge=0)


class ValidationCfg(_Strict):
    frac: float = Field(gt=0, lt=1)
    min_hosts: int = Field(gt=0)
    seed_from: list[Literal["resample", "budget", "seed"]]
    seed_rule: Literal["sha256_val_r_B_seed_first4_bigendian"]


class TrainingCfg(_Strict):
    loss: Literal["l1_standardised_target"]
    optimiser: Literal["adamw"]
    scheduler: SchedulerCfg
    max_epochs: int | TBD_S07
    early_stopping: EarlyStoppingCfg
    validation: ValidationCfg
    seeds: list[int]
    checkpoints: list[str]


class SecondaryMetricCfg(_Strict):
    within_host: bool | TBD = "TBD"


class StatsCfg(_Strict):
    bootstrap_n: int = Field(gt=0)
    cluster_bootstrap_n: int = Field(2000, gt=0)
    delta_eV: float = Field(gt=0)
    ci: float = Field(gt=0, lt=1)


class BootstrapCfg(_Strict):
    type: Literal["paired_hierarchical"]
    levels: list[Literal["resample", "test_host"]]
    draws: int = Field(gt=0)


class NStarCfg(_Strict):
    upper_quantile: float = Field(gt=0, lt=1)
    alpha: float = Field(gt=0, lt=1)
    must_hold_for_all_larger_budgets: bool
    if_none: Literal["lower_bound_gt_max_budget"]


class AnalysisCfg(_Strict):
    primary_metric: Literal["site_mae"]
    secondary_metrics: list[str]
    advantage: Literal["mae_S_minus_mae_D"]
    prediction: Literal["seed_ensemble_mean"]
    bootstrap: BootstrapCfg
    n_star: NStarCfg
    c2_ceiling_budget: int
    c2_ceiling_ci: float = Field(gt=0, lt=1)
    display_ci: float = Field(gt=0, lt=1)
    repeat_for_secondary: str
    multiplicity_correction: Literal["none"]
    staging_contrasts: list[str]
    staging_null_ci: float = Field(gt=0, lt=1)
    equivalence_margin_eV: float | None = None     # TOST margin
    equivalence_tost_ci: float | None = None


class HighMomentCfg(_Strict):
    threshold_muB: float = Field(gt=0)
    models: list[str]
    budget: int


class ModelListCfg(_Strict):
    models: list[str]


class CgcnnCheckCfg(_Strict):
    models: list[str]
    budgets: list[int]
    resamples: list[int]
    seeds: list[int]


class EpochCapCfg(_Strict):
    """Logged 2026-10-07: S and D rerun with a larger epoch cap; stage ``capsens``."""
    models: list[str]
    budget: int
    resamples: list[int]
    seeds: list[int]
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    cap_insensitive_if_abs_delta_A_le_eV: float = Field(gt=0)


class UnselectedVariantSweepCfg(_Strict):
    """UnselectedVariantSweepCfg."""
    variant: Literal["D-state", "D-late"]
    budgets: list[int]
    resamples: list[int]
    seeds: list[int]


class SensitivityCfg(_Strict):
    exclude_high_moment: HighMomentCfg
    epoch_cap: EpochCapCfg
    kiyohara_split: ModelListCfg
    cgcnn_check: CgcnnCheckCfg
    unselected_injection_variant: Literal["tuning_runs_only"]
    unselected_variant_sweep: UnselectedVariantSweepCfg


class HparamCrossCfg(_Strict):
    """Item 1 of: S trained at ``budget`` with the tuned values of ``anchor``."""
    model: str
    cells: list[tuple[int, int]]          # (budget, anchor)
    resamples: list[int]
    seeds: list[int]


class DescriptorAblationCfg(_Strict):
    """Item 3 of: D with a descriptor class at its training mean; ``none`` is the full model, retrained so that
    its checkpoint exists for the permutation importances."""
    budget: int
    resamples: list[int]
    seeds: list[int]
    ablations: list[Literal["none", "desc_host", "desc_site"]]
    permutation_repeats: int = Field(gt=0)
    permutation_seed: int


class V2Cfg(_Strict):
    """Backbone design screen. Selection: the variant with the lowest mean over ``screen_anchors`` of the
    seed-mean validation MAE (E_f, eV)."""
    variants: dict[str, dict[str, str | bool]]
    screen_anchors: list[int]
    screen_seeds: list[int]
    selection: Literal["lowest_mean_val_mae_over_anchors"]
    selected: str | None = None           # set by the logged rule from the screen


class DiagnosticsCfg(_Strict):
    """DiagnosticsCfg."""
    hparam_cross: HparamCrossCfg
    descriptor_ablation: DescriptorAblationCfg


class RelaxCfg(_Strict):
    cell: Literal["unit"]
    relax_cell: bool
    relax_positions: bool
    optimiser: Literal["FIRE"]
    fmax_eV_per_A: float = Field(gt=0)
    max_steps: int = Field(gt=0)


class StratifyCfg(_Strict):
    by: Literal["cation_d_electron_count"]
    oxidation_states: Literal["pymatgen_oxi_state_guess"]
    bins: list[str]


class MlipNullCfg(_Strict):
    median_internal_rmsd_A_below: float = Field(gt=0)
    delta_mae_ci_includes_zero: bool


class MlipCfg(_Strict):
    model: str
    size: Literal["medium"]
    dtype: Literal["float64"]
    version: str | TBD
    relax: RelaxCfg
    tiling: Literal["release_supercell_matrix"]
    geometry_components: list[str]
    conditions: list[str]
    retrain: Literal[False]
    models: list[str]
    budgets: list[int | Literal["N_star"]]
    budget_if_no_n_star: int
    D_descriptors: Literal["dft"]
    stratify: StratifyCfg
    null_rule: MlipNullCfg


class ProbeCheckpointsCfg(_Strict):
    model: Literal["S"]
    seeds: list[int]
    resamples: Literal["all"]
    budgets: Literal["all"]


class AlphaCvCfg(_Strict):
    folds: int = Field(gt=1)
    grouped: bool
    fit_on: Literal["training_sites"]


class ProbePassCfg(_Strict):
    trend: str
    beats_controls_min_budget: int
    otherwise: Literal["move_to_si"]


class ProbeCfg(_Strict):
    enabled: bool
    checkpoints: ProbeCheckpointsCfg
    features: Literal["readout_input_vector"]
    regressor: Literal["ridge"]
    alpha_cv: AlphaCvCfg
    targets: Literal["D_descriptors"]
    report: list[str]
    controls: list[str]
    pass_rule: ProbePassCfg


class LocoCfg(_Strict):
    grouping: Literal["s02_definition_ii_a_distinct_group_set"]
    k: int = Field(gt=1)
    training_size: Literal["max_feasible"]
    models: list[str]
    seeds: list[int]
    uncertainty: Literal["host_bootstrap"]


class EnsembleUncertaintyCfg(_Strict):
    spread: str
    calibration: str


class RobustnessCfg(_Strict):
    loco: LocoCfg
    ensemble_uncertainty: EnsembleUncertaintyCfg
    variance_tables: list[str]


class ScreeningCfg(_Strict):
    candidates: list[str]
    all_polymorphs: bool
    min_resamples_tested: int = Field(ge=1)
    model: Literal["S"]
    budget: int
    aggregate: Literal["mean_over_resamples_tested"]


class GatesCfg(_Strict):
    G1: str
    C0_kiyohara_max_mae_eV: float = Field(gt=0)
    G2_tag: str
    cost_gate: str


class PathsCfg(_Strict):
    results_dir: str
    figures_dir: str
    handoff_dir: str


class Config(_Strict):
    data: DataCfg
    split: SplitCfg
    budgets: BudgetsCfg
    curve: CurveCfg
    graph: GraphCfg
    models: ModelsCfg
    tuning: TuningCfg
    training: TrainingCfg
    stats: StatsCfg
    analysis: AnalysisCfg
    secondary_metric: SecondaryMetricCfg = Field(default_factory=SecondaryMetricCfg)
    sensitivity: SensitivityCfg
    diagnostics: DiagnosticsCfg
    v2: V2Cfg
    mlip: MlipCfg
    probe: ProbeCfg
    robustness: RobustnessCfg
    screening: ScreeningCfg
    gates: GatesCfg
    deviations_log: str
    paths: PathsCfg


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML file and return the validated Config."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
