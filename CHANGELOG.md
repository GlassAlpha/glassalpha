# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **E6.5: Demographic Shift Simulator** (P0 Feature - Distribution Robustness Testing)

  - **Post-stratification reweighting**: Simulates demographic distribution changes via sample weights
    - Tests model robustness under realistic population shifts
    - Adjusts group proportions by specified percentage points (e.g., +10pp, -5pp)
    - Mathematically sound reweighting: `weight[group] *= (p_target / p_original)`
    - Validates shifted proportion stays within [0.01, 0.99] bounds
  - **Before/after metrics comparison**: Recomputes all metrics with adjusted weights
    - **Fairness**: TPR, FPR, demographic parity, precision, recall
    - **Calibration**: ECE, Brier score (with bootstrap CIs)
    - **Performance**: Accuracy, F1, precision, recall
  - **Degradation detection**: Quantifies metric changes under demographic shifts
    - Reports absolute differences (percentage points)
    - Supports configurable degradation thresholds for CI gates
    - Gate statuses: PASS (no violations), WARNING (approaching threshold), FAIL (exceeds threshold)
  - **CLI integration**: `--check-shift` and `--fail-on-degradation` flags
    - Syntax: `--check-shift attribute:shift` (e.g., `gender:+0.1` for +10pp increase)
    - Multiple shifts supported: `--check-shift gender:+0.1 --check-shift age:-0.05`
    - Exit code 1 if degradation exceeds threshold (CI-friendly)
  - **JSON export**: Complete shift analysis results in sidecar file
    - Exports: shift_specification, baseline_metrics, shifted_metrics, degradation, gate_status, violations
    - File: `{output}.shift_analysis.json`
    - Programmatically parseable for dashboards and monitoring
  - **Deterministic execution**: Fully seeded for reproducibility
    - Seeded weight computation
    - Seeded metric recomputation
    - Byte-identical results across runs
  - **Validation**:
    - Binary attribute requirement (multi-class deferred to enterprise)
    - Feasibility checks (prevent impossible shifts)
    - Clear error messages with exact format requirements
  - Configuration: CLI-only feature (no YAML config required)
  - API: `run_shift_analysis()`, `ShiftAnalysisResult`, `ShiftSpecification`, `compute_shifted_weights()`
  - Test coverage: 30+ contract tests + German Credit integration (parsing, validation, reweighting, metrics, determinism)
  - Module: `glassalpha.metrics.shift.{reweighting, runner}`
  - Documentation: [Shift Testing Guide](site/docs/guides/shift-testing.md) with CI/CD integration examples
  - **Why critical**: Regulatory stress testing (EU AI Act), production readiness validation, CI/CD deployment gates

- **E8: SR 11-7 Compliance Mapping Document** (P1 Feature - Banking Regulatory Guidance)

  - **Comprehensive clause-to-artifact mapping**: Maps all SR 11-7 sections to specific GlassAlpha features
    - Section III.A (Documentation, Testing, Monitoring) → Audit PDF, Shift Testing, CI/CD
    - Section III.B (Validation, Sensitivity, Outcomes) → Calibration, Perturbations, Recourse
    - Section III.C (Assumptions, Data Quality, Limitations) → Manifests, Dataset Bias, Model Card
    - Section IV (Independence, Evidence) → Reproducibility, Evidence Packs
    - Section V (Fairness) → Group, Intersectional, Individual Fairness
  - **Examiner Q&A examples**: Ready-to-use responses for common audit questions
  - **Citation templates**: How to reference GlassAlpha artifacts in compliance documentation
  - **Quick reference table**: One-page summary of SR 11-7 coverage
  - **Validation workflows**: Commands demonstrating reproducibility and independence
  - **Evidence pack structure**: Complete audit bundle for regulatory submission
  - **Sample size guidance**: Statistical power requirements per protected group
  - **Monitoring integration**: CI/CD examples for ongoing model risk management
  - Documentation: [SR 11-7 Mapping](site/docs/compliance/sr-11-7-mapping.md)
  - **Why critical**: Enables banking institutions to demonstrate SR 11-7 compliance, supports regulatory audits

- **E9: README Positioning Refresh** (P1 Feature - Distribution & Adoption)

  - **Pain-first messaging**: Opens with "Ever tried explaining your ML model to a regulator?"
  - **Policy-as-code emphasis**: Shows YAML-based compliance rules, not dashboards
  - **Feature highlights**: Comprehensive listing of completed E-series features
    - Compliance & Fairness: E5, E5.1, E10, E11, E12
    - Explainability & Outcomes: E2, E2.5
    - Robustness & Stability: E6, E6+, E6.5, E10+
    - Regulatory Compliance: SR 11-7, Evidence Packs, Reproducibility
  - **Statistical rigor positioning**: "95% confidence intervals on everything"
  - **Byte-identical reproducibility**: SHA256-verified evidence packs
  - **CI/CD deployment gates**: Shift testing examples with `--fail-on-degradation`
  - **Three-part differentiation**: Policy-as-code, reproducibility, statistical confidence
  - **Why critical**: Improves conversion from repo visitors to users, positions against dashboards/SaaS tools

- **1.2C: QuickStart CLI Generator** (P0 Feature - Onboarding & Adoption)

  - **Complete project scaffolding**: `glassalpha quickstart` generates ready-to-run audit projects
    - Directory structure: data/, models/, reports/, configs/
    - Audit configuration file (audit_config.yaml) with sensible defaults
    - Example run script (run_audit.py) demonstrating programmatic API
    - Project README with next steps and advanced usage examples
    - .gitignore tailored for GlassAlpha projects
  - **Interactive mode**: Guided prompts for dataset and model selection
    - Dataset choices: German Credit (1K samples), Adult Income (48K samples)
    - Model choices: XGBoost (recommended), LightGBM (fast), Logistic Regression (simple)
    - Customizable project name
  - **Non-interactive mode**: Command-line flags for CI/CD automation
    - `--dataset`, `--model`, `--output` flags
    - `--no-interactive` for scripted project generation
  - **Time-to-first-audit**: <60 seconds from install to generated audit report
    - `cd my-audit-project && python run_audit.py`
    - No manual config editing required for built-in datasets
  - **Documentation included**: Every generated project includes README with:
    - Quick start (3 commands to first audit)
    - CI/CD integration examples
    - Custom data setup instructions
    - Links to user guides and documentation
  - CLI: `glassalpha quickstart` (interactive), `glassalpha quickstart --dataset german_credit --model xgboost --no-interactive`
  - **Why critical**: Eliminates onboarding friction, enables immediate value demonstration

- **E6+: Adversarial Perturbation Sweeps** (P2 Feature - Robustness Verification)

  - **Epsilon-perturbation stability testing**: Validates model robustness under small input changes
    - Perturbs non-protected features by ε ∈ {0.01, 0.05, 0.1} (1%, 5%, 10% Gaussian noise)
    - Measures max prediction delta (L∞ norm) across all samples
    - Reports robustness score = max delta across epsilon values
  - **Protected feature exclusion**: Never perturbs gender, race, or other sensitive attributes
    - Prevents synthetic bias introduction during robustness testing
    - Validates that all features marked protected are excluded
  - **Gate logic**: Automatic PASS/FAIL/WARNING based on configurable threshold
    - PASS: max_delta < threshold
    - WARNING: threshold ≤ max_delta < 1.5 × threshold
    - FAIL: max_delta ≥ 1.5 × threshold
    - Default threshold: 0.15 (15% max prediction change)
  - **Deterministic perturbations**: Fully seeded for byte-identical reproducibility
    - Uses seeded Gaussian noise generation
    - Stable sort for epsilon values
  - **JSON export**: All results serializable for programmatic access
    - Exports: robustness_score, max_delta, per_epsilon_deltas, gate_status, threshold
  - **Conditional execution**: Only runs when `metrics.stability.enabled = true`
  - **Automatic integration**: Runs after fairness/calibration in audit pipeline
  - Configuration: `metrics.stability.enabled`, `metrics.stability.epsilon_values`, `metrics.stability.threshold`
  - CLI: Perturbation results included when stability metrics enabled
  - API: `run_perturbation_sweep()`, `PerturbationResult`
  - Test coverage: 22 contract tests + German Credit integration (determinism, epsilon validation, gates, protected features)
  - Module: `glassalpha.metrics.stability.perturbation`
  - **Why critical**: Emerging regulator demand for robustness proofs (EU AI Act, NIST AI RMF), prevents adversarial failures

- **E12: Dataset-Level Bias Audit** (P1 Feature - Root-Cause Bias Detection)

  - **Proxy Correlation Detection**: Identifies non-protected features correlating with protected attributes
    - Multi-level severity system: ERROR (|r|>0.5), WARNING (0.3<|r|≤0.5), INFO (|r|≤0.3)
    - Pearson correlation for continuous-continuous pairs
    - Point-biserial for continuous-categorical pairs
    - Cramér's V for categorical-categorical pairs
    - Flags potential indirect discrimination pathways
  - **Distribution Drift Analysis**: Detects feature distribution shifts between train and test
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features
    - Identifies data quality issues and distribution mismatch
  - **Statistical Power for Sampling Bias**: Power calculations for detecting undersampling
    - Severity levels: ERROR (power<0.5), WARNING (0.5≤power<0.7), OK (power≥0.7)
    - Detects insufficient sample sizes before model training
    - Prevents false confidence in fairness metrics
  - **Continuous Attribute Binning**: Configurable binning strategies for age and other continuous protected attributes
    - Domain-specific bins (age: [18, 25, 35, 50, 65, 100])
    - Custom bins (user-specified)
    - Equal-width and equal-frequency strategies
    - Binning recorded in manifest for reproducibility
  - **Train/Test Split Imbalance Detection**: Chi-square tests for protected group distribution differences
    - Flags biased data splitting (e.g., gender representation differs between splits)
    - Prevents evaluation bias from imbalanced splits
  - **Automatic Integration**: All 5 checks run before model evaluation in audit pipeline
  - **JSON Export**: All results serializable for programmatic access
  - **Deterministic**: Fully seeded for byte-identical reproducibility
  - CLI: Dataset bias metrics included in standard audit output
  - API: `compute_dataset_bias_metrics()`, `compute_proxy_correlations()`, `compute_distribution_drift()`, `compute_sampling_bias_power()`, `detect_split_imbalance()`, `bin_continuous_attribute()`
  - Test coverage: 27 contract tests + 6 integration tests with German Credit dataset
  - Module: `glassalpha.metrics.fairness.dataset`
  - **Why critical**: Catches bias at the source (most unfairness originates in data, not models)

- **E5.1: Basic Intersectional Fairness** (P1 Feature - Sophistication Signal)

  - **Two-way intersectional analysis**: Detects bias at intersections of protected attributes (e.g., gender×race, age×income)
  - **Cartesian product group creation**: Automatically generates all intersectional groups from attribute combinations
  - **Full fairness metrics for each intersectional group**: TPR, FPR, precision, recall, selection rate computed per intersection
  - **Disparity metrics**: Max-min differences and ratios across intersectional groups
  - **Sample size warnings**: Reuses E10 infrastructure (n<10 ERROR, 10≤n<30 WARNING, n≥30 OK)
  - **Bootstrap confidence intervals**: 95% CIs for all intersectional metrics using E10 bootstrap infrastructure
  - **Statistical power analysis**: Power calculations for detecting disparity in each intersectional group
  - **Deterministic computation**: Fully seeded for byte-identical reproducibility
  - **Automatic integration**: Runs alongside group fairness when `data.intersections` specified in config
  - **JSON export**: All intersectional results serializable with nested structure
  - Configuration: `data.intersections: ["gender*race", "age*income"]` in YAML
  - CLI: Intersectional metrics included when intersections configured
  - API: `compute_intersectional_fairness()`, `create_intersectional_groups()`, `parse_intersection_spec()`
  - Test coverage: 23 tests covering parsing, group creation, metrics, CIs, sample warnings, determinism
  - Module: `glassalpha.metrics.fairness.intersectional`
  - Example: German Credit config updated with age×foreign_worker and gender×age intersections

- **E11: Individual Fairness Metrics** (P0 Feature - Legal Risk Coverage)

  - **Consistency Score**: Lipschitz-like metric measuring prediction stability for similar individuals
    - Configurable distance metrics (Euclidean, Mahalanobis)
    - Similarity threshold based on percentile of pairwise distances (default 90th)
    - Reports max and mean prediction differences for similar pairs
  - **Matched Pairs Report**: Identifies individuals with similar features but different predictions
    - Flags potential disparate treatment cases
    - Exports matched pairs with feature distance and prediction difference
    - Checks if protected attributes differ between matched pairs
  - **Counterfactual Flip Test**: Tests if protected attribute changes affect predictions
    - Detects disparate treatment at individual level
    - Computes disparate treatment rate across dataset
    - Supports multi-class protected attributes
  - **Automatic Integration**: Runs alongside group fairness metrics in audit pipeline
  - **JSON Export**: All results serializable for programmatic access and reporting
  - **Deterministic**: Fully seeded for byte-identical reproducibility
  - **Performance**: Pairwise distance computation optimized with vectorization
  - CLI: Individual fairness included in standard audit output
  - API: `IndividualFairnessMetrics`, `compute_consistency_score()`, `find_matched_pairs()`, `counterfactual_flip_test()`
  - Test coverage: 20+ contract tests covering determinism, edge cases, and integration
  - Module: `glassalpha.metrics.fairness.individual`

- **E10+: Calibration Confidence Intervals** (P1 Feature - Statistical Rigor for Calibration)

  - **Bootstrap CIs for ECE and Brier score**: 95% confidence intervals using deterministic bootstrap
    - Reuses E10's bootstrap infrastructure for consistency
    - Percentile method for CI computation (default: 1000 bootstrap samples)
    - Standard error calculation for uncertainty quantification
  - **Bin-wise CIs for calibration curve**: Error bars for observed frequency in each calibration bin
    - Enables visualization of uncertainty in calibration curves
    - Skips bins with <10 samples (insufficient for reliable bootstrap)
    - Exports bin-wise lower/upper bounds for plotting
  - **Backward compatible API**: Legacy `assess_calibration_quality()` still returns dict without CIs
    - New parameter: `compute_confidence_intervals=True` enables CIs
    - Returns `CalibrationResult` dataclass with optional CI fields
  - **Deterministic computation**: Fully seeded for byte-identical reproducibility
  - **JSON export**: All CIs serializable with `to_dict()` for programmatic access
  - **Automatic integration**: Ready for audit pipeline integration (deferred to follow-on work)
  - CLI: Calibration CIs available via API (PDF display deferred)
  - API: `compute_calibration_with_ci()`, `compute_bin_wise_ci()`, `assess_calibration_quality(compute_confidence_intervals=True)`
  - Module: `glassalpha.metrics.calibration.confidence`, `glassalpha.metrics.calibration.quality`
  - Test coverage: 25+ contract tests + German Credit integration tests
  - **Why P1**: Completes statistical rigor story (E10 covered fairness, E10+ covers calibration)

- **E10: Statistical Confidence & Uncertainty for Fairness Metrics** (P0 Feature)
  - Bootstrap confidence intervals for all fairness metrics (TPR, FPR, precision, recall, demographic parity)
  - Deterministic bootstrap with seeded random sampling for byte-identical reproducibility
  - Sample size adequacy checks (n<10 ERROR, 10≤n<30 WARNING, n≥30 OK)
  - Statistical power calculations for detecting disparity
  - Automatic integration with fairness runner and audit pipeline
  - Confidence interval data exported in JSON format for programmatic access
  - CLI: All fairness metrics now include 95% confidence intervals by default
  - API: `run_fairness_metrics()` accepts `compute_confidence_intervals`, `n_bootstrap`, `confidence_level`, `seed`
  - Test coverage: 21 comprehensive tests validating determinism, accuracy, and edge cases

### Changed

- **Config Schema**: Added `data.intersections` field for E5.1 intersectional fairness analysis
  - Validates intersection format: must be `"attr1*attr2"` for two-way intersections
  - Normalizes to lowercase and strips whitespace
  - Example: `intersections: ["gender*race", "age_years*foreign_worker"]`
- **Fairness Metrics Runner**: Enhanced to compute bootstrap CIs, sample size warnings, power analysis, and intersectional metrics
  - New parameter: `intersections` list for specifying intersectional analysis
  - Automatically computes intersectional fairness when intersections specified
  - Results nested under `intersectional` key in output
- **Audit Pipeline**:
  - Now passes seed to fairness runner for deterministic confidence intervals
  - Now passes intersections config to fairness runner for E5.1 analysis
  - Now computes E11 individual fairness metrics alongside group fairness
  - Passes full feature DataFrame to fairness metrics for individual-level analysis
- **AuditResults**: Fairness analysis now includes `sample_size_warnings`, `statistical_power`, `{metric}_ci`, and `individual_fairness` keys
- **`_compute_fairness_metrics()`**: Signature updated to accept full feature DataFrame for individual fairness computation

### Fixed

- None

### Performance

- Bootstrap CIs add ~2-3 seconds per fairness metric (with n_bootstrap=1000, n=200)
- No memory overhead (bootstrap samples not retained)
- Determinism has zero performance cost (uses existing seeding infrastructure)

### Security

- None

### Deprecated

- None

### Removed

- None

## Previous Releases

See Git history for previous changes.
