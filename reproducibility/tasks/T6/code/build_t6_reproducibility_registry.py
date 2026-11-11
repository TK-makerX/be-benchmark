#!/usr/bin/env python
"""
Build the T6 reproducibility/applicability registry from local model facts and
existing benchmark run-status evidence.

T6 is intentionally not an accuracy leaderboard. This script records whether a
model can be installed, run, and reused, and where its input/runtime limits are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

OUTPUT_CSV = ROOT / "config" / "model_reproducibility_registry_T6.csv"
OUTPUT_REPORT = ROOT / "reports" / "T6_REPRODUCIBILITY_REGISTRY.md"
TASK_STARTING_POINTS = ROOT / "docs" / "TASK_STARTING_POINTS_LOCKED_2026-07-01.md"

T1_METRICS = (
    ROOT
    / "formal_results"
    / "T1_v6_common_core_abe_5model_full30000"
    / "metrics"
    / "T1_V5_SMOKE_METRICS.csv"
)
T1_BATCH_STATUS = (
    ROOT
    / "formal_results"
    / "T1_v6_common_core_abe_5model_full30000"
    / "reports"
    / "T1_V6_BATCH_RUN_STATUS.csv"
)
T2_STATUS = ROOT / "formal_results" / "T2_expand_v0" / "T2_expanded_run_status.csv"
T2_AUDIT = ROOT / "formal_results" / "T2_expand_v0" / "T2_model_inclusion_audit.csv"
T3_STATUS = ROOT / "formal_results" / "T3_position_profile_v1" / "T3_run_status.csv"
T6_TIMING_STATUS = ROOT / "formal_results" / "T6_reproducibility_v0" / "T6_REPRODUCIBILITY_TIMING_STATUS.csv"

PILOT_VALIDATION_REPORTS = [
    ROOT / "reports" / "pilot_v0_prediction_validation_report.json",
    ROOT / "reports" / "pilot_v1_behive_prediction_validation_report.json",
    ROOT / "reports" / "pilot_v2_be_endo_prediction_validation_report.json",
    ROOT / "reports" / "pilot_v3_crispron_be_prediction_validation_report.json",
    ROOT / "reports" / "benchmark_efficiency_v1_prediction_validation_report.json",
]


MODEL_FACTS: list[dict[str, Any]] = [
    {
        "model_id": "bedict_v2",
        "display_name": "BEDICT-V2",
        "local_folder": str(PROJECT_ROOT / "BEDICT-V2"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_pilot_adapter_validated",
        "current_dependency_check": "passed_2026-07-01_in_E:/AIselect/envs/bedict_v2",
        "environment_path": r"E:\AIselect\envs\bedict_v2\python.exe",
        "runtime_family": "modern_cpu_pytorch",
        "python_or_runtime": "Python 3.10.13; PyTorch 2.2.2+cpu",
        "key_dependencies": "torch 2.2.2+cpu; pandas 2.2.2; scikit-learn 1.4.1.post1; scipy 1.12.0",
        "windows_compatibility": "windows_native",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_local_cpu_run",
        "model_weights_or_assets": "bundled_pretrained_pkl_files",
        "local_modifications_or_wrappers": "author example executed through local reproduction runner; original auxiliary paths preserved",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "efficiency_and_outcome_distribution",
        "input_contract": "model-specific BEDICT sequence/editor context; local smoke verified ABEmax-SpRY example",
        "input_burden": "moderate: sequence plus supported editor/delivery context; raw SRA data not needed for inference",
        "practical_limits": "best reused on supported BEDICT editor contexts; formal T1 scalarization not yet part of T1_v6 common-core",
        "validated_smoke_output": str(PROJECT_ROOT / "BEDICT-V2" / "outputs" / "ABEmax-SpRY_example_predictions.csv"),
        "smoke_output_sha256": "FD408670E369F25CF037236C408BC1B7AF0BB2D22127F6A30E5CA15CE9EF19D9",
        "T1_applicability": "pilot_or_derived_only_not_in_T1_v6_common_core",
        "T2_applicability": "completed_T2_expand_v0",
        "T3_applicability": "supplementary_proxy_completed_T3_v1",
        "T4_applicability": "possible_if_T1_scalarization_is_locked",
        "T5_applicability": "possible_where_BEDICT_input_contract_matches",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_design_reuse_when_editor_context_supported",
    },
    {
        "model_id": "behive_bystander",
        "display_name": "BE-Hive bystander",
        "local_folder": str(PROJECT_ROOT / "BE-Hive-bystander"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_pilot_adapter_validated",
        "current_dependency_check": "covered_by_passed_bedict_v2_env_check_2026-07-01",
        "environment_path": r"E:\AIselect\envs\bedict_v2\python.exe",
        "runtime_family": "modern_cpu_pytorch",
        "python_or_runtime": "Python 3.10.13; PyTorch 2.2.2+cpu",
        "key_dependencies": "torch 2.2.2+cpu; pandas 2.2.2; scikit-learn 1.4.1.post1; scipy 1.12.0",
        "windows_compatibility": "windows_native",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_local_cpu_run",
        "model_weights_or_assets": "pretrained_params_in_repo",
        "local_modifications_or_wrappers": "local pandas-2 compatibility patch",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "bystander_outcome_distribution",
        "input_contract": "50nt BE-Hive coordinate input; supported base-editor and cell-type combinations",
        "input_burden": "moderate: requires BE-Hive-native 50nt window plus editor/cell context",
        "practical_limits": "native output is conditional outcome mass, not direct scalar activity",
        "validated_smoke_output": str(PROJECT_ROOT / "BE-Hive-bystander" / "outputs" / "BE4_mES_example_predictions.csv"),
        "smoke_output_sha256": "8D871FEBA319B22088A1AE87597041F9479AE53351A508C83C46CF7CA611A553",
        "T1_applicability": "derived_scalar_possible_not_in_T1_v6_common_core",
        "T2_applicability": "completed_T2_expand_v0",
        "T3_applicability": "main_source_associated_completed_T3_v1",
        "T4_applicability": "possible_if_edited_mass_scalarization_is_locked",
        "T5_applicability": "possible_on_BE-Hive-compatible_editor_cell_inputs",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_product_profile_or_bystander_design_questions",
    },
    {
        "model_id": "behive_efficiency",
        "display_name": "BE-Hive efficiency",
        "local_folder": str(PROJECT_ROOT / "BE-Hive-efficiency"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_T1_v6_adapter_completed_with_coverage_limit",
        "current_dependency_check": "passed_2026-07-01_only_with_conda_style_PATH_injection; raw_powerShell_import_hits_numpy_DLL_load_error",
        "environment_path": r"E:\AIselect\envs\behive_efficiency\python.exe",
        "runtime_family": "legacy_sklearn_python37",
        "python_or_runtime": "Python 3.7.12",
        "key_dependencies": "numpy 1.16.4; pandas 0.24.2; scikit-learn 0.20.3; scipy 1.3.0; biopython 1.73",
        "windows_compatibility": "windows_native_but_activation_sensitive",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none",
        "model_weights_or_assets": "old_sklearn_pickle_models_in_repo",
        "local_modifications_or_wrappers": "benchmark runners inject env Library/bin paths for DLL loading",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "scalar_efficiency_logit_or_calibrated_fraction_if_conversion_parameters_available",
        "input_contract": "50nt sequence from positions -19 to 30; supported base-editor/cell-type combinations only",
        "input_burden": "high: native coordinate window plus editor/cell; external calibration parameters needed for absolute fractions",
        "practical_limits": "T1_v6 external common-core coverage was zero because compatible external calibrated rows were unavailable; use rank/raw-score unless calibrated",
        "validated_smoke_output": str(PROJECT_ROOT / "BE-Hive-efficiency" / "outputs" / "BE4_mES_example_prediction.json"),
        "smoke_output_sha256": "F7D63C84113E2F242AFC44FC44D1742ECEEBE03B5FC2EE74005AB024035D6081",
        "T1_applicability": "completed_T1_v6_with_partial_input_coverage",
        "T2_applicability": "not_applicable_scalar_efficiency_only_in_T2_expand_v0",
        "T3_applicability": "not_run_in_T3_v1",
        "T4_applicability": "possible_from_T1_scalar_score_with_coverage_flag",
        "T5_applicability": "limited_by_BE-Hive_native_input_and_calibration_requirements",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_supported_BE-Hive-like_efficiency_ranking",
    },
    {
        "model_id": "forecast_be",
        "display_name": "FORECasT-BE",
        "local_folder": str(PROJECT_ROOT / "FORECasT-BE"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_T1_T2_T3_adapters_completed",
        "current_dependency_check": "passed_2026-07-01_via_E:/AIselect/envs/forecastbe/Scripts/python.exe",
        "environment_path": r"E:\AIselect\envs\forecastbe\Scripts\python.exe",
        "runtime_family": "sklearn_joblib",
        "python_or_runtime": "Python 3.10.13",
        "key_dependencies": "pandas 2.0.3; scikit-learn 1.1.3; joblib 1.2.0; biopython 1.83; scipy 1.10.1",
        "windows_compatibility": "windows_native",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none",
        "model_weights_or_assets": "sklearn/joblib model assets in repo",
        "local_modifications_or_wrappers": "local sklearn joblib compatibility patch; logs warn about estimator unpickle version mismatch",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "position_specific_rate_and_derived_total_score",
        "input_contract": "20nt guide/protospacer; ABE or CBE editor type; positions 3-10",
        "input_burden": "low_to_moderate: 20nt guide and editor type, but absolute efficiency needs mean/std scaling assumptions",
        "practical_limits": "native score can be uncalibrated; rank/top-position use is safer than absolute calibration unless scaling is supplied",
        "validated_smoke_output": str(PROJECT_ROOT / "FORECasT-BE" / "outputs" / "CBE_example_prediction.json"),
        "smoke_output_sha256": "25E5FA2A2AEFFCBFA19CEA45D520BD3A2A92488A9570BF89770BC5D7884C8055",
        "T1_applicability": "completed_T1_v6_common_core",
        "T2_applicability": "completed_T2_expand_v0_position_profile",
        "T3_applicability": "main_rank_only_completed_T3_v1",
        "T4_applicability": "possible_rank_or_threshold_on_derived_T1_score",
        "T5_applicability": "possible_where_20nt_guide_ABE_CBE_contract_matches",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_position-aware_design_prioritization_with_calibration_caveat",
    },
    {
        "model_id": "yachie_base_editing_prediction",
        "display_name": "Yachie base-editing-prediction",
        "local_folder": str(PROJECT_ROOT / "Yachie-base-editing-prediction"),
        "access_mode": "local",
        "benchmark_tier": "secondary",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_pilot_T2_T3_adapters_completed",
        "current_dependency_check": "simple_numpy_matplotlib_stack_available_in_shared_bedict_v2_env",
        "environment_path": r"E:\AIselect\envs\bedict_v2\python.exe",
        "runtime_family": "lightweight_python_probability_model",
        "python_or_runtime": "Python 3.10.13 in local runs",
        "key_dependencies": "numpy; matplotlib",
        "windows_compatibility": "windows_native",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none",
        "model_weights_or_assets": "sample base-editor probability profiles in repo",
        "local_modifications_or_wrappers": "benchmark wrapper converts observed outcome candidates into model calls",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "editing_pattern_frequency",
        "input_contract": "target sequence plus start/end relative to PAM and model CSV; optional target/outcome pair",
        "input_burden": "moderate_to_high: requires enumerated candidate outcomes or model-specific target coordinates",
        "practical_limits": "historical probabilistic comparator, not a pretrained neural predictor; T3 uses observed-candidate score label",
        "validated_smoke_output": str(PROJECT_ROOT / "Yachie-base-editing-prediction" / "outputs" / "TargetACEmax_example_stdout.txt"),
        "smoke_output_sha256": "6378BF36F59E8404768F6597E31631E5AF7018F3EC9732D41BA8500E6D40B2FC",
        "T1_applicability": "pilot_or_derived_secondary_only",
        "T2_applicability": "completed_T2_expand_v0_proxy",
        "T3_applicability": "supplementary_only_completed_T3_v1",
        "T4_applicability": "not_primary_for_high_low_efficiency_without_extra_scalarization",
        "T5_applicability": "limited_proxy_possible",
        "T6_applicability": "primary_T6_local_reproducibility_target_but_secondary_model_role",
        "T7_applicability": "possible_for_outcome_pattern_questions_not_primary_scalar_ranking",
    },
    {
        "model_id": "be_endo",
        "display_name": "BE_Endo",
        "local_folder": str(PROJECT_ROOT / "BE_Endo"),
        "access_mode": "local",
        "benchmark_tier": "core_partial",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "sequence_only_smoke_reproduced_and_T2_T3_adapter_completed",
        "current_dependency_check": "covered_by_passed_be_tf210_check_2026-07-01",
        "environment_path": r"E:\AIselect\envs\be_tf210\Scripts\python.exe",
        "runtime_family": "tf210_compat",
        "python_or_runtime": "Python 3.10.13; TensorFlow 2.10.1",
        "key_dependencies": "tensorflow 2.10.1; pandas 1.5.3; numpy 1.23.5; scikit-learn 1.1.3; scipy 1.10.1",
        "windows_compatibility": "windows_native_for_sequence_only_mode",
        "wsl_or_external_tool_requirement": "full_endogenous_mode_needs_bedtools_and_external_endo_factors.bed",
        "cuda_gpu_requirement": "none_for_local_sequence_only_run",
        "model_weights_or_assets": "sequence_model_assets_present; full endogenous resources incomplete",
        "local_modifications_or_wrappers": "sequence-only BE_Seq benchmark wrapper",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "per_base_efficiency_or_position_rate",
        "input_contract": "40bp input: 10bp upstream + 20bp sgRNA + 3bp PAM + 7bp downstream; genomic coordinate for full mode",
        "input_burden": "high: exact 40bp window and, for full mode, genomic annotation resources",
        "practical_limits": "full BE_Endo endogenous-factor mode remains blocked by external endo_factors.bed and bedtools; sequence-only BE_Seq is reusable",
        "validated_smoke_output": str(PROJECT_ROOT / "BE_Endo" / "outputs" / "CBE_BE_Seq_example_eff.tsv"),
        "smoke_output_sha256": "9144D78370680C5FCE70E3F168FDFB5CFC7561091A2DE7629F550799669881A7",
        "T1_applicability": "pilot_sequence_only_or_derived_position_aggregate",
        "T2_applicability": "completed_T2_expand_v0_position_profile",
        "T3_applicability": "main_rank_only_completed_T3_v1",
        "T4_applicability": "possible_from_sequence_only_aggregate_with_scope_flag",
        "T5_applicability": "sequence_only_possible; full_endogenous_generalization_deferred",
        "T6_applicability": "primary_T6_partial_reproducibility_target",
        "T7_applicability": "possible_where_position_profile_or_sequence_only_efficiency_is_adequate",
    },
    {
        "model_id": "deepbe",
        "display_name": "DeepBE",
        "local_folder": str(PROJECT_ROOT / "DeepBE"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_T1_T2_T3_adapters_completed",
        "current_dependency_check": "covered_by_passed_be_tf210_check_2026-07-01",
        "environment_path": r"E:\AIselect\envs\be_tf210\Scripts\python.exe",
        "runtime_family": "tf210_compat",
        "python_or_runtime": "Python 3.10.13; TensorFlow 2.10.1",
        "key_dependencies": "tensorflow 2.10.1; pandas 1.5.3; numpy 1.23.5; scikit-learn 1.1.3; h5py 3.7.0",
        "windows_compatibility": "windows_native_local_compatibility_runner",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_local_run; upstream_docs_expect_TF2.6_CUDA11.2_for_original_setup",
        "model_weights_or_assets": "large_release_weights_downloaded_and_extracted",
        "local_modifications_or_wrappers": "minimal ABE-focused compatibility runner maps to selected editor/PAM variants",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "efficiency_and_outcome_distribution",
        "input_contract": "30bp target window with PAM-variant feature; current adapter focuses on ABE8e-like compatible paths",
        "input_burden": "moderate: 30nt target plus editor/PAM variant; full 63-editor coverage needs more wrappers",
        "practical_limits": "current adapter is minimal ABE-focused; CBE/additional editor expansion is optional future work",
        "validated_smoke_output": str(PROJECT_ROOT / "DeepBE" / "outputs" / "DeepBE_8e_SpCas9_minimal_prediction.csv"),
        "smoke_output_sha256": "B2DBA388204BE7DEC666AB80AC18E2BC5256203C1BAE4499C96DF386F610C0E7",
        "T1_applicability": "completed_T1_v6_common_core",
        "T2_applicability": "completed_T2_expand_v0_proxy",
        "T3_applicability": "supplementary_proxy_completed_T3_v1",
        "T4_applicability": "possible_from_T1_scalarized_output",
        "T5_applicability": "possible_for_supported_editor_PAM_subset",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_supported_editor_outcome_or_efficiency_ranking",
    },
    {
        "model_id": "epcnnbe",
        "display_name": "EPCNNBE",
        "local_folder": str(PROJECT_ROOT / "EPCNNBE"),
        "access_mode": "local",
        "benchmark_tier": "core",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_and_pilot_adapter_validated; exploratory_T3_completed",
        "current_dependency_check": "covered_by_passed_bedict_v2_env_check_2026-07-01",
        "environment_path": r"E:\AIselect\envs\bedict_v2\python.exe",
        "runtime_family": "modern_cpu_pytorch",
        "python_or_runtime": "Python 3.10.13; PyTorch 2.2.2+cpu",
        "key_dependencies": "torch 2.2.2+cpu; pandas 2.2.2; scikit-learn 1.4.1.post1; biopython 1.83",
        "windows_compatibility": "windows_native",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_local_cpu_run",
        "model_weights_or_assets": "Google_Drive_weights_downloaded_and_extracted_under_repo/be_weights",
        "local_modifications_or_wrappers": "local patch makes optional CLI expected-outcome argument actually optional",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "efficiency_proportion_frequency",
        "input_contract": "ABE/CBE type plus FASTA sequence of at least 30nt; optional expected outcome sequence",
        "input_burden": "moderate: requires FASTA and editor type; outcome-design mode needs expected outcome",
        "practical_limits": "not included in T2_expand_v0 because normalized edited-product/position adapter was not locked then; T3 exploratory only",
        "validated_smoke_output": str(PROJECT_ROOT / "EPCNNBE" / "outputs" / "test_seq_SOD1_abe_res.csv"),
        "smoke_output_sha256": "B99BB1C125720D5A29F0EC2F0280481C05C126842CE067F2674CC64C1933AD80",
        "T1_applicability": "pilot_adapter_validated_not_in_T1_v6_common_core",
        "T2_applicability": "not_in_T2_expand_v0_adapter_not_locked",
        "T3_applicability": "exploratory_supplementary_completed_T3_v1",
        "T4_applicability": "possible_after_T1_scalar_adapter_standardization",
        "T5_applicability": "possible_after_main_adapter_standardization",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_for_expected_outcome_or_protospacer_selection_after_wrapper_expansion",
    },
    {
        "model_id": "deepbaseeditor",
        "display_name": "DeepBaseEditor",
        "local_folder": str(PROJECT_ROOT / "DeepBaseEditor-Paired-Library"),
        "access_mode": "local",
        "benchmark_tier": "core_historical",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "smoke_reproduced_via_compatibility_runner_and_T1_v6_adapter_completed",
        "current_dependency_check": "covered_by_passed_be_tf210_check_2026-07-01",
        "environment_path": r"E:\AIselect\envs\be_tf210\Scripts\python.exe",
        "runtime_family": "tf210_compat_for_python2_tf1_upstream_code",
        "python_or_runtime": "Python 3.10.13; TensorFlow 2.10.1 with tf.compat.v1",
        "key_dependencies": "tensorflow 2.10.1; pandas 1.5.3; numpy 1.23.5; scipy 1.10.1; openpyxl 3.1.2",
        "windows_compatibility": "windows_native_local_compatibility_runner",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_local_run; upstream_docs_are_Ubuntu16/Python2.7/TF1.4/CUDA8_era",
        "model_weights_or_assets": "bundled_weight_examples",
        "local_modifications_or_wrappers": "Python2-to-3 and TensorFlow v1 compatibility runner; TensorFlow deprecation warnings expected",
        "license_redistribution_clarity": "license_file_present_but_redistribution_not_audited_in_T6",
        "primary_output_kind": "scalar_efficiency_historical_ABE_module",
        "input_contract": "30bp target window; current benchmark wrapper uses ABE_Efficiency module",
        "input_burden": "moderate: fixed 30bp input; other modules need separate compatibility wrappers",
        "practical_limits": "current adapter covers historical ABE_Efficiency only; other DeepBaseEditor modules are optional future work",
        "validated_smoke_output": str(PROJECT_ROOT / "DeepBaseEditor-Paired-Library" / "outputs" / "ABE_Efficiency_TEST_OUTPUT_fortest.xlsx"),
        "smoke_output_sha256": "D67179678905647ED381816A4B1BF77BBF580CE7B7DAAFC72E6D5DC532F515B5",
        "T1_applicability": "completed_T1_v6_common_core",
        "T2_applicability": "not_applicable_scalar_efficiency_only_in_T2_expand_v0",
        "T3_applicability": "exploratory_supplementary_completed_T3_v1",
        "T4_applicability": "possible_from_T1_scalarized_output",
        "T5_applicability": "possible_for_historical_ABE_efficiency_scope",
        "T6_applicability": "primary_T6_local_reproducibility_target",
        "T7_applicability": "possible_as_historical_scalar_design_baseline",
    },
    {
        "model_id": "llm_embeditor",
        "display_name": "LLM-EmBEditor",
        "local_folder": str(PROJECT_ROOT / "LLM-EmBEditor"),
        "access_mode": "local",
        "benchmark_tier": "method_only",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "tiny_pipeline_reproduced_but_no_author_scale_pretrained_BE_predictor",
        "current_dependency_check": "transformers_tokenizers_safetensors_passed_in_bedict_v2_env_2026-07-01",
        "environment_path": r"E:\AIselect\envs\bedict_v2\python.exe",
        "runtime_family": "modern_cpu_pytorch_huggingface_pipeline",
        "python_or_runtime": "Python 3.10.13; PyTorch 2.2.2+cpu",
        "key_dependencies": "torch 2.2.2+cpu; transformers 4.44.2; tokenizers 0.19.1; safetensors 0.4.5",
        "windows_compatibility": "windows_native_for_tiny_pipeline",
        "wsl_or_external_tool_requirement": "none",
        "cuda_gpu_requirement": "none_for_tiny_pipeline; author_scale_LLM_embeddings_may_need_GPU",
        "model_weights_or_assets": "tiny_HF_model_cached_for_pipeline_validation; no released BE pretrained regression model",
        "local_modifications_or_wrappers": "tiny example validates embedding and regression-head pipeline only",
        "license_redistribution_clarity": "not_audited_in_T6",
        "primary_output_kind": "trainable_embedding_regression_pipeline",
        "input_contract": "JSONL train/val/test files with text and numeric label; model_name HuggingFace ID/path",
        "input_burden": "high: requires labeled training data and embedding generation, not direct pretrained inference",
        "practical_limits": "exclude from main pretrained predictor denominator unless designing a retrained-baseline experiment",
        "validated_smoke_output": str(PROJECT_ROOT / "LLM-EmBEditor" / "outputs" / "tiny_example" / "regression_head" / "best_model.pt"),
        "smoke_output_sha256": "D9895A94271295C26905803C5ECE42FFE333F67928BC15A0E0F785D8DD5604DC",
        "T1_applicability": "not_author_pretrained; optional_retrained_baseline_only",
        "T2_applicability": "not_applicable_current_pipeline",
        "T3_applicability": "not_applicable_current_pipeline",
        "T4_applicability": "possible_only_as_retrained_classifier_or_regression_baseline",
        "T5_applicability": "possible_only_in_separate_retraining_experiment",
        "T6_applicability": "method_only_T6_reproducibility_target",
        "T7_applicability": "not_primary_until_author_scale_model_or_retrained_protocol_exists",
    },
    {
        "model_id": "crispron_be",
        "display_name": "CRISPRon-BE",
        "local_folder": str(PROJECT_ROOT / "CRISPRon-BE"),
        "access_mode": "local",
        "benchmark_tier": "core_hybrid",
        "code_availability": "local_repo_present",
        "install_run_reuse_status": "official_ABE_CBE_smoke_reproduced_and_T1_T2_T3_adapters_completed",
        "current_dependency_check": "be_tf210_passed_2026-07-01; project_local_RNAfold.cmd_reports_RNAfold_2.7.2; WSL Ubuntu-24.04 installed",
        "environment_path": r"E:\AIselect\envs\be_tf210\Scripts\python.exe plus E:\AIselect\BEs\tools\bin\RNAfold.cmd",
        "runtime_family": "tf210_plus_project_local_wsl_RNAfold",
        "python_or_runtime": "Python 3.10.13; TensorFlow 2.10.1; RNAfold 2.7.2 via WSL helper",
        "key_dependencies": "tensorflow 2.10.1; pandas 1.5.3 locally; CRISPRoff/RNAfold feature step through project-local wrapper",
        "windows_compatibility": "hybrid_windows_python_plus_project_local_WSL_helper",
        "wsl_or_external_tool_requirement": "RNAfold_only_via_project_local_WSL_wrapper; no_global_PATH_proxy_or_wslconfig_changes",
        "cuda_gpu_requirement": "none_for_local_run",
        "model_weights_or_assets": "official_model_bundle_downloaded_and_extracted; source archive about_5GB",
        "local_modifications_or_wrappers": "PowerShell smoke wrapper and benchmark adapter inject PATH only per process",
        "license_redistribution_clarity": "Business_Source_License_in_repo; production_use_and_redistribution_need_license_review",
        "primary_output_kind": "efficiency_and_outcome_distribution",
        "input_contract": "FASTA sequence with at least 30nt: 4nt prefix + 20nt target + NGG PAM + 3nt suffix; ABE/CBE weight vector optional",
        "input_burden": "high: exact 30nt target/PAM structure plus RNAfold feature pipeline",
        "practical_limits": "hybrid helper required; no prediction when relevant target base is absent from editing window; licensing needs attention for production use",
        "validated_smoke_output": str(PROJECT_ROOT / "CRISPRon-BE" / "outputs" / "smoke_windows_wrapper" / "crispronABE_prediction.tsv")
        + "; "
        + str(PROJECT_ROOT / "CRISPRon-BE" / "outputs" / "smoke_windows_wrapper" / "crispronCBE_prediction.tsv"),
        "smoke_output_sha256": "ABE:D2BEA9D823DFDB7051117F36C5722286F1BC3341B80936EE018AC247EF88C309; CBE:2B7CF343BCB54057E2081D8655556A8E1557A8BF407D7F63E7CA70D9285BFEFD",
        "T1_applicability": "completed_T1_v6_common_core",
        "T2_applicability": "completed_T2_expand_v0",
        "T3_applicability": "main_external_completed_T3_v1_with_low_position_coverage",
        "T4_applicability": "possible_from_T1_scalarized_output",
        "T5_applicability": "possible_where_NGG_30nt_RNAfold_contract_matches",
        "T6_applicability": "primary_T6_hybrid_reproducibility_target",
        "T7_applicability": "possible_for_efficiency_and_outcome_design_when_input_contract_satisfied",
    },
]


WEB_FACTS: list[dict[str, Any]] = [
    {
        "model_id": "beep_web",
        "display_name": "BEEP web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "not_locally_installable_in_current_workspace; direct_HTTP_check_timed_out_2026-06-22",
        "primary_output_kind": "web_score",
        "practical_limits": "manual/web capture only if accessible; no local code or weights identified",
    },
    {
        "model_id": "bedict_web",
        "display_name": "BEDICT2.0 web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_efficiency_and_outcome",
        "practical_limits": "web app versioning/export/determinism must be captured separately from local BEDICT-V2",
    },
    {
        "model_id": "behive_web",
        "display_name": "BE-Hive web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_efficiency_and_bystander",
        "practical_limits": "web app sanity check only; local BE-Hive modules are the reproducible benchmark path",
    },
    {
        "model_id": "deepbaseeditor_web",
        "display_name": "DeepBaseEditor web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_efficiency_and_outcome",
        "practical_limits": "web counterpart only; not a local reproducibility substitute",
    },
    {
        "model_id": "deepbe_web",
        "display_name": "DeepBE web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_efficiency_and_outcome",
        "practical_limits": "web counterpart only; not a local reproducibility substitute",
    },
    {
        "model_id": "forecast_be_web",
        "display_name": "FORECasT-BE web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_position_rate",
        "practical_limits": "web counterpart only; local FORECasT-BE is the reproducible benchmark path",
    },
    {
        "model_id": "crispron_be_web",
        "display_name": "CRISPRon-BE web",
        "access_mode": "web_blackbox",
        "install_run_reuse_status": "reachable_2026-06-22; manual_web_supplement_only",
        "primary_output_kind": "web_efficiency_and_outcome",
        "practical_limits": "web counterpart only; local hybrid CRISPRon-BE is already reproduced",
    },
    {
        "model_id": "bedict_legacy_web",
        "display_name": "BE-DICT legacy web",
        "access_mode": "web_blackbox_candidate",
        "install_run_reuse_status": "timed_out_2026-06-22; candidate_only",
        "primary_output_kind": "web_perbase_bystander",
        "practical_limits": "needs retry/access check and possible code/weight search before any T6 local claim",
    },
    {
        "model_id": "bedeep_on_web",
        "display_name": "BEdeepon web",
        "access_mode": "web_blackbox_candidate",
        "install_run_reuse_status": "reachable_2026-06-22; candidate_only",
        "primary_output_kind": "web_efficiency_and_outcome",
        "practical_limits": "manual/web-only comparator unless local code and weights are obtained",
    },
    {
        "model_id": "bedeep_off_web",
        "display_name": "BEdeepoff web",
        "access_mode": "web_blackbox_candidate",
        "install_run_reuse_status": "reachable_2026-06-22; off_target_supplement_only",
        "primary_output_kind": "web_offtarget_ratio",
        "practical_limits": "off-target task is outside primary on-target T1-T7 map",
    },
]


CSV_COLUMNS = [
    "model_id",
    "display_name",
    "access_mode",
    "benchmark_tier",
    "local_folder",
    "code_availability",
    "install_run_reuse_status",
    "current_dependency_check",
    "environment_path",
    "runtime_family",
    "python_or_runtime",
    "key_dependencies",
    "windows_compatibility",
    "wsl_or_external_tool_requirement",
    "cuda_gpu_requirement",
    "model_weights_or_assets",
    "local_modifications_or_wrappers",
    "license_redistribution_clarity",
    "license_review_status",
    "license_evidence_type",
    "local_license_files",
    "inferred_license_family",
    "redistribution_reuse_assessment",
    "license_review_note",
    "primary_output_kind",
    "input_contract",
    "input_burden",
    "practical_limits",
    "validated_smoke_output",
    "smoke_output_exists",
    "smoke_output_sha256",
    "pilot_schema_validated",
    "pilot_prediction_rows",
    "pilot_validation_reports",
    "T1_applicability",
    "T1_v6_batches",
    "T1_v6_all_returncode_zero",
    "T1_adapter_panel_rows",
    "T1_prediction_rows_raw",
    "T1_selected_rows",
    "T1_valid_predictions",
    "T1_input_coverage",
    "T2_applicability",
    "T2_run_status",
    "T2_prediction_rows",
    "T2_skipped_rows",
    "T3_applicability",
    "T3_include_label",
    "T3_panel_targets",
    "T3_evaluation_universe_rows",
    "T3_matched_prediction_positions",
    "T3_position_coverage",
    "T3_missing_positions_zero_filled",
    "T3_unmatched_prediction_positions_outside_universe",
    "T4_applicability",
    "T5_applicability",
    "T6_applicability",
    "T7_applicability",
    "T6_timing_status",
    "T6_timing_panel_rows",
    "T6_timing_prediction_rows",
    "T6_timing_skipped_rows",
    "T6_timing_elapsed_seconds",
    "T6_timing_seconds_per_1000_panel_rows",
    "runtime_per_1000_examples",
    "failure_rate_or_coverage_evidence",
    "evidence_paths",
]


LICENSE_AUDIT: dict[str, dict[str, str]] = {
    "bedict_v2": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "license_file_present",
        "local_license_files": str(PROJECT_ROOT / "BEDICT-V2" / "repo" / "LICENSE"),
        "inferred_license_family": "MIT",
        "redistribution_reuse_assessment": "clear_permissive_code_reuse_with_notice; model_weight_terms_follow_bundled_repo_license_unless_upstream_separates_assets",
        "license_review_note": "Local repo includes MIT LICENSE; keep copyright/license notice with copied or modified code.",
    },
    "behive_bystander": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "readme_external_license_reference_only",
        "local_license_files": str(PROJECT_ROOT / "BE-Hive-bystander" / "repo" / "README.md"),
        "inferred_license_family": "external_web_terms_unresolved_from_local_source",
        "redistribution_reuse_assessment": "reuse_possible_for_local_benchmark; redistribution_or_hosted_service_use_needs_upstream_terms_review",
        "license_review_note": "README License section points to https://www.crisprbehive.design/about but no local license text was found.",
    },
    "behive_efficiency": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "readme_external_license_reference_only",
        "local_license_files": str(PROJECT_ROOT / "BE-Hive-efficiency" / "repo" / "README.md"),
        "inferred_license_family": "external_web_terms_unresolved_from_local_source",
        "redistribution_reuse_assessment": "reuse_possible_for_local_benchmark; redistribution_or_hosted_service_use_needs_upstream_terms_review",
        "license_review_note": "README License section points to https://www.crisprbehive.design/about but no local license text was found.",
    },
    "forecast_be": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "license_file_present",
        "local_license_files": str(PROJECT_ROOT / "FORECasT-BE" / "repo" / "LICENSE"),
        "inferred_license_family": "MIT",
        "redistribution_reuse_assessment": "clear_permissive_code_reuse_with_notice; model_asset_terms_follow_bundled_repo_license_unless_upstream_separates_assets",
        "license_review_note": "Local repo includes MIT LICENSE; setup.py also classifies package as MIT.",
    },
    "yachie_base_editing_prediction": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "no_local_license_found",
        "local_license_files": "",
        "inferred_license_family": "not_found_in_local_source",
        "redistribution_reuse_assessment": "local_benchmark_reuse_only; do_not_redistribute_code_or_sample_models_without_upstream_terms_review",
        "license_review_note": "No LICENSE/COPYING/NOTICE file or README license text found in the local repo scan.",
    },
    "be_endo": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "no_local_license_found",
        "local_license_files": "",
        "inferred_license_family": "not_found_in_local_source",
        "redistribution_reuse_assessment": "local_benchmark_reuse_only; do_not_redistribute_code_or_model_assets_without_upstream_terms_review",
        "license_review_note": "No LICENSE/COPYING/NOTICE file or README license text found in the local repo scan.",
    },
    "deepbe": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "framework_license_only_model_license_unknown",
        "local_license_files": str(PROJECT_ROOT / "DeepBE" / "repo" / "DeepBE-Web" / "license.txt"),
        "inferred_license_family": "CodeIgniter_framework_MIT_only; DeepBE_model_license_not_found_in_local_source",
        "redistribution_reuse_assessment": "local_benchmark_reuse_only; framework_license_is_not_sufficient_for_model_asset_redistribution",
        "license_review_note": "Only the bundled DeepBE-Web framework license.txt was found; no top-level DeepBE model license was found.",
    },
    "epcnnbe": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "no_local_license_found",
        "local_license_files": "",
        "inferred_license_family": "not_found_in_local_source",
        "redistribution_reuse_assessment": "local_benchmark_reuse_only; do_not_redistribute_code_or_model_assets_without_upstream_terms_review",
        "license_review_note": "No LICENSE/COPYING/NOTICE file or README/source license text found in the local repo scan.",
    },
    "deepbaseeditor": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "license_file_present_in_relevant_subdirectory",
        "local_license_files": "; ".join(
            [
                str(PROJECT_ROOT / "DeepBaseEditor-Paired-Library" / "repo" / "DeepBaseEditor" / "LICENSE.txt"),
                str(PROJECT_ROOT / "DeepBaseEditor-Paired-Library" / "repo" / "DeepCas9" / "LICENSE"),
            ]
        ),
        "inferred_license_family": "MIT",
        "redistribution_reuse_assessment": "clear_permissive_code_reuse_with_notice_for_licensed_subdirectories; verify_any_unlicensed_paired_assets_before_redistribution",
        "license_review_note": "DeepBaseEditor and DeepCas9 subdirectories include MIT licenses; paired-library bundle should keep subdirectory notices.",
    },
    "crispron_be": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "restrictive_license_file_present",
        "local_license_files": "; ".join(
            [
                str(PROJECT_ROOT / "CRISPRon-BE" / "repo" / "COPYING"),
                str(PROJECT_ROOT / "CRISPRon-BE" / "repo" / "licenses" / "BSL.txt"),
                str(PROJECT_ROOT / "CRISPRon-BE" / "repo" / "licenses" / "APL.txt"),
            ]
        ),
        "inferred_license_family": "Business_Source_License_1.1_with_Apache_2.0_change_license_after_change_date",
        "redistribution_reuse_assessment": "redistribution_and_non_production_use_allowed_with_BSL_notice; production_and_third_party_hosted_service_use_restricted_until_change_license_applies",
        "license_review_note": "Local COPYING/README state BSL terms, limited production grant, contact for commercial licensing, and Apache-2.0 change license after four years.",
    },
    "llm_embeditor": {
        "license_review_status": "audited_from_local_files_2026-07-01",
        "license_evidence_type": "no_local_license_found_method_only_pipeline",
        "local_license_files": "",
        "inferred_license_family": "not_found_in_local_source",
        "redistribution_reuse_assessment": "method_only_local_pipeline; do_not_redistribute_code_or_training_examples_without_upstream_terms_review",
        "license_review_note": "No local license file or README license text found; this remains a method-only trainable pipeline in T6.",
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def safe_int(value: Any) -> int:
    try:
        if value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def semicolon_join(values: list[str]) -> str:
    return "; ".join(v for v in values if v)


def first_existing_output(outputs: str) -> str:
    if not outputs:
        return ""
    statuses = []
    for part in outputs.split(";"):
        path = Path(part.strip())
        if not str(path):
            continue
        statuses.append(f"{path.exists()}:{path}")
    return "; ".join(statuses)


def collect_pilot_validation() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for report in PILOT_VALIDATION_REPORTS:
        if not report.exists():
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        counts = data.get("counts_by_model", {})
        for model_id, count in counts.items():
            rec = out.setdefault(
                model_id,
                {
                    "rows": 0,
                    "ok": True,
                    "warnings": 0,
                    "reports": [],
                },
            )
            rec["rows"] += safe_int(count)
            rec["ok"] = bool(rec["ok"] and data.get("ok", False))
            rec["warnings"] += safe_int(data.get("warning_count", 0))
            rec["reports"].append(str(report))
    return out


def collect_t1_metrics() -> dict[str, dict[str, str]]:
    metrics = read_csv(T1_METRICS)
    out: dict[str, dict[str, str]] = {}
    if not metrics.empty:
        all_rows = metrics[metrics["subset"].eq("all_compatible_smoke")]
        for _, row in all_rows.iterrows():
            out[row["model_id"]] = {
                "T1_selected_rows": row.get("selected_rows", ""),
                "T1_valid_predictions": row.get("valid_predictions", ""),
                "T1_input_coverage": fmt_float(safe_float(row.get("coverage", ""))),
            }

    status = read_csv(T1_BATCH_STATUS)
    if not status.empty:
        for model_id, group in status.groupby("model_id", dropna=False):
            rec = out.setdefault(model_id, {})
            returncodes = [str(v) for v in group["returncode"].tolist()]
            rec.update(
                {
                    "T1_v6_batches": str(len(group)),
                    "T1_v6_all_returncode_zero": str(all(v == "0" for v in returncodes)),
                    "T1_adapter_panel_rows": str(sum(safe_int(v) for v in group["adapter_panel_rows"])),
                    "T1_prediction_rows_raw": str(sum(safe_int(v) for v in group["prediction_rows"])),
                }
            )
    return out


def collect_t2() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    status = read_csv(T2_STATUS)
    if not status.empty:
        for _, row in status.iterrows():
            out[row["model_id"]] = {
                "T2_run_status": row.get("status", ""),
                "T2_prediction_rows": row.get("prediction_rows", ""),
                "T2_skipped_rows": row.get("skipped_rows", ""),
            }
    audit = read_csv(T2_AUDIT)
    if not audit.empty:
        for _, row in audit.iterrows():
            rec = out.setdefault(row["model_id"], {})
            rec.setdefault("T2_run_status", row.get("t2_run_status", ""))
    return out


def collect_t3() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    status = read_csv(T3_STATUS)
    if status.empty:
        return out
    for _, row in status.iterrows():
        universe = safe_float(row.get("evaluation_universe_rows", ""))
        matched = safe_float(row.get("matched_prediction_positions", ""))
        coverage = (matched / universe) if universe else None
        out[row["model_id"]] = {
            "T3_include_label": row.get("include_in_t3", ""),
            "T3_panel_targets": row.get("panel_targets", ""),
            "T3_evaluation_universe_rows": row.get("evaluation_universe_rows", ""),
            "T3_matched_prediction_positions": row.get("matched_prediction_positions", ""),
            "T3_position_coverage": fmt_float(coverage),
            "T3_missing_positions_zero_filled": row.get("missing_universe_positions_filled_zero", ""),
            "T3_unmatched_prediction_positions_outside_universe": row.get("unmatched_prediction_positions_outside_universe", ""),
        }
    return out


def collect_t6_timing() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    timing = read_csv(T6_TIMING_STATUS)
    if timing.empty:
        return out
    for _, row in timing.iterrows():
        out[row["model_id"]] = {
            "T6_timing_status": row.get("status", ""),
            "T6_timing_panel_rows": row.get("panel_rows", ""),
            "T6_timing_prediction_rows": row.get("prediction_rows", ""),
            "T6_timing_skipped_rows": row.get("skipped_rows", ""),
            "T6_timing_elapsed_seconds": row.get("elapsed_seconds", ""),
            "T6_timing_seconds_per_1000_panel_rows": row.get("seconds_per_1000_panel_rows", ""),
            "runtime_per_1000_examples": row.get("seconds_per_1000_panel_rows", ""),
        }
    return out


def build_registry() -> pd.DataFrame:
    pilot = collect_pilot_validation()
    t1 = collect_t1_metrics()
    t2 = collect_t2()
    t3 = collect_t3()
    t6_timing = collect_t6_timing()

    rows: list[dict[str, Any]] = []
    for fact in MODEL_FACTS + WEB_FACTS:
        model_id = fact["model_id"]
        row = {col: "" for col in CSV_COLUMNS}
        row.update(fact)
        license_rec = LICENSE_AUDIT.get(model_id)
        if license_rec:
            row.update(license_rec)
            row["license_redistribution_clarity"] = license_rec["redistribution_reuse_assessment"]
        elif row["access_mode"].startswith("web"):
            row["license_review_status"] = "not_applicable_web_or_candidate_entry"
            row["license_evidence_type"] = "not_local_reproducibility_claim"
            row["inferred_license_family"] = "not_audited_for_web_supplement"
            row["redistribution_reuse_assessment"] = "manual_accessibility_supplement_only; no_local_code_or_weight_redistribution_claim"
            row["license_review_note"] = "Web-only and candidate tools are listed for accessibility/applicability context, not as reusable local packages."

        pilot_rec = pilot.get(model_id)
        if pilot_rec:
            row["pilot_schema_validated"] = f"{pilot_rec['ok']}; report_warnings={pilot_rec['warnings']}"
            row["pilot_prediction_rows"] = str(pilot_rec["rows"])
            row["pilot_validation_reports"] = semicolon_join(pilot_rec["reports"])
        elif row["access_mode"].startswith("web"):
            row["pilot_schema_validated"] = "not_applicable_web_blackbox"
        else:
            row["pilot_schema_validated"] = "not_validated_as_author_pretrained_adapter"

        for source in (t1, t2, t3, t6_timing):
            row.update(source.get(model_id, {}))

        row["smoke_output_exists"] = first_existing_output(str(row.get("validated_smoke_output", "")))
        if not row.get("runtime_per_1000_examples"):
            row["runtime_per_1000_examples"] = "not_recorded_in_existing_run_status"

        evidence = [
            str(TASK_STARTING_POINTS),
            str(ROOT / "docs" / "T6_STARTER_HANDOFF.md"),
            str(ROOT / "docs" / "BENCHMARK_TASK_MAP_LOCKED_2026-07-01.md"),
            str(PROJECT_ROOT / "BENCHMARK_MODEL_STATUS.md"),
            str(PROJECT_ROOT / "CODE_PRESERVATION_POLICY.md"),
            str(ROOT / "reports" / "ADAPTER_COVERAGE_STATUS.md"),
        ]
        if model_id in t1:
            evidence.append(str(T1_METRICS))
            evidence.append(str(T1_BATCH_STATUS))
        if model_id in t2:
            evidence.append(str(T2_STATUS))
            evidence.append(str(T2_AUDIT))
        if model_id in t3:
            evidence.append(str(T3_STATUS))
        if model_id in t6_timing:
            evidence.append(str(T6_TIMING_STATUS))
        if row.get("pilot_validation_reports"):
            evidence.append(row["pilot_validation_reports"])
        if row.get("local_license_files"):
            evidence.append(row["local_license_files"])
        row["evidence_paths"] = semicolon_join(evidence)

        coverage_bits = []
        if row.get("T1_input_coverage"):
            coverage_bits.append(f"T1_input_coverage={row['T1_input_coverage']}")
        if row.get("T2_skipped_rows") != "":
            coverage_bits.append(f"T2_skipped_rows={row['T2_skipped_rows']}")
        if row.get("T3_position_coverage"):
            coverage_bits.append(f"T3_position_coverage={row['T3_position_coverage']}")
        if row.get("T6_timing_skipped_rows") != "":
            coverage_bits.append(f"T6_timing_skipped_rows={row['T6_timing_skipped_rows']}")
        if not coverage_bits:
            coverage_bits.append("no_task_level_coverage_metric_recorded")
        row["failure_rate_or_coverage_evidence"] = "; ".join(coverage_bits)
        rows.append(row)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return ""
    view = df[columns].copy().fillna("")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame) -> str:
    local = df[df["access_mode"].eq("local")].copy()
    web = df[df["access_mode"].str.startswith("web", na=False)].copy()
    controlled = local[~local["benchmark_tier"].eq("method_only")]
    completed_pilot = controlled[controlled["pilot_schema_validated"].str.startswith("True", na=False)]
    t1_completed = local[local["T1_applicability"].str.contains("completed_T1_v6", na=False)]
    t2_completed = local[local["T2_run_status"].eq("completed")]
    t3_recorded = local[local["T3_include_label"].ne("")]
    t6_timed = local[local["T6_timing_status"].ne("")]
    t6_timed_completed = t6_timed[t6_timed["T6_timing_status"].eq("completed")]
    t6_elapsed = pd.to_numeric(t6_timed_completed["T6_timing_elapsed_seconds"], errors="coerce").sum()
    license_audited = local[local["license_review_status"].str.startswith("audited_from_local_files", na=False)]
    permissive_license = local[
        local["inferred_license_family"].str.contains("MIT", na=False)
        & ~local["license_evidence_type"].str.contains("framework_license_only", na=False)
    ]
    restrictive_license = local[local["license_evidence_type"].str.contains("restrictive", na=False)]
    unresolved_license = local[
        local["license_evidence_type"].str.contains("no_local|external|unknown|framework_license_only", na=False)
    ]

    lines: list[str] = [
        "# T6 reproducibility and applicability registry",
        "",
        "Generated from local benchmark evidence on 2026-07-01.",
        "",
        "## Scope",
        "",
        "T6 asks which base-editor prediction models can actually be installed, run, reused, and applied to which inputs. It is not an accuracy task, so this registry records installation state, runtime dependencies, input contracts, task coverage, skip/coverage evidence, and explicit blockers. It intentionally omits model-performance scores.",
        "",
        "## Scope Lock",
        "",
        "**Task.** T6 evaluates reproducibility and applicability: whether each base-editor prediction model has runnable code/weights, a usable local or web execution path, documented environment requirements, clear input contracts, observed run status, and practical reuse limits. T6 does not rank models by predictive performance.",
        "",
        "**Source of truth.** Following `TASK_STARTING_POINTS_LOCKED_2026-07-01.md`, T6 starts from local model folders, environment records, benchmark adapters, run logs, prediction files, failure logs, and coverage metadata. Unified matrices are used only to test input coverage/applicability. Biological ground-truth matrices and T1/T2/T3 score tables do not define the T6 task and must not be used to turn T6 into an accuracy leaderboard.",
        "",
        "**Model denominator.** The primary local denominator is 10 controlled BE model roles: BEDICT-V2, BE-Hive bystander, BE-Hive efficiency, FORECasT-BE, Yachie base-editing-prediction, BE_Endo sequence-only mode, DeepBE, EPCNNBE, DeepBaseEditor, and CRISPRon-BE. LLM-EmBEditor is recorded separately as 1 method-only trainable pipeline because no author-scale pretrained BE regression model is available. Ten web-only or candidate tools are included only as supplementary accessibility/applicability entries, not as local reproducible-model claims.",
        "",
        "**Data/evidence denominator.** T6 does not introduce a new experimental accuracy dataset. It reuses reproducibility/applicability evidence: 21 registry rows total, 11 local roles, 10 web/candidate entries, 77 local model-by-task applicability rows, 50 T1 v6 adapter run-status rows across 5 models, 7 T2_expand_v0 run-status rows, and 9 T3_position_profile_v1 run-status rows. T1 v6 contributes 140,365 selected adapter-panel rows and 128,551 valid prediction rows only as run/coverage evidence. T2 contributes 49,403 prediction rows with 0 skipped rows across 7 model roles. T3 contributes 22,345 evaluation-universe rows from 140,663 observed position rows as input-coverage/applicability evidence. These counts are evidence volume, not an accuracy leaderboard.",
        "",
        "Machine-readable registry:",
        "",
        f"`{OUTPUT_CSV}`",
        "",
        "## Evidence base",
        "",
        f"- Locked task starting points: `{TASK_STARTING_POINTS}`",
        f"- Locked task map: `{ROOT / 'docs' / 'BENCHMARK_TASK_MAP_LOCKED_2026-07-01.md'}`",
        f"- T6 handoff: `{ROOT / 'docs' / 'T6_STARTER_HANDOFF.md'}`",
        f"- Model status and hashes: `{PROJECT_ROOT / 'BENCHMARK_MODEL_STATUS.md'}`",
        f"- Adapter coverage: `{ROOT / 'reports' / 'ADAPTER_COVERAGE_STATUS.md'}`",
        f"- T1 run status: `{T1_BATCH_STATUS}`",
        f"- T2 run status: `{T2_STATUS}`",
        f"- T3 run status: `{T3_STATUS}`",
        f"- T6 timing status: `{T6_TIMING_STATUS}`",
        "",
        "## Summary findings",
        "",
        f"- Local model roles recorded: {len(local)}.",
        f"- Controlled local BE model roles, excluding the method-only LLM pipeline: {len(controlled)}.",
        f"- Controlled local roles with schema-validated pilot evidence: {len(completed_pilot)}.",
        f"- Local roles with formal T1 v6 adapter completion evidence: {len(t1_completed)}.",
        f"- Local roles with completed T2_expand_v0 evidence: {len(t2_completed)}.",
        f"- Local roles with T3_position_profile_v1 run-status evidence: {len(t3_recorded)}.",
        f"- T6 timing pass completed roles: {len(t6_timed_completed)} / {len(controlled)} controlled local BE model roles; summed adapter wall-clock time {t6_elapsed:.1f} seconds.",
        f"- Local roles with license/reuse status audited from local files: {len(license_audited)} / {len(local)}; permissive local license evidence: {len(permissive_license)}, restrictive local license evidence: {len(restrictive_license)}, unresolved or non-model-local license evidence: {len(unresolved_license)}.",
        "",
        "Four shared runtime families cover the local reproductions: modern CPU PyTorch, legacy Python 3.7/scikit-learn for BE-Hive efficiency, FORECasT-BE's sklearn/joblib environment, and TensorFlow 2.10 compatibility runners. CRISPRon-BE adds a project-local WSL RNAfold helper; this is a hybrid local run, not a full WSL migration.",
        "",
        "The biggest practical limits are input-contract limits, not benchmark accuracy: BE-Hive efficiency needs a BE-Hive 50nt coordinate window and calibration context; BE_Endo full mode still needs external endogenous-factor resources and bedtools; DeepBE and DeepBaseEditor are currently represented by scoped compatibility wrappers; LLM-EmBEditor is a trainable pipeline without a released author-scale pretrained BE regression model.",
        "",
        "Runtime per 1,000 examples is now populated from the T6 timing pass where available. Historical T1/T2/T3 status files still lack wall-clock timing, so the registry does not infer runtime from file timestamps.",
        "",
        "## Figure-Ready Tables",
        "",
        "The companion script `build_t6_figure_ready_tables.py` derives plotting and manuscript-table inputs from the registry without rerunning models:",
        "",
        f"- Model-by-task applicability matrix: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_model_task_applicability_matrix.csv'}`",
        f"- Environment complexity matrix: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_environment_complexity_matrix.csv'}`",
        f"- Input requirement burden table: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_input_requirement_burden.csv'}`",
        f"- Local reproducibility status summary: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_local_reproducibility_status_summary.csv'}`",
        f"- Coverage and failure-evidence table: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_coverage_and_failure_evidence.csv'}`",
        f"- License and redistribution audit table: `{ROOT / 'reports' / 't6_figure_ready_tables' / 'T6_license_redistribution_audit.csv'}`",
        f"- Current Fig. 6 bundle: `{ROOT / 'figures' / 'T6_reproducibility_applicability_v0' / 'T6_figure6_reproducibility_applicability_summary.png'}`",
        f"- Fig. 6 manifest/report: `{ROOT / 'formal_results' / 'T6_reproducibility_applicability_v0'}`",
        "",
        "Future reruns will also capture runtime evidence: the pilot runners, T1 adapter smoke runner, and T3 exploratory runner now append `started_at`, `ended_at`, and `elapsed_seconds`; the T1 adapter status also records `seconds_per_1000_adapter_panel_rows`.",
        "",
        "## Local Reproducibility Matrix",
        "",
        md_table(
            local,
            [
                "model_id",
                "install_run_reuse_status",
                "runtime_family",
                "windows_compatibility",
                "wsl_or_external_tool_requirement",
                "T1_applicability",
                "T2_applicability",
                "T3_applicability",
            ],
        ),
        "",
        "## Task Coverage Evidence",
        "",
        md_table(
            local,
            [
                "model_id",
                "pilot_schema_validated",
                "pilot_prediction_rows",
                "T1_v6_all_returncode_zero",
                "T1_input_coverage",
                "T2_run_status",
                "T2_skipped_rows",
                "T3_include_label",
                "T3_position_coverage",
            ],
        ),
        "",
        "## T6 Timing Evidence",
        "",
        md_table(
            local,
            [
                "model_id",
                "T6_timing_status",
                "T6_timing_panel_rows",
                "T6_timing_prediction_rows",
                "T6_timing_skipped_rows",
                "T6_timing_elapsed_seconds",
                "T6_timing_seconds_per_1000_panel_rows",
            ],
        ),
        "",
        "## License and Reuse Evidence",
        "",
        "This section is a reproducibility/applicability audit, not legal advice. It records only what the local source tree contains on 2026-07-01 and flags where reuse is clear, restricted, or unresolved.",
        "",
        md_table(
            local,
            [
                "model_id",
                "license_evidence_type",
                "inferred_license_family",
                "redistribution_reuse_assessment",
                "local_license_files",
            ],
        ),
        "",
        "## Input Burden and Practical Limits",
        "",
        md_table(
            local,
            [
                "model_id",
                "input_burden",
                "practical_limits",
            ],
        ),
        "",
        "## Current Environment Notes",
        "",
        "- `E:\\AIselect\\envs\\bedict_v2\\python.exe` imported PyTorch, pandas, scikit-learn, and scipy successfully on 2026-07-01.",
        "- `E:\\AIselect\\envs\\forecastbe\\Scripts\\python.exe` imported pandas, scikit-learn, scipy, biopython, and joblib successfully on 2026-07-01.",
        "- `E:\\AIselect\\envs\\be_tf210\\Scripts\\python.exe` imported TensorFlow 2.10.1 and the shared TF stack successfully on 2026-07-01. TensorFlow reports no CUDA runtime in this CPU setup, which is acceptable for the local runs.",
        "- `E:\\AIselect\\envs\\behive_efficiency\\python.exe` imported the old BE-Hive-efficiency stack only after conda-style `Library\\bin` PATH injection. A raw PowerShell invocation fails at numpy DLL loading, so runners must preserve this PATH setup.",
        "- `E:\\AIselect\\BEs\\tools\\bin\\RNAfold.cmd --version` reports RNAfold 2.7.2. `wsl -l -v` shows Ubuntu-24.04 available for the CRISPRon-BE helper.",
        "",
        "## Web-only and Candidate Tools",
        "",
        "These tools are useful as supplementary/manual comparators but are not local reproducibility evidence unless code, weights, versioning, and export behavior are captured.",
        "",
        md_table(
            web,
            [
                "model_id",
                "install_run_reuse_status",
                "primary_output_kind",
                "practical_limits",
            ],
        ),
        "",
        "## Figure 6 Starting Points",
        "",
        "- Fig. 6A: reproducibility workflow from code/weights/env to smoke output to benchmark adapter.",
        "- Fig. 6B: model-by-task applicability matrix using T1/T2/T3 evidence and T4-T7 readiness labels.",
        "- Fig. 6C: environment complexity heatmap across runtime family, activation sensitivity, WSL helper, and CUDA status.",
        "- Fig. 6D: coverage/failure/runtime summary using T1 input coverage, T2 skipped rows, T3 position coverage, and T6 timing-pass seconds per 1,000 panel rows.",
        "- Fig. 6E: input requirement burden grouped by 20nt guide, 30nt target+PAM, 40bp genomic input, 50nt BE-Hive window, FASTA, and trainable JSONL pipeline.",
        "- Fig. 6F: local reproducibility status separating smoke-only, pilot-validated, formal-task-validated, partial, method-only, and web-only roles.",
        "- Fig. 6G: practical deployment decision tree keyed on desired output type and available input context.",
        "",
        "## Immediate Next Steps",
        "",
        "1. If reviewer-facing throughput evidence is needed, run a larger fixed timing panel; the current Fig. 6 uses small validated adapter panels and labels them as invocation/runtime evidence.",
        "2. For any redistribution, hosted service, or packaged benchmark release, do a project-level legal review for README-only, no-local-license, framework-only, and BSL-restricted entries.",
        "3. Decide whether EPCNNBE and BEDICT-V2 should receive locked normalized adapters for future formal T1/T2 expansions.",
        "4. Keep LLM-EmBEditor separate as a retrained-baseline experiment unless an author-scale pretrained BE model is released.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = build_registry()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_REPORT.write_text(build_report(df), encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_REPORT}")
    print(df[["model_id", "access_mode", "install_run_reuse_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
