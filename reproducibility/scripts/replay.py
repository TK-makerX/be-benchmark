"""Replay frozen benchmark analyses in an isolated output tree."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = "benchmark_core"
REV = "SCI-data/返修"
T1_V8 = CORE + "/formal_results/T1_v8_label_blind_known_crispron_clean"
T3_V1 = CORE + "/formal_results/T3_position_profile_v1"
T4_V21 = CORE + "/formal_results/T4_high_low_classification_v2_1_context_unit_ci_revision"
T7_V2 = CORE + "/formal_results/T7_unique_design_recommendation_v2"
MATRIX = CORE + "/data/unified_matrix_v5"
CRISPR = "CRISPRon-BE/repo/data_for_model_development_V1.0"


def baseline(name, executable, options, checks, mounts=(), copies=()):
    return {"name": name, "mounts": mounts, "copies": copies,
            "commands": [(f"tasks/{name}/code/{executable}", options)], "checks": checks}


def revision(name, folder, executable, options, mounts, filenames):
    return {"name": name, "mounts": [f"{REV}/{folder}", *mounts], "copies": [],
            "commands": [(f"tasks/revision/{folder}/{executable}", options)],
            "checks": [(f"{{R}}/{folder}/{f}", f"tasks/revision/{folder}/{f}") for f in filenames]}


def cases():
    t1 = baseline("T1", "replay_scalar.py", ["--output-dir", "{O}"],
                  [("{O}/"+f, "tasks/T1/metrics/"+f) for f in (
                      "PRIMARY_ASSOCIATION_METRICS.csv", "METADATA_STRATIFIED_ASSOCIATION.csv",
                      "PRIMARY_DESIGN_CLUSTER_BOOTSTRAP_95CI.csv", "PAIRED_SPEARMAN_DIFFERENCE_BOOTSTRAP.csv")],
                  [T1_V8+"/data", T1_V8+"/metrics", T1_V8+"/predictions/harmonized"])
    t1["commands"].append(("tasks/T1/code/replay_calibration_v81.py",
                           ["--v8-root", f"{{P}}/{T1_V8}", "--output-dir", "{O}/v8_1"]))
    t1["checks"] += [("{O}/v8_1/metrics/"+f, "tasks/T1/expected/v8_1/"+f) for f in
                     ("CRISPRON_ABSOLUTE_CALIBRATION_SUMMARY.csv", "CRISPRON_PREDICTED_EFFICIENCY_DECILE_CALIBRATION.csv")]

    t2 = baseline("T2", "score_t2_expanded.py", ["--matrix-dir", "{B}/tasks/T2/inputs", "--runs-dir",
                   "{B}/tasks/T2/predictions", "--output-dir", "{O}"],
                  [("{O}/"+f, "tasks/T2/expected/"+f) for f in
                   ("T2_expanded_outcome_metrics.csv", "T2_expanded_position_metrics.csv")],
                  [CORE+"/config/model_registry.csv"])
    t3 = baseline("T3", "unused", [], [("{O}/results/T3_corrected_metrics.csv",
                   "tasks/revision/02_R2.1/denominator_correction/results/T3_corrected_metrics.csv")],
                  [REV+"/02_R2.1/execution_eligibility", REV+"/02_R2.1/overlap_audit/CRISPRon_actual_backend_by_target.csv"],
                  [("tasks/T3/expected/initial/"+f, "{P}/SCI-data/Fig.4/source_data/T3_position_profile_v1/"+f) for f in
                   ("T3_position_joined_universe_predictions.csv","T3_position_metrics.csv")] +
                  [("tasks/revision/02_R2.1/denominator_correction/results/FORECasT_empty_output_replay.json",
                    "{O}/results/FORECasT_empty_output_replay.json")])
    project = ["--project-root", "{P}", "--revision-root", "{R}"]
    rev = ["--revision-root", "{R}"]
    t3["commands"] = [("tasks/revision/02_R2.1/denominator_correction/code/correct_t3_denominators.py",
                       project + ["--output-dir", "{O}"])]
    t4 = baseline("T4","verify_t4_fig5_v2_1_statistics.py", ["--root", f"{{P}}/{T4_V21}"], [],
                  [T4_V21+"/source_data",T4_V21+"/tables"])
    t5 = baseline("T5","replay_joint_matching.py",["--output-dir","{O}"],
                  [("{O}/"+f,"tasks/T5/tables/"+f) for f in (
                      "T5_REFERENCE_TO_SHIFT_CONTRASTS_V1_1.csv","T5_T1_JOINT_MATCHING_ELIGIBILITY_V1_1.csv",
                      "T5_T1_JOINT_MATCHED_DUPLICATE_AUDIT_V1_1.csv")])
    t6 = baseline("T6","build_t6_figure_ready_tables.py",[],
                  [(f"{{P}}/{CORE}/reports/t6_figure_ready_tables/"+f,"tasks/T6/expected/"+f) for f in (
                      "T6_model_task_applicability_matrix.csv", "T6_coverage_and_failure_evidence.csv")],
                  [CORE+"/config/model_reproducibility_registry_T6.csv"],
                  [("tasks/T6/code/build_t6_figure_ready_tables.py",f"{{P}}/{CORE}/scripts/build_t6_figure_ready_tables.py")])
    t6["commands"] = [(f"{{P}}/{CORE}/scripts/build_t6_figure_ready_tables.py", [])]
    t7 = baseline("T7","replay_recommendation.py",["--output-dir","{O}"], [
        ("{O}/"+f, "tasks/T7/expected/"+f) for f in ("T7_unique_design_model_group_scores.csv",
         "T7_unique_design_random_group_scores.csv", "T7_unique_design_random_metric_distribution.csv",
         "T7_unique_design_oracle_group_scores.csv")])

    def r(name, folder, executable, options, mounts, outputs):
        return revision(name, folder, executable, options, mounts, outputs)

    return {spec["name"]:spec for spec in [
        t1, t2, t3, t4, t5, t6, t7,
        r("neighbors","02_R2.1","overlap_audit/audit_sequence_neighbors.py",project,
          [T1_V8+"/data", CRISPR],
          ["overlap_audit/CRISPRon_guide_neighbor_summary.csv", "overlap_audit/T1_neighborhood_exclusion_associations.csv"]),
        r("cross-source","02_R2.1","external_validation/analyze_external.py",rev,[],
          ["external_validation/results/external_association_metrics.csv",
           "external_validation/results/external_paired_association_differences.csv"]),
        r("external-endpoint","02_R2.1","endpoint_correction/audit_head_and_prepare_external.py",project,[CRISPR],
          ["endpoint_correction/CRISPRon_training_head_endpoint_identity_by_source.csv",
           "endpoint_correction/Marquart_ABE_compatible_inference_panel.csv"]),
        r("common","03_R2.2","code/analyze_common_panels.py",project,
          [CORE+"/formal_results/T2_expand_v0",REV+"/02_R2.1/denominator_correction/data"],
          ["results/common_position_metrics.csv","results/common_product_metrics.csv",
           "results/common_position_paired_differences.csv","results/common_product_paired_differences.csv"]),
        r("random","04_R2.3","code/score_random_profiles.py",rev,[REV+"/03_R2.2/code"],
          ["results/random_position_metrics.csv","results/random_product_metrics.csv",
           "results/random_position_paired_differences.csv","results/random_product_paired_differences.csv"]),
        r("calibration","06_R2.5","code/crossfit_assay_calibration.py",project,[T1_V8+"/data"],
          ["results/calibration_metrics.csv","results/calibration_bins.csv","results/fold_fit_parameters.csv"]),
        r("missing","07_R2.6","code/summarize_missing_policies.py",rev,
          [REV+"/02_R2.1/denominator_correction/results"],
          ["results/equal_target_missing_policy_sensitivity.csv"]),
        r("shortlist","08_R2.7","code/analyze_shortlist_baselines.py",project,
          [T7_V2, MATRIX+"/design_master_v5.csv",REV+"/09_R2.8"],
          ["results/shortlist_summary.csv","results/shortlist_per_group.csv","results/shortlist_paired_contrasts.csv"]),
        r("paired","09_R2.8","code/complete_paired_metrics.py",project,[T7_V2+"/T7_unique_design_model_group_scores.csv"],
          ["results/model_paired_comparison.csv", "results/shared_guide_subsampling_summary.csv"]),
        r("source-audit","10_R2.9","code/audit_source_heterogeneity.py",project,
          [MATRIX+"/source_inventory_v5.csv",MATRIX+"/observation_matrix_v5.csv"],
          ["results/catalog_source_measurement_inventory.csv", "results/numerical_scale_audit.csv"]),
        r("scalar-followups","14_R3.1","code/analyze_scalar_followups.py",rev,[REV+"/06_R2.5/data"],
          ["results/all_model_oof_calibration_bins.csv"]),
        r("substitution","17_R3.3","code/score_substitutions.py",project,[T1_V8+"/data"],
          ["results/substitution_sensitivity.csv","results/paired_sensitivity_differences.csv","results/execution_coverage.csv"]),
        r("shuffle","20_R3.6","code/score_shuffles.py",project,[T4_V21+"/source_data"],
          ["results/shuffle_discrimination.csv","results/original_score_replay.csv","results/execution_coverage.csv"]),
    ]}


def transfer(source, target):
    if not source.is_file():
        raise FileNotFoundError(f"Missing input {source}; fetch its asset from assets.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def check_table(actual, frozen):
    a = pd.read_csv(actual, low_memory=False)
    b = pd.read_csv(frozen, low_memory=False)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9, atol=1e-10)
    return len(a)


def run(case, work):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    project = work / "project"
    env = os.environ.copy()
    env.update(BE_REPLAY_PROJECT=str(project), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    mapping = {"B":str(ROOT),"P":str(project),"R":str(project/REV),"O":str(work/"output")}
    with (ROOT / "payload_manifest.csv").open(encoding="utf-8", newline="") as handle:
        aliases = list(csv.DictReader(handle))
    for row in aliases:
        alias = row["project_alias"]
        if alias and any(alias == p or alias.startswith(p+"/") for p in case["mounts"]):
            transfer(ROOT / row["path"], project / alias)
    for src, dst in case["copies"]:
        transfer(ROOT / src, Path(dst.format_map(mapping)))

    for executable, options in case["commands"]:
        script = ROOT / executable.format_map(mapping)
        command = [sys.executable, str(script), *[arg.format_map(mapping) for arg in options]]
        print(f"Running {script.name}", flush=True)
        subprocess.run(command, check=True, cwd=work, env=env)
    checks = []
    for output, reference in case["checks"]:
        actual = Path(output.format_map(mapping))
        try:
            count = check_table(actual, ROOT / reference)
        except (ValueError, AssertionError, OSError) as error:
            raise ValueError(f"Frozen table mismatch for {reference}: {error}") from error
        checks.append({"table": reference, "rows": count})
    report = {"task":case["name"],"status":"PASS","tables_recomputed_and_compared":checks}
    (work / "replay_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    plans = cases()
    parser.add_argument("task", nargs="?", choices=sorted(plans))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(plans))
        return
    if not args.task or not args.work_dir:
        parser.error("Provide a task and a new output directory")
    run(plans[args.task], args.work_dir)


if __name__ == "__main__":
    main()

