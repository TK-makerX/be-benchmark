"""Reuse frozen CRISPRon T2/T3 predictions after known-development exclusions.

Only output inside the authorized audit folder. No inference, model fitting,
plotting, original manuscript changes or original result promotion.
"""
from __future__ import annotations

import os
import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("BE_REPLAY_PROJECT", "."))
CORE = ROOT / "benchmark_core"
OUT = Path(__file__).resolve().parent
assert OUT == (ROOT / "SCI-data/返修/02_R2.1/overlap_audit").resolve()
N_BOOT = 5000
SEED = 20261025
MIN_CI_TARGETS = 20  # Explicit pragmatic reporting guard, not a power calculation.
SCOPE = "editor_class_matched_development__forward__any"
sources = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register(path, role):
    p = Path(path).resolve()
    sources[str(p)] = {"path": str(p), "role": role, "sha256_before": digest(p), "bytes": p.stat().st_size}
    return p


def read(path, role, **kwargs):
    return pd.read_csv(register(path, role), **kwargs)


def write(name, frame):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def fasta(path):
    register(path, "frozen native backend input FASTA")
    result, seq_id = {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(">"):
            seq_id = line[1:].strip()
            assert seq_id not in result
            result[seq_id] = ""
        elif line.strip():
            assert seq_id
            result[seq_id] += line.strip().upper()
    return result


def verify_backend(task, panel):
    run_dir = CORE / "runs" / ("T2_expand_v0" if task == "T2" else "T3_position_profile_v1") / "crispron_be"
    register(run_dir / "raw_t2_crispron_be_manifest.csv", "frozen adapter raw-output manifest")
    raw_position = read(run_dir / "predictions_t2_crispron_be_positions.csv", "frozen harmonized position predictions", dtype=str).fillna("")
    try:
        skipped = read(run_dir / "skipped_t2_crispron_be.csv", "frozen explicit adapter skip reasons", dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        skipped = pd.DataFrame(columns=["target_profile_id", "reason"])
    rows = []
    for backend_class in ["ABE", "CBE"]:
        backend = run_dir / backend_class.lower()
        if not backend.exists():
            continue
        sequences = fasta(backend / "30mers.fa")
        log_path = register(backend / "crispron_be.log", "historical invocation log with actual ABE/CBE argument")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        commands = [line for line in log.splitlines() if "DeepCRISPRonBE_eval.py" in line and f" {backend_class} " in line]
        assert len(commands) == 1, (task, backend_class, "missing or ambiguous logged backend command")
        native = read(backend / f"crispron{backend_class}_prediction.tsv", "frozen native backend outcome predictions", sep="\t", dtype=str).fillna("")
        native_ids = set(native.seq_id)
        assert native_ids <= set(sequences)
        for target_id, seq in sequences.items():
            p = panel.loc[panel.target_profile_id.eq(target_id)]
            assert len(p) == 1
            p = p.iloc[0]
            assert p.sequence_context_30nt == seq
            observed_backend_labels = sorted(set(raw_position.loc[raw_position.target_profile_id.eq(target_id), "model_editor"]) - {""})
            assert not observed_backend_labels or observed_backend_labels == [f"CRISPRon-{backend_class}"]
            assert p.editor_class == backend_class, "Cannot use experimental-class audit as actual backend-class exclusion"
            rows.append({"task": task, "target_profile_id": target_id, "experimental_editor": p.editor, "experimental_editor_class": p.editor_class,
                "actual_backend_class": backend_class, "backend_class_equals_experimental_class": True,
                "backend_executed": True, "adapter_skip_reason": "",
                "input_30nt_matches_native_fasta": True, "backend_fasta_file": str(backend / "30mers.fa"),
                "native_prediction_file": str(backend / f"crispron{backend_class}_prediction.tsv"), "native_prediction_present": target_id in native_ids,
                "native_outcome_rows": int(native.seq_id.eq(target_id).sum()), "harmonized_position_rows_before_observed_join": int(raw_position.target_profile_id.eq(target_id).sum()),
                "harmonized_model_editor_values": ";".join(observed_backend_labels), "logged_command": commands[0],
                "backend_weight_directory_from_code": str(ROOT / "CRISPRon-BE/repo/data/CRISPRonBE_models" / backend_class),
                "weight_directory_evidence": "logged editor argument plus DeepCRISPRonBE_eval.py predict_X loads path_to_model/editor/model*; historical weight-file hashes not asserted",
            })
    executed_ids = {r["target_profile_id"] for r in rows}
    skipped_ids = set(skipped.target_profile_id) if "target_profile_id" in skipped else set()
    assert not (executed_ids & skipped_ids), "Explicit skip and native execution overlap requires manual investigation"
    assert executed_ids | skipped_ids == set(panel.target_profile_id)
    assert set(raw_position.target_profile_id) <= executed_ids
    for _, p in panel[panel.target_profile_id.isin(skipped_ids)].iterrows():
        reason = ";".join(sorted(set(skipped.loc[skipped.target_profile_id.eq(p.target_profile_id), "reason"])))
        rows.append({"task": task, "target_profile_id": p.target_profile_id, "experimental_editor": p.editor, "experimental_editor_class": p.editor_class,
            "actual_backend_class": "not_executed", "backend_class_equals_experimental_class": "not_applicable", "backend_executed": False,
            "adapter_skip_reason": reason, "input_30nt_matches_native_fasta": "not_in_native_fasta", "backend_fasta_file": "", "native_prediction_file": "",
            "native_prediction_present": False, "native_outcome_rows": 0, "harmonized_position_rows_before_observed_join": 0,
            "harmonized_model_editor_values": "", "logged_command": "not_called", "backend_weight_directory_from_code": "not_called",
            "weight_directory_evidence": "explicit adapter skip table; absent from both native FASTA files and harmonized predictions"})
    assert len(rows) == len(panel)
    assert {r["target_profile_id"] for r in rows} == set(panel.target_profile_id)
    return rows


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(np.corrcoef(pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy())[0, 1])


def as_number(value):
    if value in ["", None]:
        return np.nan
    return float(value)


def t2_metrics(data, products, reference):
    target_rhos, tops = [], []
    for _, sub in data.groupby("target_profile_id"):
        sp = reference["corr"](sub["_pred"], sub["_obs"], rank=True)
        if sp is not None:
            target_rhos.append(sp)
        obs_top = sub.sort_values(["_obs", "_pos"], ascending=[False, True]).iloc[0]["_pos"]
        pred_top = sub.sort_values(["_pred", "_pos"], ascending=[False, True]).iloc[0]["_pos"]
        tops.append(obs_top == pred_top)
    out = {"matched_position_rows": len(data), "matched_targets": data.target_profile_id.nunique(),
        "pooled_spearman": reference["corr"](data["_pred"], data["_obs"], rank=True),
        "pooled_pearson": reference["corr"](data["_pred"], data["_obs"]),
        "mae": reference["mae"](data["_pred"], data["_obs"]), "rmse": reference["rmse"](data["_pred"], data["_obs"]),
        "mean_per_target_spearman": np.mean(target_rhos) if target_rhos else np.nan,
        "top1_position_accuracy": np.mean(tops) if tops else np.nan,
        "product_matched_targets": len(products), "mean_jsd": pd.to_numeric(products.jsd, errors="coerce").mean(),
        "median_jsd": pd.to_numeric(products.jsd, errors="coerce").median(),
        "mean_observed_product_spearman": pd.to_numeric(products.observed_product_spearman, errors="coerce").mean(),
        "top1_product_accuracy": products.top1_product_match.astype(str).str.lower().eq("true").mean(),
        "mean_predicted_mass_on_observed_products": pd.to_numeric(products.predicted_mass_on_observed_products, errors="coerce").mean(),
        "n_defined_target_position_spearman": len(target_rhos), "n_defined_target_product_spearman": pd.to_numeric(products.observed_product_spearman, errors="coerce").notna().sum(),
    }
    return out


def check_replication(task, calculated, expected, cols, checks):
    for col in cols:
        observed, target = as_number(calculated[col]), as_number(expected[col])
        difference = abs(observed - target)
        passed = bool((np.isnan(observed) and np.isnan(target)) or difference < 1e-9)
        checks.append({"task": task, "metric": col, "recomputed_full_panel": observed, "frozen_formal_value": target, "absolute_difference": difference, "passed": passed})
        assert passed, (task, col, observed, target)


def main():
    started = datetime.now(timezone.utc).isoformat()
    source_code_paths = [
        ROOT / "SCI-data/Fig.3/code/score_t2_expanded.py",
        ROOT / "SCI-data/Fig.4/code/score_t3_position_profile.py",
        ROOT / "SCI-data/Fig.4/code/t3_coverage_sensitivity/analyze_t3_coverage_sensitivity.py",
        CORE / "adapters/run_t2_crispron_be.py", CORE / "adapters/run_crispron_be_pilot.py",
        ROOT / "CRISPRon-BE/repo/bin/DeepCRISPRonBE_eval.py",
    ]
    for path in source_code_paths:
        register(path, "reviewed reference scoring or backend-routing code; not run as pipeline")
    # The two reviewed modules have main guards. Import their pure metric helpers;
    # do not invoke main, scoring I/O, inference adapters or publishing scripts.
    t2ref = runpy.run_path(str(source_code_paths[0]), run_name="reference_metric_functions")
    t3ref = runpy.run_path(str(source_code_paths[1]), run_name="reference_metric_functions")
    audit = read(OUT / "record_overlap_audit.csv", "completed exact-overlap audit", dtype=str).fillna("")
    register(OUT / "audit_execution_manifest.json", "completed audit contract and source-hash validation")
    memberships, backend_rows, panel_summary = [], [], []
    panels = {}
    for task, path in [("T2", CORE / "data/processed/T2_bystander_matrix/panels/T2_expand_crispron_be.csv"), ("T3", CORE / "data/processed/T3_position_profile_matrix/panels_v1/T3_v1_crispron_be.csv")]:
        panel = read(path, f"frozen CRISPRon {task} input panel", dtype=str).fillna("")
        this_backend = verify_backend(task, panel)
        backend_rows += this_backend
        by_id = {row["target_profile_id"]: row for row in this_backend}
        a = audit[audit.task.eq(task) & audit.record_id.isin(panel.target_profile_id)].copy()
        assert len(a) == len(panel)
        a["backend_executed"] = a.record_id.map(lambda x: by_id[x]["backend_executed"])
        a["actual_backend_class"] = a.record_id.map(lambda x: by_id[x]["actual_backend_class"])
        a["adapter_skip_reason"] = a.record_id.map(lambda x: by_id[x]["adapter_skip_reason"])
        a["experimental_class_development_overlap"] = a[SCOPE].str.lower().eq("true")
        a["known_actual_backend_development_overlap"] = a.backend_executed & a.experimental_class_development_overlap
        a["retained_executed_known_nonoverlap"] = a.backend_executed & ~a.known_actual_backend_development_overlap
        a["sensitivity_membership"] = np.where(~a.backend_executed, "ineligible_adapter_skip", np.where(a.known_actual_backend_development_overlap, "excluded_known_executed_backend_development_sequence", "retained_executed_no_match_in_known_backend_development_lists"))
        memberships.append(a)
        panels[task] = a
        for kept, name in [(False, "excluded"), (True, "retained")]:
            subset = a[a.retained_executed_known_nonoverlap] if kept else a[a.known_actual_backend_development_overlap]
            write(f"CRISPRon_{task}_{name}_target_manifest.csv", subset)
        write(f"CRISPRon_{task}_adapter_skipped_target_manifest.csv", a[~a.backend_executed])
        panel_summary.append({"task": task, "original_input_targets": len(a), "excluded_known_backend_development_targets": int(a.known_actual_backend_development_overlap.sum()),
            "executed_backend_targets": int(a.backend_executed.sum()), "ineligible_adapter_skipped_targets": int((~a.backend_executed).sum()),
            "retained_targets": int(a.retained_executed_known_nonoverlap.sum()), "native_design_ids": "not_encoded_in_T2_T3_manifests",
            "retained_sequence_clusters_guide20_context30": a.loc[a.retained_executed_known_nonoverlap, "sequence_cluster_id"].nunique(),
            "editor_backend_mapping": "executed targets verified against native FASTA/output/log; explicit skips excluded before overlap-sensitive performance", "exclusion_rule": "first remove explicit adapter skips; then same-orientation exact guide20 OR context30 match in actual backend-class author train OR test lists"})
    write("CRISPRon_actual_backend_by_target.csv", pd.DataFrame(backend_rows))
    write("CRISPRon_sensitivity_target_membership.csv", pd.concat(memberships, ignore_index=True))
    write("CRISPRon_exclusion_flow.csv", pd.DataFrame(panel_summary))

    f2 = CORE / "formal_results/T2_expand_v0"
    f3 = CORE / "formal_results/T3_position_profile_v1"
    t2 = read(f2 / "T2_expanded_position_joined_predictions.csv", "frozen formal T2 joined position predictions", dtype=str).fillna("")
    t2 = t2[t2.model_id.eq("crispron_be")].copy()
    t2["_pred"], t2["_obs"] = pd.to_numeric(t2["_pred"]), pd.to_numeric(t2["_obs"])
    p2 = read(f2 / "T2_expanded_outcome_per_target_metrics.csv", "frozen formal T2 per-target product metrics", dtype=str).fillna("")
    p2 = p2[p2.model_id.eq("crispron_be")].copy()
    m2pos = read(f2 / "T2_expanded_position_metrics.csv", "formal T2 baseline validation").query("model_id == 'crispron_be'").iloc[0]
    m2prod = read(f2 / "T2_expanded_outcome_metrics.csv", "formal T2 product baseline validation").query("model_id == 'crispron_be'").iloc[0]
    assert set(t2.target_profile_id) == set(panels["T2"].record_id) == set(p2.target_profile_id)
    checks, point_rows = [], []
    for subset, ids in [("full_original_panel", set(panels["T2"].record_id)), ("known_backend_development_overlap", set(panels["T2"].loc[panels["T2"].known_actual_backend_development_overlap, "record_id"])), ("retained_known_nonoverlap", set(panels["T2"].loc[panels["T2"].retained_executed_known_nonoverlap, "record_id"]))]:
        sub, products = t2[t2.target_profile_id.isin(ids)], p2[p2.target_profile_id.isin(ids)]
        metrics = t2_metrics(sub, products, t2ref)
        point_rows.append({"task": "T2", "subset": subset, "policy": "original_returned_position_and_original_product_metrics", "input_targets": len(ids), **metrics,
            "interpretation": "descriptive exclusion sensitivity; different subset composition; not independent external validation", "ci_status": "not_estimated_for_T2_clean_8_targets; no stable superiority inference"})
        if subset == "full_original_panel":
            check_replication("T2_positions", metrics, m2pos, ["matched_position_rows", "matched_targets", "pooled_spearman", "pooled_pearson", "mae", "rmse", "mean_per_target_spearman", "top1_position_accuracy"], checks)
            check_replication("T2_products", metrics, m2prod, ["mean_jsd", "median_jsd", "mean_observed_product_spearman", "top1_product_accuracy", "mean_predicted_mass_on_observed_products"], checks)
        if subset == "retained_known_nonoverlap":
            write("CRISPRon_T2_retained_position_rows.csv", sub)
            write("CRISPRon_T2_retained_product_metrics.csv", products.drop(columns=["source_relation_to_model"], errors="ignore"))

    t3 = read(f3 / "T3_position_joined_universe_predictions.csv", "frozen formal T3 joined evaluation universe", low_memory=False)
    t3 = t3[t3.model_id.eq("crispron_be")].copy()
    t3["has_model_prediction"] = t3.has_model_prediction.astype(str).str.lower().eq("true")
    m3 = read(f3 / "T3_position_metrics.csv", "formal T3 baseline validation").query("model_id == 'crispron_be'")
    assert set(t3.target_profile_id) == set(panels["T3"].record_id)
    clean_ids = set(panels["T3"].loc[panels["T3"].retained_executed_known_nonoverlap, "record_id"])
    executed_ids = set(panels["T3"].loc[panels["T3"].backend_executed, "record_id"])
    clean3 = t3[t3.target_profile_id.isin(clean_ids)].copy()
    for subset, ids in [("full_original_panel", set(panels["T3"].record_id)), ("execution_eligible_before_overlap", executed_ids), ("known_backend_development_overlap", executed_ids - clean_ids), ("retained_known_nonoverlap", clean_ids)]:
        whole = t3[t3.target_profile_id.isin(ids)].copy()
        for policy in ["all_observed_zero_fill", "returned_only"]:
            sub = whole if policy == "all_observed_zero_fill" else whole[whole.has_model_prediction].copy()
            if policy == "returned_only":
                sub["_pred_filled"] = sub.predicted_value_raw
            computed, per_target = t3ref["score_models"](sub, m3)
            metric = computed.iloc[0].to_dict()
            original_keys = ["evaluation_universe_rows", "evaluation_targets", "matched_prediction_positions", "prediction_coverage_fraction", "pooled_spearman", "pooled_pearson", "mean_per_target_spearman", "median_per_target_spearman", "mae", "rmse", "mean_top1_position_accuracy", "mean_top3_position_recovery"]
            point_rows.append({"task": "T3", "subset": subset, "policy": policy, "input_targets": len(ids),
                "targets_with_returned_positions": whole.loc[whole.has_model_prediction, "target_profile_id"].nunique(),
                "targets_without_returned_positions": len(ids) - whole.loc[whole.has_model_prediction, "target_profile_id"].nunique(),
                **{key: metric[key] for key in original_keys}, "interpretation": "descriptive exclusion sensitivity; subset composition changed; original zero-fill is a policy scenario, not proof of true zero",
                "ci_status": "paired_target_bootstrap_for_clean_pooled_rho_and_policy_delta_only" if subset == "retained_known_nonoverlap" else "point_estimate_only"})
            if subset == "full_original_panel" and policy == "all_observed_zero_fill":
                check_replication("T3_all_observed", metric, m3.iloc[0], original_keys, checks)
            if subset == "retained_known_nonoverlap":
                write(f"CRISPRon_T3_retained_per_target_{policy}.csv", per_target)
    write("CRISPRon_T3_retained_joined_positions.csv", clean3)
    write("CRISPRon_exclusion_point_metrics.csv", pd.DataFrame(point_rows))
    write("formal_baseline_replication_check.csv", pd.DataFrame(checks))

    target_ids = sorted(clean_ids)
    target_arrays = []
    for target_id in target_ids:
        sub = clean3[clean3.target_profile_id.eq(target_id)]
        target_arrays.append((sub["_obs"].to_numpy(float), sub["_pred_filled"].to_numpy(float), sub.has_model_prediction.to_numpy(bool)))
    supported = sum(bool(mask.any()) for _, _, mask in target_arrays)
    assert len(target_ids) >= MIN_CI_TARGETS and supported >= MIN_CI_TARGETS
    rng = np.random.default_rng(SEED)
    boot = []
    for i in range(N_BOOT):
        picks = rng.integers(0, len(target_ids), size=len(target_ids))
        obs = np.concatenate([target_arrays[j][0] for j in picks])
        pred = np.concatenate([target_arrays[j][1] for j in picks])
        mask = np.concatenate([target_arrays[j][2] for j in picks])
        all_rho, returned_rho = rho(pred, obs), rho(pred[mask], obs[mask])
        boot.append({"replicate": i + 1, "all_observed_rho": all_rho, "returned_only_rho": returned_rho,
            "delta_all_minus_returned": all_rho - returned_rho, "sampled_target_clusters": len(picks),
            "sampled_clusters_with_returned_positions_counting_multiplicity": sum(bool(target_arrays[j][2].any()) for j in picks),
            "all_observed_rows_counting_multiplicity": len(obs), "returned_rows_counting_multiplicity": int(mask.sum())})
    boot = pd.DataFrame(boot)
    write("CRISPRon_T3_retained_paired_target_bootstrap.csv", boot)
    all_point = rho(clean3["_pred_filled"], clean3["_obs"])
    ret = clean3[clean3.has_model_prediction]
    ret_point = rho(ret["_pred_filled"], ret["_obs"])
    ci_rows = []
    for column, point in [("all_observed_rho", all_point), ("returned_only_rho", ret_point), ("delta_all_minus_returned", all_point - ret_point)]:
        values = boot[column].to_numpy(float)
        values = values[np.isfinite(values)]
        lo, hi = np.quantile(values, [.025, .975], method="linear")
        ci_rows.append({"task": "T3", "subset": "retained_known_nonoverlap", "statistic": column, "point_estimate": point,
            "ci_95_lower": lo, "ci_95_upper": hi, "bootstrap_replicates_requested": N_BOOT, "bootstrap_replicates_finite": len(values), "seed": SEED,
            "resampling_unit": "target_profile_id; draw all retained executed target clusters; same draw for returned-only and all-observed policies; explicit adapter skips excluded",
            "n_input_target_clusters": len(target_ids), "n_targets_with_returned_positions": supported,
            "inference_scope": "conditional on existing selected BE-Hive panel; not study-level external validation or a test that overlap caused the change"})
    write("CRISPRon_T3_retained_paired_policy_CI.csv", pd.DataFrame(ci_rows))
    for source in sources.values():
        source["sha256_after"] = digest(source["path"])
        source["unchanged_during_sensitivity"] = source["sha256_before"] == source["sha256_after"]
    assert all(x["unchanged_during_sensitivity"] for x in sources.values())
    write("sensitivity_input_source_hashes.csv", pd.DataFrame(sources.values()))
    manifest = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
        "original_baseline_metrics_reproduced": all(x["passed"] for x in checks), "n_baseline_metrics_checked": len(checks), "execution_or_explicit_skip_status_verified_for_all_input_targets": True,
        "exclusion_rule": "exclude explicit adapter skips first; forward exact 20nt OR 30nt against actual ABE/CBE backend author train/test union; actual class agrees with experimental class for every executed target",
        "no_model_inference_or_fitting": True, "no_original_file_changes": True, "no_figures_or_manuscript_edits": True,
        "T2_clean_targets": 8, "T2_CI": "not estimated: eight targets are descriptive only; threshold 20 is pragmatic, not formal sample-size justification",
        "T3_clean_targets": len(clean_ids), "T3_clean_targets_with_returned_positions": supported, "bootstrap_replicates": N_BOOT, "seed": SEED,
        "T3_execution_denominator_correction": "300 original panel targets = 149 executed (69 ABE + 80 CBE) + 151 explicit no-editor-mapping skips; 149 executed = 127 known development matches + 22 retained",
        "original_versus_retained_comparison": "different target composition; no paired original-versus-clean significance claim or attribution entirely to leakage",
        "product_metrics": "reaggregate frozen per-target product metrics using original definitions; no new outcome calls",
        "unknown_training_provenance": "Other models not audited against their own row-level training lists. CRISPRon matches on other models' evaluation panels are not evidence of their own training overlap.",
        "remaining_limitations": "Exact source-sequence audit does not establish full training independence, same editor variant/cell, outcome-independent sampling or new common model panels; zero-fill semantics remain unresolved under R2.6",
    }
    (OUT / "sensitivity_execution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"flow": panel_summary, "clean_CI": ci_rows, "execution": manifest}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
