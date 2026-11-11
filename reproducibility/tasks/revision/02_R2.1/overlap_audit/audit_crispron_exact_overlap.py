"""Read-only sequence-overlap audit; writes only beneath this script's folder.

No model inference, performance calculation, fitting, or source-file mutation.
Python standard library only. Run with the bundled Python listed in README.md.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("E:/AIselect/BEs")
OUT = Path(__file__).resolve().parent
EXPECTED_OUT = (ROOT / "SCI-data/返修/02_R2.1/overlap_audit").resolve()
assert OUT == EXPECTED_OUT, "Keep all audit output inside the authorized directory"
CORE = ROOT / "benchmark_core"
DEV = ROOT / "CRISPRon-BE/repo/data_for_model_development_V1.0"
T1 = CORE / "formal_results/T1_v8_label_blind_known_crispron_clean"
LEGACY_SCRIPT = CORE / "scripts/build_t1_v8_label_blind_panel.py"
LEGACY_CONTRACT = T1 / "manifest/LABEL_BLIND_SELECTION_CONTRACT.json"
PRIMARY_MODELS = ["behive_bystander", "forecast_be", "bedict_v2", "deepbe", "crispron_be", "be_endo", "yachie"]
T2_FILES = {
    "behive_bystander": "T2_expand_behive_native50.csv",
    "forecast_be": "T2_expand_forecast_position.csv",
    "bedict_v2": "T2_expand_bedict_v2.csv",
    "deepbe": "T2_expand_deepbe.csv",
    "crispron_be": "T2_expand_crispron_be.csv",
    "be_endo": "T2_expand_be_endo.csv",
    "yachie": "T2_expand_yachie_targets.csv",
}
SCOPES = ["legacy_ABE_development", "editor_class_matched_development", "all_readable_development"]
IDENTITY_FIELDS = ["observation_id", "target_profile_id", "design_id", "source_profile_id", "source_family", "source_dataset", "source_file", "source_row_key", "editor", "editor_class", "cell_context", "intended_edit", "measurement_scale"]
inputs = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register(path, role):
    path = Path(path).resolve()
    inputs[str(path)] = {"path": str(path), "role": role, "bytes": path.stat().st_size, "sha256_before": sha(path)}


def clean_dna(value):
    # Exact legacy T1 normalization: uppercase, retain A/C/G/T only.
    return "".join(c for c in str(value or "").upper() if c in "ACGT")


def extract_30mer(context, guide):
    # Exact legacy first-occurrence rule, including its boundary conditions.
    context, guide = clean_dna(context), clean_dna(guide)
    idx = context.find(guide)
    if idx < 4 or idx + 26 > len(context):
        return ""
    result = context[idx - 4:idx + 26]
    return result if len(result) == 30 else ""


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def valid(seq, length):
    return len(seq) == length and all(c in "ACGT" for c in seq)


def csv_read(path, delimiter=","):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def csv_write(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0] if rows else [])
    path = OUT / name
    op = gzip.open if name.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def occurrence_count(context, guide):
    return len(list(re.finditer(f"(?={re.escape(guide)})", context))) if guide else 0


def load_references():
    references = []
    source_summary = []
    raw_index = []
    for editor_class in ["ABE", "CBE"]:
        for split in ["train", "test"]:
            path = DEV / f"{editor_class}_{split}.tsv.gz"
            register(path, f"CRISPRon-{editor_class} published development {split}")
            groups = {}
            qa = Counter()
            guide_lengths, context_lengths, offsets = Counter(), Counter(), Counter()
            for record_number, raw in enumerate(csv_read(path, "\t"), 1):
                guide, context = clean_dna(raw["refs"]), clean_dna(raw["surro_target"])
                seq30 = extract_30mer(context, guide)
                key = (raw["refs"], raw["surro_target"], raw.get("source", ""), raw.get("partition", ""))
                if key not in groups:
                    ref_id = "DEV_" + hashlib.sha256((path.name + "\x1f" + "\x1f".join(key)).encode()).hexdigest()[:20]
                    groups[key] = {
                        "reference_id": ref_id, "development_file": str(path), "development_file_sha256": inputs[str(path.resolve())]["sha256_before"],
                        "development_editor_class": editor_class, "development_split": split,
                        "development_source": raw.get("source", ""), "development_partition": raw.get("partition", ""),
                        "refs_raw": raw["refs"], "surro_target_raw": raw["surro_target"], "guide_20nt_clean": guide,
                        "surro_target_clean": context, "context_30nt_legacy": seq30,
                        "guide_offset_in_surro_target_0based": context.find(guide),
                        "guide_occurrences_in_surro_target": occurrence_count(context, guide),
                        "reverse_guide_occurrences_in_surro_target": occurrence_count(context, reverse_complement(guide)),
                        "normalization_changed_guide": guide != raw["refs"].upper(),
                        "normalization_changed_context": context != raw["surro_target"].upper(),
                        "valid_20nt": valid(guide, 20), "valid_30nt": valid(seq30, 30),
                        "first_data_record_1based": record_number, "last_data_record_1based": record_number, "outcome_record_count": 0,
                        "exact_editor_variant_available": False, "cell_context_available": False,
                    }
                entry = groups[key]
                entry["outcome_record_count"] += 1
                entry["last_data_record_1based"] = record_number
                raw_index.append({"development_file": path.name, "data_record_1based": record_number, "reference_id": entry["reference_id"]})
                qa["outcome_records"] += 1
                qa["valid_20nt_records"] += valid(guide, 20)
                qa["valid_30nt_records"] += valid(seq30, 30)
                qa["guide_occurs_multiple_times_records"] += occurrence_count(context, guide) > 1
                qa["forward_guide_absent_records"] += guide not in context
                qa["normalization_changed_guide_records"] += guide != raw["refs"].upper()
                qa["normalization_changed_context_records"] += context != raw["surro_target"].upper()
                guide_lengths[len(guide)] += 1
                context_lengths[len(context)] += 1
                offsets[context.find(guide)] += 1
            entries = list(groups.values())
            references.extend(entries)
            source_summary.append({"development_file": path.name, "editor_class": editor_class, "split": split, **dict(qa),
                "reference_catalog_records": len(entries), "unique_nonempty_20nt": len({r["guide_20nt_clean"] for r in entries if r["guide_20nt_clean"]}),
                "unique_nonempty_30nt": len({r["context_30nt_legacy"] for r in entries if r["context_30nt_legacy"]}),
                "guide_length_counts": json.dumps(dict(sorted(guide_lengths.items()))), "surro_target_length_counts": json.dumps(dict(sorted(context_lengths.items()))),
                "guide_offset_counts": json.dumps(dict(sorted(offsets.items()))),
            })
    csv_write("development_reference_catalog.csv", references)
    csv_write("development_outcome_record_index.csv.gz", raw_index)
    csv_write("development_source_summary.csv", source_summary)
    return references, source_summary


def load_panels():
    specs = [("T1", "common_four_scalar", T1 / "data/T1_V8_LABEL_BLIND_SELECTED_METADATA.csv.gz", "primary")]
    specs += [("T2", model, CORE / "data/processed/T2_bystander_matrix/panels" / filename, "primary") for model, filename in T2_FILES.items()]
    specs += [("T3", model, CORE / "data/processed/T3_position_profile_matrix/panels_v1" / f"T3_v1_{model}.csv", "primary" if model in PRIMARY_MODELS else "exploratory") for model in PRIMARY_MODELS + ["epcnnbe", "deepbaseeditor"]]
    records, panels, membership = {}, [], []
    for task, model, path, role in specs:
        register(path, f"{task} existing {role} input manifest: {model}")
        panel_keys = []
        for rownum, raw in enumerate(csv_read(path), 1):
            native_id = raw.get("observation_id") or raw.get("target_profile_id")
            assert native_id, (path, rownum, "missing stable ID")
            key = (task, native_id)
            guide_raw, seq30_raw = raw.get("protospacer_20nt", ""), raw.get("sequence_30nt", raw.get("sequence_context_30nt", ""))
            guide, seq30 = clean_dna(guide_raw), clean_dna(seq30_raw)
            original_context = raw.get("sequence_context_original", "")
            sequence50 = raw.get("sequence_50nt", "")
            reconstructed30 = extract_30mer(original_context, guide) if original_context else ""
            recovered30 = reconstructed30 if not seq30 and valid(reconstructed30, 30) else ""
            rec = {"task": task, "record_id": native_id, **{f: raw.get(f, "") for f in IDENTITY_FIELDS},
                "guide_20nt_raw": guide_raw, "guide_20nt_clean": guide, "context_30nt_raw": seq30_raw, "context_30nt_clean": seq30,
                "sequence_context_original": original_context, "sequence_50nt": sequence50,
                "context_30nt_reconstructed_from_original": reconstructed30,
                "missing_30nt_recovered_from_original": bool(recovered30),
                "context_30nt_with_missing_recovered": seq30 or recovered30,
                "sequence_cluster_id": "SEQ_" + hashlib.sha256((guide + "\x1f" + seq30).encode()).hexdigest()[:20],
                "valid_20nt": valid(guide, 20), "valid_30nt": valid(seq30, 30),
                "normalization_changed_20nt": guide != guide_raw.upper(), "normalization_changed_30nt": seq30 != seq30_raw.upper(),
                "guide_at_30nt_offset4": valid(guide, 20) and valid(seq30, 30) and seq30[4:24] == guide,
                "guide_occurrences_in_30nt": occurrence_count(seq30, guide),
                "reverse_guide_occurrences_in_30nt": occurrence_count(seq30, reverse_complement(guide)),
                "30nt_reextract_from_original_context_agrees": extract_30mer(original_context, guide) == seq30 if original_context else "not_available",
                "30nt_reextract_from_50nt_agrees": extract_30mer(sequence50, guide) == seq30 if sequence50 else "not_available",
                "legacy_T1_flag_in_input": raw.get("known_crispron_development_overlap_any", "not_available"),
            }
            if key in records:
                assert records[key] == rec, f"Same target ID has differing sequence/context: {key}"
            else:
                records[key] = rec
            panel_keys.append(key)
            membership.append({"task": task, "panel_model": model, "role": role, "record_id": native_id, "input_file": str(path), "input_data_record_1based": rownum})
        assert len(panel_keys) == len(set(panel_keys)), f"Duplicate IDs in {path}"
        panels.append({"task": task, "panel": model, "role": role, "keys": panel_keys, "input_file": str(path)})
    for task in ["T2", "T3"]:
        primary_keys = sorted({key for p in panels if p["task"] == task and p["role"] == "primary" for key in p["keys"]})
        panels.append({"task": task, "panel": "UNION_seven_primary", "role": "primary_union", "keys": primary_keys, "input_file": "union of primary input manifests; deduplicated by target_profile_id"})
        if task == "T3":
            all_keys = sorted(k for k in records if k[0] == task)
            panels.append({"task": task, "panel": "UNION_all_nine_including_exploratory", "role": "all_union", "keys": all_keys, "input_file": "union including EPCNNBE and DeepBaseEditor exploratory input manifests"})
    csv_write("panel_membership.csv", membership)
    return records, panels


def scope_accepts(scope, rec, ref):
    if scope == "legacy_ABE_development":
        return ref["development_editor_class"] == "ABE"
    if scope == "editor_class_matched_development":
        return rec["editor_class"] in ["ABE", "CBE"] and ref["development_editor_class"] == rec["editor_class"]
    return True


def main():
    started = datetime.now(timezone.utc).isoformat()
    assert extract_30mer("AAAACCCCCCCCCCCCCCCCCCCCTGGAAA", "C" * 20) == "AAAACCCCCCCCCCCCCCCCCCCCTGGAAA"
    assert reverse_complement(reverse_complement("ACGTTT")) == "ACGTTT"
    for p, role in [(LEGACY_SCRIPT, "original T1 exclusion implementation"), (LEGACY_CONTRACT, "original T1 source hash contract"), (DEV / "README.md", "author-provided development data description"), (CORE / "scripts/prepare_t2_expanded_panels.py", "existing T2 input window provenance"), (CORE / "scripts/prepare_t3_v1_expanded_panels.py", "existing T3 input window provenance")]:
        register(p, role)
    original_contract = json.loads(LEGACY_CONTRACT.read_text(encoding="utf-8-sig"))
    references, source_summary = load_references()
    original_source_hash_match = {split: sha(DEV / f"ABE_{split}.tsv.gz") == original_contract[f"{split}_sha256"] for split in ["train", "test"]}
    assert all(original_source_hash_match.values()), "Original ABE source hashes changed: replication scope no longer identical"
    records, panels = load_panels()
    index = {20: defaultdict(list), 30: defaultdict(list)}
    for ref in references:
        for window, field in [(20, "guide_20nt_clean"), (30, "context_30nt_legacy")]:
            if ref[field]:
                index[window][ref[field]].append(ref)
    hits = []
    for rec in records.values():
        for scope in SCOPES:
            rec[f"{scope}__recovered_missing30_forward"] = False
            rec[f"{scope}__recovered_missing30_reverse_complement_literal"] = False
            for direction in ["forward", "reverse_complement_literal"]:
                for window in [20, 30]:
                    rec[f"{scope}__{direction}__{window}nt"] = False
                for split in ["train", "test"]:
                    rec[f"{scope}__{direction}__{split}_any"] = False
        queries = [(20, "guide_20nt_clean", "as_stored"), (30, "context_30nt_clean", "as_stored")]
        if rec["missing_30nt_recovered_from_original"]:
            queries.append((30, "context_30nt_reconstructed_from_original", "missing30_recovered_from_original_context"))
        for window, field, representation in queries:
            sequence = rec[field]
            if not sequence:
                continue
            for direction, query in [("forward", sequence), ("reverse_complement_literal", reverse_complement(sequence))]:
                for ref in index[window].get(query, []):
                    accepted = [s for s in SCOPES if scope_accepts(s, rec, ref)]
                    for scope in accepted:
                        if representation == "as_stored":
                            rec[f"{scope}__{direction}__{window}nt"] = True
                            rec[f"{scope}__{direction}__{ref['development_split']}_any"] = True
                        else:
                            rec[f"{scope}__recovered_missing30_{direction}"] = True
                    hits.append({"task": rec["task"], "record_id": rec["record_id"], "observation_id": rec["observation_id"], "target_profile_id": rec["target_profile_id"], "design_id": rec["design_id"],
                        "reference_id": ref["reference_id"], "match_window_nt": window, "direction_policy": direction, "window_representation": representation,
                        "evaluation_sequence_before_orientation": sequence, "evaluation_sequence_queried": query, "development_matched_sequence": query,
                        "development_file": ref["development_file"], "development_file_sha256": ref["development_file_sha256"],
                        "development_split": ref["development_split"], "development_editor_class": ref["development_editor_class"],
                        "development_source": ref["development_source"], "development_partition": ref["development_partition"],
                        "development_guide_20nt": ref["guide_20nt_clean"], "development_context_30nt": ref["context_30nt_legacy"],
                        "first_development_data_record_1based": ref["first_data_record_1based"], "development_outcome_record_count": ref["outcome_record_count"],
                        "evaluation_editor": rec["editor"], "evaluation_editor_class": rec["editor_class"], "evaluation_cell_context": rec["cell_context"],
                        "editor_class_comparable": rec["editor_class"] == ref["development_editor_class"],
                        "exact_editor_variant_comparability": "unknown_not_encoded_in_development_table", "cell_context_comparability": "unknown_not_encoded_in_development_table",
                        "experimental_observation_identity": "not_established_by_sequence_match", "accepted_audit_scopes": ";".join(accepted),
                        "interpretation": "legacy same-orientation exact sequence hit" if direction == "forward" else "literal reverse-complement sensitivity only; not proof of identical native guide/context contract",
                    })
        for scope in SCOPES:
            for direction in ["forward", "reverse_complement_literal"]:
                rec[f"{scope}__{direction}__any"] = any(rec[f"{scope}__{direction}__{w}nt"] for w in [20, 30])
            rec[f"{scope}__rc_only_any"] = rec[f"{scope}__reverse_complement_literal__any"] and not rec[f"{scope}__forward__any"]
        if rec["task"] == "T1":
            assert str(rec["legacy_ABE_development__forward__any"]).lower() == rec["legacy_T1_flag_in_input"].lower(), "Legacy T1 exclusion flag not replicated"
    csv_write("record_overlap_audit.csv", sorted(records.values(), key=lambda r: (r["task"], r["record_id"])))
    csv_write("exact_overlap_hits.csv", hits)
    summaries = []
    for panel in panels:
        rows = [records[k] for k in panel["keys"]]
        for scope in SCOPES:
            comparable = [r for r in rows if scope != "editor_class_matched_development" or r["editor_class"] in ["ABE", "CBE"]]
            overlap = [r for r in rows if r[f"{scope}__forward__any"]]
            summary = {"task": panel["task"], "panel": panel["panel"], "panel_role": panel["role"], "audit_scope": scope,
                "n_input_records": len(rows), "n_observation_ids": len({r["observation_id"] for r in rows if r["observation_id"]}) or "not_encoded",
                "n_target_profile_ids": len({r["target_profile_id"] for r in rows if r["target_profile_id"]}) or "not_encoded",
                "n_native_design_ids": len({r["design_id"] for r in rows if r["design_id"]}) or "not_encoded",
                "n_sequence_clusters_guide20_context30": len({r["sequence_cluster_id"] for r in rows}),
                "n_comparable_20nt_records": sum(r["valid_20nt"] for r in comparable), "n_comparable_30nt_records": sum(r["valid_30nt"] for r in comparable),
                "n_comparable_both_records": sum(r["valid_20nt"] and r["valid_30nt"] for r in comparable),
                "n_missing_30nt_recovered_from_original": sum(r["missing_30nt_recovered_from_original"] for r in comparable),
                "n_comparable_30nt_after_recovery": sum(valid(r["context_30nt_with_missing_recovered"], 30) for r in comparable),
                "coverage_20nt_fraction": sum(r["valid_20nt"] for r in comparable) / len(rows), "coverage_30nt_fraction": sum(r["valid_30nt"] for r in comparable) / len(rows),
                "n_forward_20nt_hits": sum(r[f"{scope}__forward__20nt"] for r in rows), "n_forward_30nt_hits": sum(r[f"{scope}__forward__30nt"] for r in rows),
                "n_forward_either_hits": len(overlap), "n_forward_train_hits": sum(r[f"{scope}__forward__train_any"] for r in rows), "n_forward_test_hits": sum(r[f"{scope}__forward__test_any"] for r in rows),
                "n_recovered_missing30_forward_hits": sum(r[f"{scope}__recovered_missing30_forward"] for r in rows),
                "n_forward_30nt_hits_after_recovery": sum(r[f"{scope}__forward__30nt"] or r[f"{scope}__recovered_missing30_forward"] for r in rows),
                "n_additional_records_from_30nt_recovery": sum(r[f"{scope}__recovered_missing30_forward"] and not r[f"{scope}__forward__any"] for r in rows),
                "n_forward_hit_native_design_ids": len({r["design_id"] for r in overlap if r["design_id"]}) if panel["task"] == "T1" else "not_encoded",
                "n_forward_hit_sequence_clusters": len({r["sequence_cluster_id"] for r in overlap}),
                "n_forward_remaining_records": len(rows) - len(overlap),
                "n_rc_literal_20nt_hits": sum(r[f"{scope}__reverse_complement_literal__20nt"] for r in rows), "n_rc_literal_30nt_hits": sum(r[f"{scope}__reverse_complement_literal__30nt"] for r in rows),
                "n_rc_literal_either_hits": sum(r[f"{scope}__reverse_complement_literal__any"] for r in rows), "n_rc_only_added_records": sum(r[f"{scope}__rc_only_any"] for r in rows),
                "experimental_context_comparability": "editor class check available; exact editor variant/cell not established in development TSV",
                "input_file": panel["input_file"],
            }
            summaries.append(summary)
    csv_write("panel_overlap_summary.csv", summaries)
    qa = []
    for panel in panels:
        rows = [records[k] for k in panel["keys"]]
        qa.append({"task": panel["task"], "panel": panel["panel"], "n_records": len(rows),
            "n_invalid_20nt": sum(not r["valid_20nt"] for r in rows), "n_invalid_30nt": sum(not r["valid_30nt"] for r in rows),
            "n_guide_not_at_context30_offset4": sum(not r["guide_at_30nt_offset4"] for r in rows),
            "n_guide_occurs_multiple_times_in_context30": sum(r["guide_occurrences_in_30nt"] > 1 for r in rows),
            "n_normalization_changed_20nt": sum(r["normalization_changed_20nt"] for r in rows), "n_normalization_changed_30nt": sum(r["normalization_changed_30nt"] for r in rows),
            "n_reextract_original_context_disagrees": sum(r["30nt_reextract_from_original_context_agrees"] is False for r in rows),
            "n_reextract_50nt_disagrees": sum(r["30nt_reextract_from_50nt_agrees"] is False for r in rows),
            "n_missing30_recovered_from_original_context": sum(r["missing_30nt_recovered_from_original"] for r in rows),
            "editor_class_counts": json.dumps(dict(Counter(r["editor_class"] for r in rows))),
        })
    csv_write("panel_orientation_and_sequence_QA.csv", qa)
    other_models = ["BE-Hive efficiency", "BE-Hive bystander", "DeepBE", "FORECasT-BE", "DeepBaseEditor", "BEDICT-V2", "BE_Endo", "Yachie", "EPCNNBE"]
    provenance = [{"model_role": "CRISPRon-BE (ABE/CBE)", "row_level_training_relation_status": "known_local_development_lists_audited", "strict_external_independence_established": False,
        "evidence": "ABE/CBE train and author test TSVs; source README and hashes recorded", "limitation": "Known development-sequence overlap only; no claim that readable lists prove all training uses or exact experimental identity"}]
    provenance += [{"model_role": m, "row_level_training_relation_status": "unknown_not_audited_with_row_level_training_list", "strict_external_independence_established": False,
        "evidence": "No row-level training membership list for this model loaded by this CRISPRon-specific audit", "limitation": "Do not reinterpret CRISPRon non-overlap as absence of overlap for this model; no exhaustive absence-of-files claim"} for m in other_models]
    csv_write("training_provenance_status.csv", provenance)
    for item in inputs.values():
        item["sha256_after"] = sha(item["path"])
        item["unchanged_during_audit"] = item["sha256_before"] == item["sha256_after"]
    assert all(x["unchanged_during_audit"] for x in inputs.values())
    csv_write("input_source_hashes.csv", inputs.values())
    contract = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable, "python_version": sys.version,
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha(__file__), "output_directory": str(OUT),
        "analysis_type": "exact sequence-overlap audit only; no performance evaluation, model inference, retraining, or original file changes",
        "legacy_definition": "uppercase + retain ACGT; same-orientation guide equality; first guide occurrence in development surro_target, slice [idx-4:idx+26]; ABE train/test union",
        "legacy_original_source_hashes_match": original_source_hash_match, "T1_legacy_stored_flags_reproduced": True,
        "additional_scopes": "CBE-readable editor-class-matched and all-readable ABE+CBE development; separate literal reverse-complement sensitivity",
        "missing_window_recovery": "As-stored 30nt coverage retained; missing 30nt windows separately reconstructed from original sequence context with the same legacy first-guide rule; recovered-only hits flagged separately",
        "reverse_complement_limitation": "Reverse-complementing a 30nt 4+20+6 window yields 6+20+4, not the same native input layout; literal matches do not establish equivalent model input or experimental context",
        "statistical_unit": "T1 observation_id and existing design_id; T2/T3 target_profile_id. Sequence clusters are diagnostic hashes of guide20+context30, not recovered native design IDs",
        "hash_input_count": len(inputs), "all_inputs_unchanged": True, "record_count": len(records), "hit_reference_window_rows": len(hits),
        "development_catalog_note": "Outcome rows coalesced by original refs+surro_target+source+partition within file; exact raw record membership retained in development_outcome_record_index.csv.gz",
    }
    (OUT / "audit_execution_manifest.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = [s for s in summaries if s["audit_scope"] == "legacy_ABE_development" and (s["panel"] == "common_four_scalar" or s["panel"] == "UNION_seven_primary")]
    print(json.dumps({"execution": contract, "legacy_primary_task_summary": primary, "source_summary": source_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
