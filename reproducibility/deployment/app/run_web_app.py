from __future__ import annotations

import csv
import html
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote

from be_predictor.candidates import generate_candidates, score_candidates
from be_predictor.service import analyze


HOST = os.environ.get("BE_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("BE_WEB_PORT", "8765"))


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BE-Benchmark prediction router</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #222; background: #fbfaf7; }
    .card { max-width: 1180px; margin: auto; background: white; border: 1px solid #eadfd2; border-radius: 18px; padding: 26px; box-shadow: 0 8px 24px rgba(60,40,20,0.08); }
    h1 { margin-top: 0; color: #6b4b2e; }
    h2 { color: #6b4b2e; margin-top: 28px; }
    h3 { color: #7b5b40; margin-bottom: 8px; }
    label { display: block; margin-top: 14px; font-weight: 600; }
    textarea, input, select { width: 100%; padding: 10px; border: 1px solid #d8c8b8; border-radius: 10px; font-family: Consolas, monospace; font-size: 14px; box-sizing: border-box; }
    button { margin-top: 18px; padding: 11px 20px; border: 0; border-radius: 999px; background: #F1766D; color: white; font-weight: 700; cursor: pointer; }
    button:hover { background: #DB614F; }
    pre { white-space: pre-wrap; background: #fff7ed; border: 1px solid #f1d8bd; border-radius: 12px; padding: 16px; overflow-x: auto; }
    .note { color: #666; font-size: 14px; line-height: 1.5; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
    .mini-card { background: #fffaf3; border: 1px solid #f0dfcc; border-radius: 14px; padding: 14px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee2d4; padding: 9px 8px; text-align: left; vertical-align: top; }
    th { background: #fff3e4; color: #6b4b2e; position: sticky; top: 0; }
    .table-wrap { max-height: 520px; overflow: auto; border: 1px solid #f0dfcc; border-radius: 12px; }
    .status { display: inline-block; padding: 4px 9px; border-radius: 999px; font-weight: 700; font-size: 12px; }
    .completed { background: #DDF2DF; color: #176B3A; }
    .skipped { background: #FFF0C9; color: #805500; }
    .failed { background: #FAD5D1; color: #9C241B; }
    .notrun { background: #E8E3F3; color: #51427A; }
    .downloads a { display: inline-block; margin-right: 10px; margin-top: 10px; color: white; text-decoration: none; background: #7A70B5; padding: 9px 14px; border-radius: 999px; font-weight: 700; }
    .small { font-size: 12px; color: #666; }
    .mono { font-family: Consolas, monospace; word-break: break-all; }
  </style>
</head>
<body>
<div class="card">
  <h1>BE-Benchmark-informed prediction router</h1>
  <p class="note">MVP v0.5: routes a sequence/editor input to reproduced base-editor models, invokes feasible local backends, generates candidate guides/windows from long sequences, and renders exportable reports. It does not fabricate numeric predictions when an input contract is not met.</p>
  <form method="post" action="/predict">
    <label>DNA sequence</label>
    <textarea name="sequence" rows="4" placeholder="Example: GGGGAAAAACCCCCGGGGGTTTTTGGGAAA">GGGGAAAAACCCCCGGGGGTTTTTGGGAAA</textarea>
    <label>Editor</label>
    <input name="editor" value="ABE8e">
    <label>Cell/context label (optional; required for BE-Hive native models)</label>
    <input name="cell" value="" placeholder="Example: mES">
    <label>Cas/PAM variant (optional)</label>
    <input name="cas" value="SpRY" placeholder="Example: SpCas9, SpRY, NG">
    <label>Reference edited outcome (optional; required for outcome-specific EPCNNBE/Yachie calls)</label>
    <input name="outcome" value="GGGGAAAAGCCCCCGGGGGTTTTTGGGAAA" placeholder="30nt edited outcome sequence">
    <label>Task</label>
    <select name="task">
      <option value="auto">auto</option>
      <option value="scalar_efficiency">scalar_efficiency</option>
      <option value="outcome_profile">outcome_profile</option>
      <option value="position_profile">position_profile</option>
      <option value="high_low_screening">high_low_screening</option>
      <option value="design_recommendation">design_recommendation</option>
    </select>
    <label><input style="width:auto" type="checkbox" name="run_models" value="1"> Run all feasible local backend connectors. This may take about 1-2 minutes when CRISPRon-BE is included.</label>
    <label><input style="width:auto" type="checkbox" name="generate_candidates" value="1"> Generate candidate guides/windows from this sequence instead of single-input prediction</label>
    <label>Maximum candidates for candidate generation</label>
    <input name="max_candidates" value="50">
    <label>Candidate scoring backends (optional; comma-separated: forecast_be, deepbaseeditor)</label>
    <input name="score_backends" value="forecast_be,deepbaseeditor">
    <button type="submit">Route models</button>
  </form>
  __RESULT__
</div>
</body>
</html>"""


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: object, digits: int = 4) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _status_badge(status: str) -> str:
    status = status or "not_run"
    cls = {
        "completed": "completed",
        "skipped": "skipped",
        "failed": "failed",
        "not_implemented": "notrun",
    }.get(status, "notrun")
    return f'<span class="status {cls}">{_esc(status)}</span>'


def _result_csv(result: dict) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model_id",
            "status",
            "prediction_count",
            "skipped_count",
            "score_or_efficiency",
            "top_outcome",
            "top_outcome_value",
            "reason",
        ],
    )
    writer.writeheader()
    for output in result.get("model_outputs", []):
        summary = output.get("prediction_summary", {})
        eff = ""
        top_outcome = ""
        top_value = ""
        if "predicted_efficiency_score" in output:
            eff = output.get("predicted_efficiency_score")
        elif summary.get("efficiency"):
            first = summary["efficiency"][0]
            eff = first.get("predicted_efficiency")
            if eff in {None, ""}:
                eff = first.get("predicted_score")
        if summary.get("top_outcomes"):
            first_outcome = summary["top_outcomes"][0]
            top_outcome = first_outcome.get("outcome", "")
            top_value = first_outcome.get("proportion")
            if top_value is None:
                top_value = first_outcome.get("frequency")
            if top_value is None:
                top_value = first_outcome.get("score")
        writer.writerow(
            {
                "model_id": output.get("model_id", ""),
                "status": output.get("status", ""),
                "prediction_count": output.get("prediction_count", ""),
                "skipped_count": output.get("skipped_count", ""),
                "score_or_efficiency": eff,
                "top_outcome": top_outcome,
                "top_outcome_value": top_value,
                "reason": output.get("reason", ""),
            }
        )
    return handle.getvalue()


def _download_links(result: dict) -> str:
    json_payload = quote(json.dumps(result, ensure_ascii=False, indent=2))
    csv_payload = quote(_result_csv(result))
    return (
        '<div class="downloads">'
        f'<a download="be_prediction_result.json" href="data:application/json;charset=utf-8,{json_payload}">Download JSON</a>'
        f'<a download="be_prediction_backend_summary.csv" href="data:text/csv;charset=utf-8,{csv_payload}">Download CSV summary</a>'
        "</div>"
    )


def _candidate_csv(report: dict) -> str:
    handle = io.StringIO()
    candidates = report.get("candidates", [])
    if not candidates:
        handle.write("candidate_id\n")
        return handle.getvalue()
    fieldnames = []
    for row in candidates:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(candidates)
    return handle.getvalue()


def _candidate_download_links(report: dict) -> str:
    json_payload = quote(json.dumps(report, ensure_ascii=False, indent=2))
    csv_payload = quote(_candidate_csv(report))
    return (
        '<div class="downloads">'
        f'<a download="be_candidate_generation_result.json" href="data:application/json;charset=utf-8,{json_payload}">Download candidate JSON</a>'
        f'<a download="be_candidate_generation_candidates.csv" href="data:text/csv;charset=utf-8,{csv_payload}">Download candidate CSV</a>'
        "</div>"
    )


def _candidate_report_html(report: dict) -> str:
    inp = report.get("input", {})
    warnings = "<br>".join(_esc(w) for w in report.get("warnings", [])) or "None"
    backend_status = report.get("scoring_backend_status", {})
    backend_rows = ""
    for backend, status in backend_status.items():
        backend_rows += (
            "<tr>"
            f"<td>{_esc(backend)}</td>"
            f"<td>{_status_badge(status.get('status', ''))}</td>"
            f"<td>{_esc(status.get('scored_candidates', ''))}</td>"
            f"<td>{_esc(status.get('reason', ''))}</td>"
            "</tr>"
        )
    backend_table = ""
    if backend_rows:
        backend_table = (
            "<h2>Candidate scoring backend status</h2>"
            "<div class='table-wrap'><table><thead><tr><th>Backend</th><th>Status</th><th>Scored candidates</th><th>Reason</th></tr></thead><tbody>"
            + backend_rows
            + "</tbody></table></div>"
        )
    rows = ""
    for row in report.get("candidates", [])[:200]:
        rows += (
            "<tr>"
            f"<td>{_esc(row.get('rank_after_scoring', ''))}</td>"
            f"<td>{_esc(row.get('candidate_id'))}</td>"
            f"<td>{_esc(row.get('strand'))}</td>"
            f"<td class='mono'>{_esc(row.get('guide_20nt'))}</td>"
            f"<td class='mono'>{_esc(row.get('pam_3nt'))}</td>"
            f"<td class='mono'>{_esc(row.get('window_30nt'))}</td>"
            f"<td>{_esc(row.get('editable_positions_1based'))}</td>"
            f"<td>{_fmt(row.get('heuristic_rank_score'))}</td>"
            f"<td>{_fmt(row.get('forecast_be_score'))}</td>"
            f"<td>{_fmt(row.get('deepbaseeditor_efficiency'))}</td>"
            f"<td>{_esc(row.get('warnings'))}</td>"
            "</tr>"
        )
    return (
        "<h2>Candidate generation result</h2>"
        + _candidate_download_links(report)
        + "<div class='grid'>"
        + f"<div class='mini-card'><b>Input length</b><br>{_esc(inp.get('sequence_length'))}</div>"
        + f"<div class='mini-card'><b>Editor family</b><br>{_esc(inp.get('editor_family'))}</div>"
        + f"<div class='mini-card'><b>Total candidates before limit</b><br>{_esc(report.get('total_candidates_before_limit'))}</div>"
        + f"<div class='mini-card'><b>Returned candidates</b><br>{_esc(report.get('returned_candidates'))}</div>"
        + "</div>"
        + f"<p class='note'><b>Warnings:</b> {warnings}</p>"
        + backend_table
        + "<h2>Candidate table</h2>"
        + "<div class='table-wrap'><table>"
        + "<thead><tr><th>Rank</th><th>ID</th><th>Strand</th><th>Guide 20nt</th><th>PAM</th><th>30nt window</th><th>Editable pos.</th><th>Heuristic</th><th>FORECasT-BE</th><th>DeepBaseEditor eff.</th><th>Warnings</th></tr></thead>"
        + "<tbody>"
        + rows
        + "</tbody></table></div>"
    )


def _input_qc_html(result: dict) -> str:
    assessment = result["sequence_assessment"]
    inp = result["input"]
    warnings = assessment.get("warnings") or []
    warning_html = "<br>".join(_esc(w) for w in warnings) if warnings else "None"
    return f"""
    <h2>Input QC</h2>
    <div class="grid">
      <div class="mini-card"><b>Editor family</b><br>{_esc(inp.get('editor_family'))}</div>
      <div class="mini-card"><b>Task</b><br>{_esc(inp.get('requested_task'))}</div>
      <div class="mini-card"><b>Input type</b><br>{_esc(assessment.get('inferred_input_type'))}</div>
      <div class="mini-card"><b>PAM / NGG-like</b><br>{_esc(assessment.get('pam'))} / {_esc(assessment.get('is_ngg_like'))}</div>
      <div class="mini-card"><b>ABE editable positions 4-8</b><br>{_esc(assessment.get('abe_candidate_positions_1based'))}</div>
      <div class="mini-card"><b>CBE editable positions 4-8</b><br>{_esc(assessment.get('cbe_candidate_positions_1based'))}</div>
    </div>
    <p class="note"><b>Warnings:</b> {warning_html}</p>
    """


def _backend_summary_html(result: dict) -> str:
    outputs = result.get("model_outputs") or []
    if not outputs:
        return "<h2>Backend execution</h2><p class='note'>Backends were not executed. Check “Run all feasible local backend connectors” to run real models.</p>"
    rows = []
    for output in outputs:
        summary = output.get("prediction_summary", {})
        score = ""
        if "predicted_efficiency_score" in output:
            score = _fmt(output.get("predicted_efficiency_score"))
        elif summary.get("efficiency"):
            item = summary["efficiency"][0]
            score = _fmt(item.get("predicted_efficiency") if item.get("predicted_efficiency") is not None else item.get("predicted_score"))
        top_outcome = ""
        if summary.get("top_outcomes"):
            item = summary["top_outcomes"][0]
            value = item.get("proportion")
            if value is None:
                value = item.get("frequency")
            if value is None:
                value = item.get("score")
            top_outcome = f"<span class='mono'>{_esc(item.get('outcome'))}</span><br><span class='small'>{_fmt(value)}</span>"
        rows.append(
            "<tr>"
            f"<td class='mono'>{_esc(output.get('model_id'))}</td>"
            f"<td>{_status_badge(output.get('status', ''))}</td>"
            f"<td>{_esc(output.get('prediction_count', ''))}</td>"
            f"<td>{_esc(output.get('skipped_count', ''))}</td>"
            f"<td>{score}</td>"
            f"<td>{top_outcome}</td>"
            f"<td>{_esc(output.get('reason', ''))}</td>"
            "</tr>"
        )
    return (
        "<h2>Backend execution summary</h2>"
        "<div class='table-wrap'><table>"
        "<thead><tr><th>Backend</th><th>Status</th><th>Predictions</th><th>Skipped</th><th>Score/Efficiency</th><th>Top outcome</th><th>Reason</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _prediction_details_html(result: dict) -> str:
    outputs = result.get("model_outputs") or []
    blocks = []
    for output in outputs:
        if output.get("status") != "completed":
            continue
        summary = output.get("prediction_summary", {})
        block = [f"<h3>{_esc(output.get('display_name', output.get('model_id')))}</h3>"]
        if "predicted_efficiency_score" in output:
            block.append(
                "<p>"
                f"<b>Score:</b> {_fmt(output.get('predicted_efficiency_score'))}<br>"
                f"<span class='small'>{_esc(output.get('score_scale', ''))}</span>"
                "</p>"
            )
            positions = output.get("position_scores") or []
            if positions:
                cells = "".join(
                    f"<tr><td>{_esc(item.get('position'))}</td><td>{_fmt(item.get('score'))}</td></tr>"
                    for item in positions
                    if item.get("score") is not None
                )
                block.append("<div class='table-wrap'><table><thead><tr><th>Position</th><th>Score</th></tr></thead><tbody>" + cells + "</tbody></table></div>")
        else:
            eff_items = summary.get("efficiency") or []
            if eff_items:
                cells = "".join(
                    "<tr>"
                    f"<td>{_fmt(item.get('predicted_efficiency'))}</td>"
                    f"<td>{_fmt(item.get('predicted_score'))}</td>"
                    f"<td>{_esc(item.get('model_editor'))}</td>"
                    "</tr>"
                    for item in eff_items[:5]
                )
                block.append("<div class='table-wrap'><table><thead><tr><th>Efficiency</th><th>Score</th><th>Editor</th></tr></thead><tbody>" + cells + "</tbody></table></div>")
            positions = summary.get("position_scores") or []
            if positions:
                cells = "".join(
                    f"<tr><td>{_esc(item.get('position'))}</td><td>{_fmt(item.get('score'))}</td><td>{_esc(item.get('model_editor'))}</td></tr>"
                    for item in positions[:20]
                )
                block.append("<div class='table-wrap'><table><thead><tr><th>Position</th><th>Score</th><th>Editor</th></tr></thead><tbody>" + cells + "</tbody></table></div>")
            outcomes = summary.get("top_outcomes") or []
            if outcomes:
                cells = ""
                for item in outcomes[:10]:
                    value = item.get("proportion")
                    label = "proportion"
                    if value is None:
                        value = item.get("frequency")
                        label = "frequency"
                    if value is None:
                        value = item.get("score")
                        label = "score"
                    cells += (
                        "<tr>"
                        f"<td class='mono'>{_esc(item.get('outcome'))}</td>"
                        f"<td>{_fmt(value)}</td>"
                        f"<td>{_esc(label)}</td>"
                        f"<td>{_esc(item.get('model_editor'))}</td>"
                        "</tr>"
                    )
                block.append("<div class='table-wrap'><table><thead><tr><th>Outcome</th><th>Value</th><th>Scale</th><th>Editor</th></tr></thead><tbody>" + cells + "</tbody></table></div>")
        blocks.append("<div class='mini-card'>" + "".join(block) + "</div>")
    if not blocks:
        return ""
    return "<h2>Prediction details</h2><div class='grid'>" + "".join(blocks) + "</div>"


def _routes_html(result: dict) -> str:
    rows = []
    for route in result.get("routes", []):
        task_text = "<br>".join(
            f"{_esc(t.get('task_label'))}: {_esc(t.get('confidence_tier'))}"
            for t in route.get("task_routes", [])
        )
        basis = "<br>".join(_esc(t.get("benchmark_basis")) for t in route.get("task_routes", []) if t.get("benchmark_basis"))
        rows.append(
            "<tr>"
            f"<td>{_esc(route.get('display_name'))}</td>"
            f"<td>{task_text}</td>"
            f"<td>{basis}</td>"
            f"<td>{_esc(route.get('major_caveat'))}</td>"
            "</tr>"
        )
    return (
        "<h2>Benchmark-informed routing</h2>"
        "<div class='table-wrap'><table>"
        "<thead><tr><th>Model</th><th>Recommended roles</th><th>Benchmark basis</th><th>Caveat</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _result_html(result: dict) -> str:
    return (
        "<h2>Result</h2>"
        + _download_links(result)
        + _input_qc_html(result)
        + _backend_summary_html(result)
        + _prediction_details_html(result)
        + _routes_html(result)
        + "<h2>Interpretation guardrail</h2>"
        + f"<p class='note'>{_esc(result.get('global_interpretation'))}</p>"
    )


class Handler(BaseHTTPRequestHandler):
    def _send_html(self, content: str, status: int = 200) -> None:
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(HTML_PAGE.replace("__RESULT__", ""))
            return
        self._send_html("<h1>404</h1>", status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/api/predict":
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("generate_candidates", False):
                    report = generate_candidates(
                        sequence=payload.get("sequence", ""),
                        editor=payload.get("editor", "ABE"),
                        strands=payload.get("strands", "both"),
                        max_candidates=int(payload.get("max_candidates", 50)),
                    )
                    score_backends = payload.get("score_backends", [])
                    if isinstance(score_backends, str):
                        score_backends = [x.strip() for x in score_backends.split(",") if x.strip()]
                    if score_backends:
                        report = score_candidates(
                            report,
                            editor=payload.get("editor", "ABE"),
                            score_backends=score_backends,
                            cas_variant=payload.get("cas_variant", ""),
                        )
                    self._send_json(report)
                    return
                result = analyze(
                    sequence=payload.get("sequence", ""),
                    editor=payload.get("editor", "ABE"),
                    task=payload.get("task", "auto"),
                    run_models=bool(payload.get("run_models", False)),
                    backend_ids=payload.get("backend_ids"),
                    cell_context=payload.get("cell_context", ""),
                    cas_variant=payload.get("cas_variant", ""),
                    reference_outcome=payload.get("reference_outcome", ""),
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if self.path == "/predict":
            form = parse_qs(raw.decode("utf-8"))
            sequence = form.get("sequence", [""])[0]
            editor = form.get("editor", ["ABE"])[0]
            task = form.get("task", ["auto"])[0]
            cell = form.get("cell", [""])[0]
            cas = form.get("cas", [""])[0]
            outcome = form.get("outcome", [""])[0]
            run_models = form.get("run_models", ["0"])[0] == "1"
            generate_candidate_mode = form.get("generate_candidates", ["0"])[0] == "1"
            if generate_candidate_mode:
                try:
                    max_candidates = int(form.get("max_candidates", ["50"])[0])
                except ValueError:
                    max_candidates = 50
                score_backends = [
                    x.strip()
                    for x in form.get("score_backends", [""])[0].split(",")
                    if x.strip()
                ]
                report = generate_candidates(
                    sequence=sequence,
                    editor=editor,
                    strands="both",
                    max_candidates=max_candidates,
                )
                if score_backends:
                    report = score_candidates(report, editor=editor, score_backends=score_backends, cas_variant=cas)
                self._send_html(HTML_PAGE.replace("__RESULT__", _candidate_report_html(report)))
                return
            result = analyze(
                sequence=sequence,
                editor=editor,
                task=task,
                run_models=run_models,
                cell_context=cell,
                cas_variant=cas,
                reference_outcome=outcome,
            )
            self._send_html(HTML_PAGE.replace("__RESULT__", _result_html(result)))
            return
        self._send_html("<h1>404</h1>", status=404)

    def log_message(self, format: str, *args) -> None:
        print("%s - - %s" % (self.address_string(), format % args))


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"BE prediction router running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
