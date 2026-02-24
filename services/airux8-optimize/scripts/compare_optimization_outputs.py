"""
Compare optimization outputs: legacy vs updated (model-based) logic.

Loads two zone-schedule CSVs (wide format), aligns by datetime and zone,
and compares hourly AC setting contents per zone (set_temp, mode, fan_speed,
numb_units_on, ac_on_off) and power consumption only.

Usage (from services/airux8-optimize):
  uv run python scripts/compare_optimization_outputs.py \\
    --legacy data/04_PlanningData/Clea/zone_schedule_20260218_20260221_fallback.csv \\
    --updated data/04_PlanningData/Clea/zone_schedule_20260218_20260221_model.csv

  uv run python scripts/compare_optimization_outputs.py \\
    --legacy path/to/fallback.csv --updated path/to/model.csv \\
    --output-dir data/04_PlanningData/Clea --report

  With graphs and tables (HTML report):
  uv run python scripts/compare_optimization_outputs.py \\
    --legacy fallback.csv --updated model.csv --output-dir ./out --html

  This writes comparison_report.html (data embedded; open in browser directly)
  and comparison_data.json (same data, for reference).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Per-zone columns to compare: hourly AC settings + power only (full column name is {zone}_{suffix})
COMPARE_SUFFIXES = [
    "set_temp",
    "mode",
    "fan_speed",
    "numb_units_on",
    "ac_on_off",
    "power",
]

# Float columns: compare with tolerance
FLOAT_SUFFIXES = {"set_temp", "power"}
FLOAT_TOLERANCE = 1e-3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare legacy vs updated optimization output CSVs (wide format)."
    )
    p.add_argument(
        "--legacy",
        required=True,
        help="Path to legacy (fallback) logic output CSV.",
    )
    p.add_argument(
        "--updated",
        required=True,
        help="Path to updated (model-based) logic output CSV.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="If set, write comparison_report.csv and comparison_summary.txt here.",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Write diff report files (requires --output-dir or writes to cwd).",
    )
    p.add_argument(
        "--zones",
        default=None,
        help="Comma-separated zone names to compare (default: all zones present in both files).",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with tables and graphs (writes to --output-dir or current dir).",
    )
    p.add_argument(
        "--diff-table-rows",
        type=int,
        default=200,
        help="Max rows of differences to show in HTML table (default: 200).",
    )
    return p.parse_args()


def _infer_zones(df: pd.DataFrame) -> list[str]:
    """Infer zone names from wide columns like 'Area 1_set_temp', 'Meeting Room_mode', ..."""
    zones = set()
    for c in df.columns:
        for suf in COMPARE_SUFFIXES:
            suffix = "_" + suf
            if c.endswith(suffix):
                zone = c[: -len(suffix)]
                # Exclude non-zone columns (e.g. Area 1_hist_indoor_temp -> zone Area 1_hist)
                if zone and not zone.endswith("_hist") and zone not in ("forecast_outdoor", "forecast_solar"):
                    zones.add(zone)
                break
    return sorted(zones)


def _compare_values(
    legacy_val: object,
    updated_val: object,
    suffix: str,
) -> bool:
    """Return True if values are considered equal."""
    if pd.isna(legacy_val) and pd.isna(updated_val):
        return True
    if pd.isna(legacy_val) or pd.isna(updated_val):
        return False
    if suffix in FLOAT_SUFFIXES:
        try:
            l = float(legacy_val)
            u = float(updated_val)
            return abs(l - u) <= FLOAT_TOLERANCE
        except (TypeError, ValueError):
            return str(legacy_val).strip() == str(updated_val).strip()
    return str(legacy_val).strip() == str(updated_val).strip()


def run_comparison(
    legacy_path: str | Path,
    updated_path: str | Path,
    zones_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load both CSVs, merge on datetime, compare per-zone columns.

    Returns:
        (merged_df, diff_rows_df, stats_dict)
        - merged_df: one row per datetime, with legacy_* and updated_* for each compared column (for zones present).
        - diff_rows_df: long format rows where at least one field differs (datetime, zone, field, legacy_value, updated_value).
        - stats_dict: summary counts and totals.
    """
    legacy_df = pd.read_csv(legacy_path)
    updated_df = pd.read_csv(updated_path)

    legacy_df["datetime"] = pd.to_datetime(legacy_df["datetime"])
    updated_df["datetime"] = pd.to_datetime(updated_df["datetime"])

    zones_legacy = _infer_zones(legacy_df)
    zones_updated = _infer_zones(updated_df)
    zones = sorted(set(zones_legacy) & set(zones_updated))
    if zones_filter:
        zones = [z for z in zones if z in zones_filter]
    if not zones:
        raise ValueError(
            "No common zones found between the two files. "
            f"Legacy zones: {zones_legacy}, Updated zones: {zones_updated}"
        )

    # Align on datetime (inner join)
    merged = legacy_df[["datetime"]].copy()
    merged = merged.merge(
        updated_df[["datetime"]],
        on="datetime",
        how="inner",
    ).drop_duplicates()
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Build merged with legacy_ and updated_ prefixed columns for compare fields
    for zone in zones:
        for suf in COMPARE_SUFFIXES:
            lcol = f"{zone}_{suf}"
            ucol = f"{zone}_{suf}"
            if lcol not in legacy_df.columns or ucol not in updated_df.columns:
                continue
            lseries = legacy_df.set_index("datetime")[lcol].reindex(merged["datetime"])
            useries = updated_df.set_index("datetime")[ucol].reindex(merged["datetime"])
            merged[f"{zone}_{suf}_legacy"] = lseries.values
            merged[f"{zone}_{suf}_updated"] = useries.values

    # Diff rows (long format: datetime, zone, field, legacy_value, updated_value)
    diff_rows = []
    for _, row in merged.iterrows():
        dt = row["datetime"]
        for zone in zones:
            for suf in COMPARE_SUFFIXES:
                lcol = f"{zone}_{suf}_legacy"
                ucol = f"{zone}_{suf}_updated"
                if lcol not in merged.columns or ucol not in merged.columns:
                    continue
                lv = row[lcol]
                uv = row[ucol]
                if not _compare_values(lv, uv, suf):
                    diff_rows.append(
                        {
                            "datetime": dt,
                            "zone": zone,
                            "field": suf,
                            "legacy_value": lv,
                            "updated_value": uv,
                        }
                    )

    diff_df = pd.DataFrame(diff_rows)

    # Stats
    total_rows = len(merged)
    rows_with_any_diff = 0
    if not diff_df.empty:
        rows_with_any_diff = merged["datetime"].isin(diff_df["datetime"].unique()).sum()

    total_power_legacy = 0.0
    total_power_updated = 0.0
    for zone in zones:
        pcol_l = f"{zone}_power_legacy"
        pcol_u = f"{zone}_power_updated"
        if pcol_l in merged.columns:
            total_power_legacy += merged[pcol_l].fillna(0).astype(float).sum()
        if pcol_u in merged.columns:
            total_power_updated += merged[pcol_u].fillna(0).astype(float).sum()

    stats = {
        "total_rows": total_rows,
        "rows_with_any_diff": rows_with_any_diff,
        "total_diffs": len(diff_df),
        "zones": zones,
        "total_power_legacy": total_power_legacy,
        "total_power_updated": total_power_updated,
        "power_delta": total_power_updated - total_power_legacy,
        "diff_by_zone": diff_df.groupby("zone").size().to_dict() if not diff_df.empty else {},
        "diff_by_field": diff_df.groupby("field").size().to_dict() if not diff_df.empty else {},
    }

    return merged, diff_df, stats


def _json_serializer(obj):
    """Convert numpy/pandas types and NaN for JSON."""
    if hasattr(obj, "item"):
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if pd.isna(obj):
        return None
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_default(o):
    """Used as default= in json.dumps for numpy/NaN/datetime. Never return float nan/inf."""
    if hasattr(o, "item"):
        v = o.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _make_json_safe(obj):
    """Recursively convert payload so json.dumps never sees nan/inf or numpy types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "item"):
        return _make_json_safe(obj.item())
    if pd.isna(obj):
        return None
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # numpy integer types etc.
    try:
        return int(obj)
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_report_payload(
    merged: pd.DataFrame,
    diff_df: pd.DataFrame,
    stats: dict,
    legacy_path: Path,
    updated_path: Path,
    chart_total_power: dict,
    chart_power_by_zone: dict,
    chart_units_by_zone: dict,
    chart_settemp_by_zone: dict,
    max_diff_table_rows: int,
) -> dict:
    """Build a single JSON-serializable payload for the report viewer."""
    diff_table = []
    if not diff_df.empty:
        sample = diff_df.head(max_diff_table_rows)
        for _, row in sample.iterrows():
            diff_table.append({
                "datetime": _json_serializer(row["datetime"]) if pd.notna(row["datetime"]) else None,
                "zone": str(row["zone"]),
                "field": str(row["field"]),
                "legacy_value": _json_serializer(row["legacy_value"]) if pd.notna(row["legacy_value"]) else str(row["legacy_value"]),
                "updated_value": _json_serializer(row["updated_value"]) if pd.notna(row["updated_value"]) else str(row["updated_value"]),
            })

    summary = [
        {"metric": "Total power (Legacy)", "value": f"{stats['total_power_legacy']:,.0f}"},
        {"metric": "Total power (Updated)", "value": f"{stats['total_power_updated']:,.0f}"},
        {"metric": "Power delta (Updated − Legacy)", "value": f"{stats['power_delta']:+,.0f}"},
    ]
    by_zone = [{"zone": z, "differences": stats["diff_by_zone"].get(z, 0)} for z in stats["zones"]]
    by_field = [{"field": f, "differences": stats["diff_by_field"].get(f, 0)} for f in COMPARE_SUFFIXES]

    return {
        "meta": {
            "legacy_path": str(legacy_path),
            "updated_path": str(updated_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "zones": stats["zones"],
        },
        "summary": summary,
        "diff_by_zone": by_zone,
        "diff_by_field": by_field,
        "diff_table": diff_table,
        "diff_table_note": f"Showing up to {max_diff_table_rows} rows. Full list in comparison_report.csv.",
        "charts": {
            "total_power": chart_total_power,
            "power_by_zone": chart_power_by_zone,
            "units_by_zone": chart_units_by_zone,
            "settemp_by_zone": chart_settemp_by_zone,
        },
    }


def _aggregate_merged_to_hourly(merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate merged to one row per hour; power summed, other numeric fields averaged. Datetime = start of hour (one block per hour)."""
    if merged.empty or "datetime" not in merged.columns:
        return merged
    m = merged.copy()
    m["_hour"] = pd.to_datetime(m["datetime"]).dt.floor("h")
    agg = {}
    for c in m.columns:
        if c in ("datetime", "_hour"):
            continue
        if not pd.api.types.is_numeric_dtype(m[c]):
            continue
        agg[c] = "sum" if "_power_" in c else "mean"
    if not agg:
        return merged
    hourly = m[["_hour"] + list(agg.keys())].groupby("_hour", as_index=False).agg(agg)
    # Label = start of hour (00:00, 01:00, ...) so one block = one hour
    hourly["datetime"] = hourly["_hour"]
    return hourly.drop(columns=["_hour"])


def _chart_data_total_power(merged: pd.DataFrame, zones: list[str]) -> dict:
    """Chart.js data: total power (sum over zones) legacy vs updated. One bar per hour; label = start of hour."""
    if "datetime" not in merged.columns or not zones:
        return {"labels": [], "datasets": []}
    hourly = _aggregate_merged_to_hourly(merged)
    labels = hourly["datetime"].astype(str).tolist()
    total_legacy = pd.Series(0.0, index=hourly.index)
    total_updated = pd.Series(0.0, index=hourly.index)
    for z in zones:
        lcol = f"{z}_power_legacy"
        ucol = f"{z}_power_updated"
        if lcol in hourly.columns:
            total_legacy = total_legacy + hourly[lcol].fillna(0).astype(float)
        if ucol in hourly.columns:
            total_updated = total_updated + hourly[ucol].fillna(0).astype(float)
    datasets = [
        {"label": "Legacy", "data": total_legacy.tolist(), "backgroundColor": "rgba(59, 130, 246, 0.85)", "borderColor": "rgb(37, 99, 235)", "borderWidth": 1},
        {"label": "Updated", "data": total_updated.tolist(), "backgroundColor": "rgba(20, 184, 166, 0.85)", "borderColor": "rgb(13, 148, 136)", "borderWidth": 1},
    ]
    return {"labels": labels, "datasets": datasets}


def _chart_data_by_zone(
    merged: pd.DataFrame,
    zones: list[str],
    value_suffix: str,
) -> dict:
    """Chart.js data: per-zone legacy vs updated (power, numb_units_on, set_temp). One bar per hour; label = start of hour."""
    if "datetime" not in merged.columns or not zones:
        return {"labels": [], "zones": []}
    hourly = _aggregate_merged_to_hourly(merged)
    labels = hourly["datetime"].astype(str).tolist()
    result = {"labels": labels, "zones": []}
    for zone in zones:
        lcol = f"{zone}_{value_suffix}_legacy"
        ucol = f"{zone}_{value_suffix}_updated"
        legacy_vals = hourly[lcol].fillna(0).astype(float).tolist() if lcol in hourly.columns else []
        updated_vals = hourly[ucol].fillna(0).astype(float).tolist() if ucol in hourly.columns else []
        result["zones"].append({
            "zone": zone,
            "datasets": [
                {"label": "Legacy", "data": legacy_vals, "backgroundColor": "rgba(59, 130, 246, 0.85)", "borderColor": "rgb(37, 99, 235)", "borderWidth": 1},
                {"label": "Updated", "data": updated_vals, "backgroundColor": "rgba(20, 184, 166, 0.85)", "borderColor": "rgb(13, 148, 136)", "borderWidth": 1},
            ],
        })
    return result


def _viewer_html(embedded_data_json: str) -> str:
    """Return report HTML with comparison data embedded so it works when opened from file."""
    # Embed JSON in a script tag; escape </script> so it does not close the tag
    escaped_json = embedded_data_json.replace("</script>", "<\\/script>").replace("</", "<\\/")
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Optimization comparison report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: sans-serif; margin: 1.5rem; background: #f8f9fa; }
    h1 { color: #1a1a2e; }
    h2 { margin-top: 2rem; color: #16213e; }
    table { border-collapse: collapse; margin: 0.5rem 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }
    th { background: #e9ecef; }
    .chart-container { position: relative; width: 90vw; height: 400px; margin: 1rem 0; box-sizing: border-box; }
    .chart-container canvas { width: 100% !important; height: 400px !important; display: block; }
    .chart-grid { display: flex; flex-direction:column; gap: 1.5rem; }
    .meta { color: #666; font-size: 0.9rem; margin-bottom: 1rem; }
    .zone-select { margin: 0.5rem 0 1rem 0; font-size: 1rem; padding: 0.35rem 0.5rem; }
    #loading { margin: 2rem; }
    #error { color: #c00; margin: 2rem; }
  </style>
</head>
<body>
  <div id="loading">Loading…</div>
  <div id="error" style="display:none;"></div>
  <div id="report" style="display:none;"></div>
  <script id="comparison-data" type="application/json">""" + escaped_json + """</script>
  <script>
(function() {
  function formatTimeAxisLabel(val, index, ticks) {
    var chart = this.chart;
    if (!chart || !chart.data || !chart.data.labels) return val;
    var lbl = chart.data.labels[val];
    if (!lbl) return val;
    var d = new Date(lbl);
    if (isNaN(d.getTime())) return lbl;
    var h = d.getHours(), m = d.getMinutes();
    /* Midnight (00:00) = show date; else show hour (e.g. 20:00, 04:00) */
    if (h === 0 && m === 0) return (d.getMonth() + 1) + '/' + d.getDate();
    return ('0' + h).slice(-2) + ':00';
  }
  var chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: {
        ticks: { maxRotation: 45, callback: formatTimeAxisLabel, autoSkip: false },
        grid: { display: true, drawOnChartArea: true, color: 'rgba(0,0,0,0.12)', borderDash: [4, 4] },
        offset: false
      },
      y: { beginAtZero: true, grid: { display: true, color: 'rgba(0,0,0,0.12)', borderDash: [4, 4] } }
    },
    datasets: { bar: { barPercentage: 0.9, categoryPercentage: 1 } }
  };
  var chartSingleZonePower = null;

  function escapeHtml(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
  function renderTable(headers, rows) {
    var h = '<table><thead><tr>';
    headers.forEach(function(c) { h += '<th>' + escapeHtml(c) + '</th>'; });
    h += '</tr></thead><tbody>';
    rows.forEach(function(row) {
      h += '<tr>';
      headers.forEach(function(c) { h += '<td>' + escapeHtml(String(row[c] != null ? row[c] : '')) + '</td>'; });
      h += '</tr>';
    });
    return h + '</tbody></table>';
  }

  function drawCharts(data) {
    var c = data.charts || {};
    if (c.total_power && c.total_power.labels && c.total_power.labels.length) {
      var el = document.getElementById('chartTotalPower');
      if (el && typeof Chart !== 'undefined') new Chart(el, { type: 'bar', data: c.total_power, options: chartOptions });
    }
    var gridMap = { power_by_zone: 'chartPowerByZoneGrid', units_by_zone: 'chartUnitsByZoneGrid' };
    ['power_by_zone', 'units_by_zone'].forEach(function(key) {
      var chartData = c[key];
      var grid = document.getElementById(gridMap[key]);
      if (!grid || !chartData || !chartData.zones || !chartData.zones.length) return;
      chartData.zones.forEach(function(z) {
        var wrap = document.createElement('div');
        wrap.className = 'chart-container';
        wrap.setAttribute('data-zone', z.zone);
        var canvas = document.createElement('canvas'); wrap.appendChild(canvas); grid.appendChild(wrap);
        var opts = Object.assign({}, chartOptions, { plugins: { legend: { position: 'top' }, title: { display: true, text: z.zone } } });
        new Chart(canvas, { type: 'bar', data: { labels: chartData.labels, datasets: z.datasets }, options: opts });
      });
    });
  }

  function applyZoneFilter(selectedZone, data) {
    var totalDiv = document.getElementById('totalPowerContainer');
    var singleDiv = document.getElementById('singleZonePowerContainer');
    var zones = document.querySelectorAll('.chart-container[data-zone]');
    if (selectedZone === '') {
      if (totalDiv) totalDiv.style.display = 'block';
      if (singleDiv) singleDiv.style.display = 'none';
      zones.forEach(function(el) { el.style.display = 'block'; });
    } else {
      if (totalDiv) totalDiv.style.display = 'none';
      if (singleDiv) singleDiv.style.display = 'block';
      zones.forEach(function(el) { el.style.display = el.getAttribute('data-zone') === selectedZone ? 'block' : 'none'; });
      var c = data.charts || {};
      var chartData = c.power_by_zone;
      if (chartData && chartData.zones) {
        var zoneObj = chartData.zones.filter(function(z) { return z.zone === selectedZone; })[0];
        if (zoneObj) {
          var canvas = document.getElementById('chartSingleZonePower');
          if (canvas && typeof Chart !== 'undefined') {
            if (chartSingleZonePower) {
              chartSingleZonePower.data.labels = chartData.labels;
              chartSingleZonePower.data.datasets = zoneObj.datasets;
              chartSingleZonePower.update();
            } else {
              chartSingleZonePower = new Chart(canvas, { type: 'bar', data: { labels: chartData.labels, datasets: zoneObj.datasets }, options: chartOptions });
            }
          }
        }
      }
    }
  }

  function render(d) {
    var meta = d.meta || {};
    var zones = meta.zones || [];
    var html = '<h1>Optimization comparison: Legacy vs Updated</h1>';
    html += '<div class="meta"><b>Legacy:</b> ' + escapeHtml(meta.legacy_path || '') + '<br><b>Updated:</b> ' + escapeHtml(meta.updated_path || '') + '</div>';
    html += '<p><label>Zone: <select id="zoneSelect" class="zone-select"><option value="">All zones</option></select></label></p>';
    html += '<h2>Summary</h2>';
    html += renderTable(['metric', 'value'], (d.summary || []).map(function(r) { return { metric: r.metric, value: r.value }; }));
    html += '<h2>Total power over time (all zones)</h2><div id="totalPowerContainer" class="chart-container"><canvas id="chartTotalPower"></canvas></div>';
    html += '<h2>Power (selected zone)</h2><div id="singleZonePowerContainer" class="chart-container" style="display:none;"><canvas id="chartSingleZonePower"></canvas></div>';
    html += '<h2>Power by zone over time</h2><div class="chart-grid" id="chartPowerByZoneGrid"></div>';
    html += '<h2>AC units on by zone over time</h2><div class="chart-grid" id="chartUnitsByZoneGrid"></div>';
    document.getElementById('report').innerHTML = html;
    var sel = document.getElementById('zoneSelect');
    zones.forEach(function(z) {
      var opt = document.createElement('option');
      opt.value = z;
      opt.textContent = z;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', function() { applyZoneFilter(this.value, d); });
    document.getElementById('report').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    drawCharts(d);
  }

  var dataEl = document.getElementById('comparison-data');
  var d = dataEl ? JSON.parse(dataEl.textContent) : null;
  if (d) { render(d); }
  else {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
    document.getElementById('error').textContent = 'No comparison data. Run compare_optimization_outputs.py with --html first.';
  }
})();
  </script>
</body>
</html>
"""


def _write_html_report(
    merged: pd.DataFrame,
    diff_df: pd.DataFrame,
    stats: dict,
    legacy_path: Path,
    updated_path: Path,
    output_dir: Path,
    max_diff_table_rows: int = 200,
) -> Path:
    """Write comparison_report.html with data embedded (works when opened from file). Optionally write comparison_data.json."""
    zones = stats["zones"]
    chart_total_power = _chart_data_total_power(merged, zones)
    chart_power_by_zone = _chart_data_by_zone(merged, zones, "power")
    chart_units_by_zone = _chart_data_by_zone(merged, zones, "numb_units_on")
    chart_settemp_by_zone = _chart_data_by_zone(merged, zones, "set_temp")

    payload = _build_report_payload(
        merged,
        diff_df,
        stats,
        legacy_path,
        updated_path,
        chart_total_power,
        chart_power_by_zone,
        chart_units_by_zone,
        chart_settemp_by_zone,
        max_diff_table_rows,
    )

    payload_safe = _make_json_safe(payload)
    try:
        payload_json = json.dumps(payload_safe, default=_json_default)
    except (TypeError, ValueError) as e:
        # If serialization fails, recursively replace any remaining float nan/inf and retry
        def _fix_floats(obj):
            if isinstance(obj, dict):
                return {k: _fix_floats(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_fix_floats(v) for v in obj]
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        payload_safe = _fix_floats(payload_safe)
        payload_json = json.dumps(payload_safe, default=_json_default)
    html_path = output_dir / "comparison_report.html"
    html_path.write_text(_viewer_html(payload_json), encoding="utf-8")

    json_path = output_dir / "comparison_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_safe, f, default=_json_default, indent=2)

    return html_path


def main() -> int:
    args = _parse_args()
    legacy_path = Path(args.legacy)
    updated_path = Path(args.updated)

    if not legacy_path.exists():
        print(f"Error: Legacy file not found: {legacy_path}", file=sys.stderr)
        return 1
    if not updated_path.exists():
        print(f"Error: Updated file not found: {updated_path}", file=sys.stderr)
        return 1

    zones_filter = None
    if args.zones:
        zones_filter = [z.strip() for z in args.zones.split(",") if z.strip()]

    try:
        merged, diff_df, stats = run_comparison(
            legacy_path, updated_path, zones_filter=zones_filter
        )
    except Exception as e:
        print(f"Error during comparison: {e}", file=sys.stderr)
        return 1

    # Console summary
    print("=" * 60)
    print("Optimization output comparison (Legacy vs Updated)")
    print("=" * 60)
    print(f"Legacy:  {legacy_path}")
    print(f"Updated: {updated_path}")
    print(f"Zones:   {stats['zones']}")
    print(f"Rows (aligned): {stats['total_rows']}")
    print(f"Rows with at least one difference: {stats['rows_with_any_diff']}")
    print(f"Total field-level differences: {stats['total_diffs']}")
    print()
    print("Total power (sum over all zones):")
    print(f"  Legacy:  {stats['total_power_legacy']:,.0f}")
    print(f"  Updated: {stats['total_power_updated']:,.0f}")
    print(f"  Delta (updated - legacy): {stats['power_delta']:+,.0f}")
    if stats["diff_by_zone"]:
        print()
        print("Differences by zone:")
        for z in stats["zones"]:
            print(f"  {z}: {stats['diff_by_zone'].get(z, 0)}")
    if stats["diff_by_field"]:
        print()
        print("Differences by field:")
        for f in COMPARE_SUFFIXES:
            print(f"  {f}: {stats['diff_by_field'].get(f, 0)}")
    print("=" * 60)

    out_dir = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    elif args.report or args.html:
        out_dir = Path.cwd()

    if args.html:
        if out_dir is None:
            out_dir = Path.cwd()
        try:
            html_path = _write_html_report(
                merged,
                diff_df,
                stats,
                legacy_path,
                updated_path,
                out_dir,
                max_diff_table_rows=args.diff_table_rows,
            )
            print(f"Wrote HTML report: {html_path}")
        except Exception as e:
            print(f"Error generating HTML report: {e}", file=sys.stderr)
            return 1
        # Write CSV so "full list in comparison_report.csv" in HTML is valid
        if not diff_df.empty:
            diff_df.to_csv(out_dir / "comparison_report.csv", index=False)

    if args.report and out_dir is not None:
        summary_path = out_dir / "comparison_summary.txt"
        report_path = out_dir / "comparison_report.csv"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Optimization output comparison (Legacy vs Updated)\n")
            f.write(f"Legacy: {legacy_path}\n")
            f.write(f"Updated: {updated_path}\n")
            f.write(f"Rows: {stats['total_rows']}, Rows with diff: {stats['rows_with_any_diff']}, Total diffs: {stats['total_diffs']}\n")
            f.write(f"Power legacy: {stats['total_power_legacy']:.0f}, updated: {stats['total_power_updated']:.0f}, delta: {stats['power_delta']:+.0f}\n")
        if not diff_df.empty:
            diff_df.to_csv(report_path, index=False)
            print(f"Wrote diff report: {report_path}")
        print(f"Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
