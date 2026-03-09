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
    --output-dir data/04_PlanningData/Clea

  Writes comparison_report.html (data embedded; open in browser directly).
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
        help="Directory for comparison_report.html (default: current directory).",
    )
    p.add_argument(
        "--zones",
        default=None,
        help="Comma-separated zone names to compare (default: all zones present in both files).",
    )
    p.add_argument(
        "--diff-table-rows",
        type=int,
        default=200,
        help="Max rows of differences to show in HTML table (default: 200).",
    )
    return p.parse_args()


# Default 許容範囲 when master is not loaded (目標室内温度下限–上限)
DEFAULT_COMFORT_MIN, DEFAULT_COMFORT_MAX = 24.0, 26.0


def _load_comfort_range_from_master(master_path: Path, zones: list[str]) -> dict[str, list[dict]]:
    """
    Load 目標室内温度下限 and 目標室内温度上限 from 制御マスタ sheet.
    Returns dict zone -> list of 12 {min, max} for months 1–12 (index 0 = January).
    Missing zone/month use DEFAULT_COMFORT_MIN, DEFAULT_COMFORT_MAX.
    """
    out: dict[str, list[dict]] = {}
    try:
        control = pd.read_excel(master_path, sheet_name="制御マスタ", engine="openpyxl")
    except Exception as e:
        print(f"Warning: Could not read 制御マスタ from {master_path}: {e}", file=sys.stderr)
        return out
    required = ["制御区分", "月", "目標室内温度下限", "目標室内温度上限"]
    for col in required:
        if col not in control.columns:
            print(f"Warning: 制御マスタ missing column '{col}', skipping comfort range.", file=sys.stderr)
            return out
    # 月 is e.g. "1月", "2月"
    month_order = [f"{m}月" for m in range(1, 13)]
    for zone in zones:
        by_month: list[dict] = []
        for month_jp in month_order:
            row = control[
                (control["制御区分"].astype(str).str.strip() == zone)
                & (control["月"].astype(str).str.strip() == month_jp)
            ]
            if len(row) == 0:
                by_month.append({"min": DEFAULT_COMFORT_MIN, "max": DEFAULT_COMFORT_MAX})
            else:
                r = row.iloc[0]
                min_t = r["目標室内温度下限"]
                max_t = r["目標室内温度上限"]
                if pd.isna(min_t) or pd.isna(max_t):
                    by_month.append({"min": DEFAULT_COMFORT_MIN, "max": DEFAULT_COMFORT_MAX})
                else:
                    by_month.append({"min": float(min_t), "max": float(max_t)})
        out[zone] = by_month
    return out


def _load_zone_to_units_from_master(master_path: Path, zones: list[str]) -> dict[str, list[str]]:
    """
    Load zone -> list of indoor unit IDs from MASTER sheet (制御区分, 環境予測区分).
    Area 2 is split into Area2_1 / Area2_2 by indoor unit list (same as MasterDataLoader).
    Returns dict zone -> list of 環境予測区分 for each zone in zones.
    """
    out: dict[str, list[str]] = {}
    try:
        master_df = pd.read_excel(master_path, sheet_name="MASTER", engine="openpyxl")
    except Exception as e:
        print(f"Warning: Could not read MASTER from {master_path}: {e}", file=sys.stderr)
        return out
    required = ["制御区分", "環境予測区分"]
    for col in required:
        if col not in master_df.columns:
            print(f"Warning: MASTER missing column '{col}', skipping zone-to-units.", file=sys.stderr)
            return out
    zone_set = set(zones)
    for _, row in master_df.iterrows():
        zone_name = row["制御区分"]
        indoor_unit = row["環境予測区分"]
        if pd.isna(zone_name) or pd.isna(indoor_unit):
            continue
        zone_name = str(zone_name).strip()
        indoor_unit_str = str(indoor_unit).strip()
        if zone_name in ("Area 2", "エリア2"):
            if indoor_unit_str in ("D-8北2", "D-6北1", "D-7南2", "D-5南1"):
                zone_name = "Area2_1"
            elif indoor_unit_str in ("D-4北2", "D-2北1"):
                zone_name = "Area2_2"
            else:
                continue
        if zone_name not in zone_set:
            continue
        if zone_name not in out:
            out[zone_name] = []
        if indoor_unit_str not in out[zone_name]:
            out[zone_name].append(indoor_unit_str)
    return out


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

    # Align on datetime (outer join = union) so no rows are dropped; missing side gets NaN
    all_dts = (
        pd.concat([legacy_df[["datetime"]], updated_df[["datetime"]]], ignore_index=True)
        .drop_duplicates()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    merged = all_dts.copy()

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

    # Outdoor temp and per-zone indoor temp (for charts only; not compared)
    if "forecast_outdoor_temp" in legacy_df.columns:
        merged["forecast_outdoor_temp"] = (
            legacy_df.set_index("datetime")["forecast_outdoor_temp"].reindex(merged["datetime"]).values
        )
    for zone in zones:
        icol = f"{zone}_indoor_temp"
        if icol in legacy_df.columns:
            merged[f"{zone}_indoor_temp_legacy"] = (
                legacy_df.set_index("datetime")[icol].reindex(merged["datetime"]).values
            )
        if icol in updated_df.columns:
            merged[f"{zone}_indoor_temp_updated"] = (
                updated_df.set_index("datetime")[icol].reindex(merged["datetime"]).values
            )

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


def _resolve_unit_schedule_path(zone_schedule_path: Path) -> Path:
    """From zone_schedule_*_historical.csv or zone_schedule_*_model.csv return unit_schedule_*_historical.csv or unit_schedule_*_model.csv."""
    name = zone_schedule_path.name
    if not name.startswith("zone_schedule_"):
        return zone_schedule_path.parent / name.replace("zone_schedule", "unit_schedule", 1)
    return zone_schedule_path.parent / name.replace("zone_schedule_", "unit_schedule_", 1)


def _unit_mode_from_schedule(
    unit_df: pd.DataFrame,
    unit_ids: list[str] | None = None,
) -> dict | None:
    """
    Build 台モード chart data from unit_schedule CSV (Date Time + *_Mode columns).
    If unit_ids is provided, only *_Mode columns for those units are counted (per-zone).
    Returns labels, totalUnits, coolingUnits, heatingUnits, fanUnits, unitModeData, unitModeColors, maxUnits.
    """
    if unit_df.empty:
        return None
    dt_col = "Date Time" if "Date Time" in unit_df.columns else "datetime"
    if dt_col not in unit_df.columns:
        return None
    mode_cols = [c for c in unit_df.columns if c.endswith("_Mode")]
    if unit_ids is not None and unit_ids:
        unit_set = set(u.strip() for u in unit_ids)
        mode_cols = [c for c in mode_cols if c.replace("_Mode", "") in unit_set]
    if not mode_cols:
        return None
    # Normalize mode: COOL, HEAT, FAN (OFF excluded from stack)
    def norm(s):
        if pd.isna(s):
            return None
        v = str(s).strip().upper()
        if v in ("COOL", "HEAT", "FAN"):
            return v
        return None
    unit_ids = [c.replace("_Mode", "") for c in mode_cols]
    labels = []
    cooling_units = []
    heating_units = []
    fan_units = []
    modes_per_hour: list[list[str]] = []
    for _, row in unit_df.iterrows():
        labels.append(str(row[dt_col]))
        modes = [norm(row[c]) for c in mode_cols]
        modes_per_hour.append([m if m else "OFF" for m in modes])
        cooling_units.append(sum(1 for m in modes if m == "COOL"))
        heating_units.append(sum(1 for m in modes if m == "HEAT"))
        fan_units.append(sum(1 for m in modes if m == "FAN"))
    total_units = [c + h + f for c, h, f in zip(cooling_units, heating_units, fan_units)]
    max_units = max(total_units) if total_units else 1
    max_units = max(max_units, 1)
    # unitModeData: for stack u (0..max_units-1), value at h is 1 if total_units[h] > u else 0
    unit_mode_data = []
    unit_mode_colors = []
    MODE_COLOR = {
        "cooling": "rgba(0, 163, 224, 0.45)",
        "heating": "rgba(230, 126, 34, 0.45)",
        "fan": "rgba(120, 120, 120, 0.45)",
    }
    for u in range(max_units):
        unit_mode_data.append([1 if total_units[h] > u else 0 for h in range(len(labels))])
        row_colors = []
        for h in range(len(labels)):
            if u >= total_units[h]:
                row_colors.append("transparent")
            elif u < cooling_units[h]:
                row_colors.append(MODE_COLOR["cooling"])
            elif u < cooling_units[h] + heating_units[h]:
                row_colors.append(MODE_COLOR["heating"])
            else:
                row_colors.append(MODE_COLOR["fan"])
        unit_mode_colors.append(row_colors)
    return {
        "labels": labels,
        "totalUnits": total_units,
        "coolingUnits": cooling_units,
        "heatingUnits": heating_units,
        "fanUnits": fan_units,
        "unitModeData": unit_mode_data,
        "unitModeColors": unit_mode_colors,
        "maxUnits": max_units,
        "unitIds": unit_ids,
        "modesPerHour": modes_per_hour,
    }


def _attach_temperature_to_unit_mode(
    unit_mode: dict,
    chart_temperature: dict,
    use_legacy: bool,
) -> None:
    """Attach outdoorTemp, indoorTemp, setTemp to unit_mode (same length as labels) for overlay. Mutates unit_mode."""
    if not unit_mode or not chart_temperature or not unit_mode.get("labels"):
        return
    labels = unit_mode["labels"]
    temp_labels = chart_temperature.get("labels") or []
    outdoor = chart_temperature.get("outdoor_temp") or []
    zones = chart_temperature.get("zones") or []
    zone_data = zones[0] if zones else {}
    if use_legacy:
        indoor = (zone_data.get("indoor_legacy") or [])
        set_temp = (zone_data.get("set_temp_legacy") or [])
    else:
        indoor = (zone_data.get("indoor_updated") or [])
        set_temp = (zone_data.get("set_temp_updated") or [])
    # Parse to comparable time (hour)
    def parse_label(s):
        try:
            return pd.Timestamp(s).floor("h")
        except Exception:
            return None
    temp_idx_by_hour = {}
    for i, lb in enumerate(temp_labels):
        t = parse_label(lb)
        if t is not None:
            temp_idx_by_hour[t] = i
    outdoor_temp = []
    indoor_temp = []
    set_temp_out = []
    for lb in labels:
        t = parse_label(lb)
        idx = temp_idx_by_hour.get(t) if t is not None else None
        if idx is not None:
            outdoor_temp.append(outdoor[idx] if idx < len(outdoor) else None)
            indoor_temp.append(indoor[idx] if idx < len(indoor) else None)
            set_temp_out.append(set_temp[idx] if idx < len(set_temp) else None)
        else:
            outdoor_temp.append(None)
            indoor_temp.append(None)
            set_temp_out.append(None)
    unit_mode["outdoorTemp"] = outdoor_temp
    unit_mode["indoorTemp"] = indoor_temp
    unit_mode["setTemp"] = set_temp_out


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
    chart_temperature: dict,
    comfort_range_by_zone_month: dict[str, list[dict]],
    max_diff_table_rows: int,
    unit_mode_legacy: dict | None = None,
    unit_mode_updated: dict | None = None,
    unit_mode_by_zone: dict[str, dict] | None = None,
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
        {"metric": "総電力（実績）", "value": f"{stats['total_power_legacy']:,.0f}"},
        {"metric": "総電力（モデル出力）", "value": f"{stats['total_power_updated']:,.0f}"},
        {"metric": "差分（モデル − 実績）", "value": f"{stats['power_delta']:+,.0f}"},
    ]
    by_zone = [{"zone": z, "differences": stats["diff_by_zone"].get(z, 0)} for z in stats["zones"]]
    by_field = [{"field": f, "differences": stats["diff_by_field"].get(f, 0)} for f in COMPARE_SUFFIXES]

    return {
        "meta": {
            "legacy_path": str(legacy_path),
            "updated_path": str(updated_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "zones": stats["zones"],
            "facility": legacy_path.parent.name,
        },
        "summary": summary,
        "diff_by_zone": by_zone,
        "diff_by_field": by_field,
        "diff_table": diff_table,
        "diff_table_note": f"Showing up to {max_diff_table_rows} rows.",
        "temperature": chart_temperature,
        "comfort_range_by_zone_month": comfort_range_by_zone_month,
        "charts": {
            "total_power": chart_total_power,
            "power_by_zone": chart_power_by_zone,
            "units_by_zone": chart_units_by_zone,
            "settemp_by_zone": chart_settemp_by_zone,
        },
        "unit_mode_legacy": unit_mode_legacy,
        "unit_mode_updated": unit_mode_updated,
        "unit_mode_by_zone": unit_mode_by_zone,
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
    hourly = hourly.drop(columns=["_hour"])
    # Reindex to full hourly range so every hour has a bar (fill missing with 0)
    hour_range = pd.date_range(
        start=hourly["datetime"].min(),
        end=hourly["datetime"].max(),
        freq="h",
        inclusive="both",
    )
    # Reindex to full hourly range; fill power with 0, leave others NaN for chart gaps
    hourly = hourly.set_index("datetime").reindex(hour_range).reset_index()
    hourly = hourly.rename(columns={"index": "datetime"})
    for c in hourly.columns:
        if c == "datetime":
            continue
        if c in agg and agg[c] == "sum":
            hourly[c] = hourly[c].fillna(0)
    return hourly


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
    # Convert Wh to kWh for chart (y-axis label is 電力消費量 kWh)
    kWh_legacy = (total_legacy / 1000.0).tolist()
    kWh_updated = (total_updated / 1000.0).tolist()
    datasets = [
        {"label": "Legacy", "data": kWh_legacy, "backgroundColor": "rgba(59, 130, 246, 0.85)", "borderColor": "rgb(37, 99, 235)", "borderWidth": 1},
        {"label": "Updated", "data": kWh_updated, "backgroundColor": "rgba(20, 184, 166, 0.85)", "borderColor": "rgb(13, 148, 136)", "borderWidth": 1},
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


def _chart_data_temperature(merged: pd.DataFrame, zones: list[str]) -> dict:
    """Chart data for 室温 and 温度推移: labels, outdoor_temp, per-zone indoor/set_temp/units_on (legacy & updated)."""
    if "datetime" not in merged.columns or not zones:
        return {"labels": [], "outdoor_temp": [], "zones": []}
    hourly = _aggregate_merged_to_hourly(merged)
    labels = hourly["datetime"].astype(str).tolist()
    if "forecast_outdoor_temp" in hourly.columns:
        outdoor = [None if pd.isna(v) else float(v) for v in hourly["forecast_outdoor_temp"]]
    else:
        outdoor = [None] * len(labels) if labels else []
    zone_list = []
    for zone in zones:
        def _series(col: str) -> list:
            if col not in hourly.columns:
                return [None] * len(labels)
            return [None if pd.isna(v) else float(v) for v in hourly[col]]

        zone_list.append({
            "zone": zone,
            "indoor_legacy": _series(f"{zone}_indoor_temp_legacy"),
            "indoor_updated": _series(f"{zone}_indoor_temp_updated"),
            "set_temp_legacy": _series(f"{zone}_set_temp_legacy"),
            "set_temp_updated": _series(f"{zone}_set_temp_updated"),
            "units_on_legacy": _series(f"{zone}_numb_units_on_legacy"),
            "units_on_updated": _series(f"{zone}_numb_units_on_updated"),
        })
    return {"labels": labels, "outdoor_temp": outdoor, "zones": zone_list}


def _viewer_html(embedded_data_json: str) -> str:
    """Return report HTML with comparison data embedded (温度推移℃, 電力消費量 kWh). 営業時間 8–20, 許容範囲 24–26°C default."""
    escaped_json = embedded_data_json.replace("</script>", "<\\/script>").replace("</", "<\\/")
    return """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>温度・電力 比較ダッシュボード | 実績 vs モデル出力</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: sans-serif; background: #f5f5f5; display: flex; flex-direction: column; align-items: center; padding: 20px; margin: 0; }
    .chart-container { width: 1100px; max-width: 100%; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 20px; margin-top: 20px; }
    .title-section { display: flex; align-items: baseline; gap: 20px; margin-bottom: 10px; flex-wrap: wrap; }
    .title-section h2 { margin: 0; font-size: 1.5rem; font-weight: normal; }
    .legend-sample { display: flex; gap: 20px; font-size: 0.85rem; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .color-dot { width: 14px; height: 14px; border-radius: 2px; }
    .chart-wrapper { position: relative; width: 100%; height: 280px; }
    .chart-wrapper.chart-wrapper-power { height: 180px; }
    .chart-wrapper.chart-wrapper-unit-mode { height: 280px; }
    .chart-wrapper canvas { display: block; width: 100% !important; height: 100% !important; }
    .unit-mode-tooltip { position: fixed; pointer-events: none; z-index: 1000; background: rgba(40,40,40,0.96); color: #eee; padding: 8px 10px; border-radius: 6px; font-size: 12px; line-height: 1.4; box-shadow: 0 2px 10px rgba(0,0,0,0.35); white-space: nowrap; display: none; }
    .unit-mode-tooltip .tooltip-time { font-weight: bold; margin-bottom: 6px; color: #fff; }
    .unit-mode-tooltip .tooltip-section { margin-bottom: 6px; }
    .unit-mode-tooltip .tooltip-section-title { font-weight: bold; color: #27ae60; margin-bottom: 2px; }
    .unit-mode-tooltip .tooltip-section-title.model { color: #3498db; }
    .unit-mode-tooltip .tooltip-line { padding: 1px 0; }
    .meta { color: #666; font-size: 0.9rem; margin-bottom: 1rem; max-width: 1100px; }
    .zone-select { margin: 0.5rem 0 1rem 0; font-size: 1rem; padding: 0.35rem 0.5rem; }
    table { border-collapse: collapse; margin: 0.5rem 0; background: white; max-width: 1100px; }
    th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }
    th { background: #e9ecef; }
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
  'use strict';
  var chartInstances = [];

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

  function hourFromLabel(labels, i) {
    if (!labels || i < 0 || i >= labels.length) return -1;
    var d = new Date(labels[i]);
    return isNaN(d.getTime()) ? -1 : d.getHours();
  }
  function inBusinessHour(hour) { return hour >= 8 && hour <= 20; }

  var defaultComfort = { min: 24, max: 26 };
  function getMonthFromLabel(labels, i) {
    if (!labels || i < 0 || i >= labels.length) return 1;
    var d = new Date(labels[i]);
    return isNaN(d.getTime()) ? 1 : d.getMonth() + 1;
  }
  var customBackgroundPlugin = {
    id: 'customBackground',
    beforeDraw: function(chart) {
      if (!chart.config || chart.config.type !== 'line') return;
      var ctx = chart.ctx, xAxis = chart.scales.x, yAxisTemp = chart.scales.y1 || chart.scales.y;
      if (!xAxis || !yAxisTemp) return;
      var chartArea = chart.chartArea;
      if (!chartArea) return;
      var labels = chart.data.labels || [];
      var comfortByMonth = (chart.options.plugins && chart.options.plugins.customBackground && chart.options.plugins.customBackground.comfortByMonth) || [];
      ctx.save();
      ctx.beginPath();
      ctx.rect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
      ctx.clip();
      for (var start = 0; start < labels.length; start++) {
        if (!inBusinessHour(hourFromLabel(labels, start))) continue;
        var end = start;
        while (end < labels.length && inBusinessHour(hourFromLabel(labels, end))) end++;
        var left = start === 0 ? 2 * xAxis.getPixelForValue(0) - xAxis.getPixelForValue(1) : (xAxis.getPixelForValue(start - 1) + xAxis.getPixelForValue(start)) / 2;
        var right = end >= labels.length ? 2 * xAxis.getPixelForValue(labels.length - 1) - xAxis.getPixelForValue(labels.length - 2) : (xAxis.getPixelForValue(end - 1) + xAxis.getPixelForValue(end)) / 2;
        ctx.fillStyle = 'rgba(46, 204, 113, 0.1)';
        ctx.fillRect(left, yAxisTemp.top, right - left, yAxisTemp.bottom - yAxisTemp.top);
        start = end - 1;
      }
      var i = 0;
      while (i < labels.length) {
        var month = getMonthFromLabel(labels, i);
        var comfort = (comfortByMonth[month - 1] != null) ? comfortByMonth[month - 1] : defaultComfort;
        var minV = comfort.min != null ? comfort.min : defaultComfort.min;
        var maxV = comfort.max != null ? comfort.max : defaultComfort.max;
        var end = i + 1;
        while (end < labels.length) {
          var m2 = getMonthFromLabel(labels, end);
          var c2 = (comfortByMonth[m2 - 1] != null) ? comfortByMonth[m2 - 1] : defaultComfort;
          if (c2.min !== minV || c2.max !== maxV) break;
          end++;
        }
        var left = i === 0 ? 2 * xAxis.getPixelForValue(0) - xAxis.getPixelForValue(1) : (xAxis.getPixelForValue(i - 1) + xAxis.getPixelForValue(i)) / 2;
        var right = end >= labels.length ? 2 * xAxis.getPixelForValue(labels.length - 1) - xAxis.getPixelForValue(labels.length - 2) : (xAxis.getPixelForValue(end - 1) + xAxis.getPixelForValue(end)) / 2;
        var yMin = yAxisTemp.getPixelForValue(minV), yMax = yAxisTemp.getPixelForValue(maxV);
        ctx.fillStyle = 'rgba(46, 204, 113, 0.1)';
        ctx.fillRect(left, yMax, right - left, yMin - yMax);
        ctx.strokeStyle = '#27ae60';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.strokeRect(left, yMax, right - left, yMin - yMax);
        i = end;
      }
      ctx.restore();
    }
  };

  Chart.register(customBackgroundPlugin);

  var unitModeTempLinesPlugin = {
    id: 'unitModeTempLines',
    afterDatasetsDraw: function(chart) {
      var opt = chart.options.plugins && chart.options.plugins.unitModeTempLines;
      if (!opt || !opt.outdoorTemp || !opt.indoorTemp || !opt.setTemp) return;
      var xAxis = chart.scales.x, yAxis = chart.scales.y1;
      if (!xAxis || !yAxis) return;
      var ctx = chart.ctx;
      var drawLine = function(data, color, dashed, width) {
        ctx.beginPath();
        var started = false;
        for (var i = 0; i < data.length; i++) {
          var v = data[i];
          if (v == null || typeof v !== 'number') { started = false; continue; }
          var x = xAxis.getPixelForValue(i);
          var y = yAxis.getPixelForValue(v);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
        ctx.stroke();
      };
      ctx.save();
      drawLine(opt.outdoorTemp, '#999', true, 1);
      drawLine(opt.indoorTemp, '#27ae60', false, 2);
      drawLine(opt.setTemp, '#27ae60', true, 2);
      ctx.restore();
    }
  };
  Chart.register(unitModeTempLinesPlugin);

  function padOrTrim(arr, len, fill) {
    if (arr.length >= len) return arr.slice(0, len);
    var out = arr.slice();
    while (out.length < len) out.push(fill != null ? fill : 0);
    return out;
  }
  function buildUnitModeChartData(um) {
    if (!um || !um.labels || !um.unitModeData || !um.unitModeColors) return null;
    var datasets = [];
    var i;
    for (i = 0; i < um.unitModeData.length; i++) {
      datasets.push({
        label: (i + 1) + '\u53f0',
        data: um.unitModeData[i],
        backgroundColor: um.unitModeColors[i],
        stack: 'units',
        yAxisID: 'y',
        barPercentage: 1,
        categoryPercentage: 1.15
      });
    }
    datasets.push({ label: '', data: um.labels.map(function() { return 0; }), backgroundColor: 'transparent', yAxisID: 'y1', barPercentage: 0.01, categoryPercentage: 0.01 });
    return { labels: um.labels, datasets: datasets };
  }
  var modeLabelMap = { 'COOL': '\u51b7\u623f', 'HEAT': '\u6696\u623f', 'FAN': '\u9001\u98a8', 'OFF': 'OFF' };
  function buildUnitModeTooltipLines(unitIds, modesAtHour) {
    if (!unitIds || !modesAtHour || unitIds.length !== modesAtHour.length) return [];
    var lines = [];
    for (var i = 0; i < unitIds.length; i++) {
      var m = modesAtHour[i] || 'OFF';
      lines.push((unitIds[i] || '') + ': ' + (modeLabelMap[m] || m));
    }
    return lines;
  }
  function muteColorForModel(c) {
    if (!c || c === 'transparent') return c;
    var m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (m) return 'rgba(' + m[1] + ',' + m[2] + ',' + m[3] + ',0.2)';
    return c;
  }
  function solidColorForLegacy(c) {
    if (!c || c === 'transparent') return c;
    var m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (m) return 'rgba(' + m[1] + ',' + m[2] + ',' + m[3] + ',0.55)';
    return c;
  }
  function buildCombinedUnitModeChartData(legacy, updated) {
    if (!legacy || !legacy.labels || !legacy.labels.length) return null;
    var labels = legacy.labels;
    var n = labels.length;
    var datasets = [];
    var maxLegacy = (legacy.unitModeData && legacy.unitModeData.length) ? legacy.unitModeData.length : 0;
    var maxUpdated = (updated && updated.unitModeData && updated.unitModeData.length) ? updated.unitModeData.length : 0;
    var i, h;
    for (i = 0; i < maxLegacy; i++) {
      var legacyColors = padOrTrim(legacy.unitModeColors[i], n, 'transparent').map(solidColorForLegacy);
      datasets.push({
        label: '\u5b9f\u7e3e ' + (i + 1) + '\u53f0',
        data: padOrTrim(legacy.unitModeData[i], n, 0),
        backgroundColor: legacyColors,
        borderColor: legacyColors.map(function(c) { return c === 'transparent' ? 'transparent' : c.replace(/,\s*[\d.]+\)$/, ', 0.9)'); }),
        borderWidth: 1,
        stack: 'legacy',
        yAxisID: 'y',
        barPercentage: 0.6,
        categoryPercentage: 0.8
      });
    }
    if (updated && updated.unitModeData && updated.unitModeColors) {
      for (i = 0; i < maxUpdated; i++) {
        var mutedColors = padOrTrim(updated.unitModeColors[i], n, 'transparent').map(muteColorForModel);
        datasets.push({
          label: '\u30e2\u30c7\u30eb\u51fa\u529b ' + (i + 1) + '\u53f0',
          data: padOrTrim(updated.unitModeData[i], n, 0),
          backgroundColor: mutedColors,
          borderColor: mutedColors,
          borderWidth: 0.5,
          stack: 'model',
          yAxisID: 'y',
          barPercentage: 0.6,
          categoryPercentage: 0.8
        });
      }
    }
    datasets.push({ label: '', data: labels.map(function() { return 0; }), backgroundColor: 'transparent', yAxisID: 'y1', barPercentage: 0.01, categoryPercentage: 0.01 });
    var tooltipData = {
      legacy: (legacy && legacy.unitIds && legacy.modesPerHour) ? { unitIds: legacy.unitIds, modesPerHour: legacy.modesPerHour } : null,
      updated: (updated && updated.unitIds && updated.modesPerHour) ? { unitIds: updated.unitIds, modesPerHour: updated.modesPerHour } : null
    };
    return { labels: labels, datasets: datasets, unitModeTooltip: tooltipData };
  }

  function buildTempDatasets(t, zoneData) {
    var outdoor = t.outdoor_temp || [];
    var ds = [];
    if (outdoor.length) {
      ds.push({ label: '\u5916\u6c17\u6e29\u5ea6\uff08\u904e\u53bb\uff09', data: outdoor, borderColor: '#999', backgroundColor: 'transparent', borderDash: [5, 3], tension: 0.3, pointRadius: 1, pointHoverRadius: 5, borderWidth: 1, yAxisID: 'y' });
    }
    if (zoneData) {
      if ((zoneData.indoor_legacy || []).some(function(v) { return v != null; }))
        ds.push({ label: '\u5ba4\u5185\u6e29\u5ea6\uff08\u904e\u53bb\uff09', data: zoneData.indoor_legacy, borderColor: '#27ae60', backgroundColor: 'transparent', tension: 0.3, pointRadius: 1, borderWidth: 2, yAxisID: 'y' });
      if ((zoneData.set_temp_legacy || []).some(function(v) { return v != null; }))
        ds.push({ label: '\u8a2d\u5b9a\u6e29\u5ea6\uff08\u904e\u53bb\uff09', data: zoneData.set_temp_legacy, borderColor: '#27ae60', backgroundColor: 'transparent', borderDash: [5, 3], borderWidth: 2, pointRadius: 0, yAxisID: 'y' });
      if ((zoneData.indoor_updated || []).some(function(v) { return v != null; }))
        ds.push({ label: '\u5ba4\u5185\u6e29\u5ea6\uff08\u4e88\u6e2c\uff09', data: zoneData.indoor_updated, borderColor: '#3498db', backgroundColor: 'transparent', tension: 0.3, pointRadius: 1, borderWidth: 2, yAxisID: 'y' });
      if ((zoneData.set_temp_updated || []).some(function(v) { return v != null; }))
        ds.push({ label: '\u8a2d\u5b9a\u6e29\u5ea6\uff08\u4e88\u6e2c\uff09', data: zoneData.set_temp_updated, borderColor: '#3498db', backgroundColor: 'transparent', borderDash: [3, 3], borderWidth: 2, pointRadius: 0, yAxisID: 'y' });
    }
    return ds;
  }

  function drawCharts(data, selectedZone) {
    var t = data.temperature || {};
    var labels = t.labels || [];
    var zones = t.zones || [];
    var zoneData = selectedZone ? zones.filter(function(z) { return z.zone === selectedZone; })[0] : (zones[0] || null);
    var umTip = document.getElementById('unitModeTooltipEl');
    if (umTip) umTip.style.display = 'none';
    while (chartInstances.length) { chartInstances.pop().destroy(); }

    if (labels.length && (t.outdoor_temp && t.outdoor_temp.length || zoneData)) {
      var comfortByMonth = (data.comfort_range_by_zone_month && selectedZone && data.comfort_range_by_zone_month[selectedZone]) ? data.comfort_range_by_zone_month[selectedZone] : [];
      var monthFromData = labels.length ? getMonthFromLabel(labels, 0) : 1;
      var comfort = (comfortByMonth && comfortByMonth[monthFromData - 1] != null) ? comfortByMonth[monthFromData - 1] : defaultComfort;
      var rangeMin = comfort.min != null ? comfort.min : 24, rangeMax = comfort.max != null ? comfort.max : 26;
      var comfortLabel = document.getElementById('comfortRangeLabel');
      if (comfortLabel) comfortLabel.textContent = Math.round(rangeMin * 10) / 10 + '~' + Math.round(rangeMax * 10) / 10;
      var tempOpts = {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 10, bottom: 10, left: 35, right: 10 } },
        plugins: { legend: { display: true, position: 'top' }, tooltip: { mode: 'index', intersect: false }, customBackground: { comfortByMonth: comfortByMonth } },
        scales: {
          x: { grid: { display: true, color: 'rgba(150,150,150,0.1)', borderDash: [5,5] }, ticks: { maxRotation: 45, minRotation: 30, font: { size: 10 }, callback: function(val, i) { var lbl = labels[i]; if (!lbl) return val; var d = new Date(lbl); if (isNaN(d.getTime())) return lbl; if (d.getHours() === 0 && d.getMinutes() === 0) return (d.getMonth()+1)+'/'+d.getDate(); return ('0'+d.getHours()).slice(-2)+':00'; } } },
          y: { position: 'left', min: 0, max: 42, ticks: { stepSize: 5, callback: function(v) { return v + '°C'; } }, title: { display: true, text: '温度 ℃' }, grid: { color: 'rgba(120,120,120,0.1)', borderDash: [5,5] } }
        }
      };
      var c2 = document.getElementById('tempChart');
      if (c2) chartInstances.push(new Chart(c2.getContext('2d'), { type: 'line', data: { labels: labels, datasets: buildTempDatasets(t, zoneData) }, options: tempOpts }));
    }

    var powerByZone = data.charts && data.charts.power_by_zone;
    var cp = data.charts && data.charts.total_power;
    var powerData = null;
    if (powerByZone && powerByZone.labels && powerByZone.zones && selectedZone) {
      var zonePower = powerByZone.zones.filter(function(z) { return z.zone === selectedZone; })[0];
      if (zonePower && zonePower.datasets && zonePower.datasets.length >= 2) {
        var toKwh = function(v) { return (v || 0) / 1000; };
        powerData = {
          labels: powerByZone.labels,
          datasets: [
            { label: '\u5b9f\u7e3e', data: zonePower.datasets[0].data.map(toKwh), backgroundColor: '#27ae60', borderColor: '#27ae60', barPercentage: 0.6, categoryPercentage: 0.8 },
            { label: '\u30e2\u30c7\u30eb\u51fa\u529b', data: zonePower.datasets[1].data.map(toKwh), backgroundColor: 'rgba(46,204,113,0.25)', borderColor: '#27ae60', borderWidth: 0.2, barPercentage: 0.6, categoryPercentage: 0.8 }
          ]
        };
      }
    }
    if (!powerData && cp && cp.labels && cp.labels.length && cp.datasets && cp.datasets.length) {
      powerData = { labels: cp.labels, datasets: [{ label: '\u5b9f\u7e3e', data: cp.datasets[0].data, backgroundColor: '#27ae60', borderColor: '#27ae60', barPercentage: 0.6, categoryPercentage: 0.8 }, { label: '\u30e2\u30c7\u30eb\u51fa\u529b', data: cp.datasets[1].data, backgroundColor: 'rgba(46,204,113,0.25)', borderColor: '#27ae60', borderWidth: 0.2, barPercentage: 0.6, categoryPercentage: 0.8 }] };
    }
    if (powerData) {
      var c3 = document.getElementById('powerChart');
      if (c3) chartInstances.push(new Chart(c3.getContext('2d'), {
        type: 'bar',
        data: powerData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: true, position: 'top' } },
          scales: {
            x: { grid: { display: true, color: 'rgba(150,150,150,0.1)', borderDash: [5,5] }, ticks: { maxRotation: 45, minRotation: 30, font: { size: 9 }, callback: function(val, i) { var lbl = powerData.labels[i]; if (!lbl) return val; var d = new Date(lbl); if (isNaN(d.getTime())) return lbl; if (d.getHours() === 0 && d.getMinutes() === 0) return (d.getMonth()+1)+'/'+d.getDate(); return ('0'+d.getHours()).slice(-2)+':00'; } } },
            y: { beginAtZero: true, min: 0, ticks: { callback: function(v) { return v; } }, title: { display: true, text: '電力消費量 kWh' }, grid: { color: 'rgba(120,120,120,0.1)', borderDash: [5,5] } }
          }
        }
      }));
    }

    var legacyUm = data.unit_mode_legacy;
    var updatedUm = data.unit_mode_updated;
    if (data.unit_mode_by_zone && selectedZone && data.unit_mode_by_zone[selectedZone]) {
      var z = data.unit_mode_by_zone[selectedZone];
      legacyUm = z.legacy || legacyUm;
      updatedUm = z.updated || updatedUm;
    }
    var hasUnitMode = (legacyUm && legacyUm.labels && legacyUm.labels.length) || (updatedUm && updatedUm.labels && updatedUm.labels.length);
    if (hasUnitMode) {
      var combinedData = buildCombinedUnitModeChartData(legacyUm, updatedUm);
      var unitModeEl = document.getElementById('unitModeChart');
      if (combinedData && unitModeEl) {
        var maxUnits = Math.max(legacyUm ? (legacyUm.maxUnits || 1) : 0, updatedUm ? (updatedUm.maxUnits || 1) : 0, 5);
        var tickCallback = function(val, i) { var lbl = combinedData.labels[i]; if (!lbl) return val; var d = new Date(lbl); if (isNaN(d.getTime())) return lbl; if (d.getHours() === 0 && d.getMinutes() === 0) return (d.getMonth()+1)+'/'+d.getDate(); return ('0'+d.getHours()).slice(-2)+':00'; };
        var unitModeTooltipEl = document.getElementById('unitModeTooltipEl');
        chartInstances.push(new Chart(unitModeEl.getContext('2d'), {
          type: 'bar',
          data: combinedData,
          options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { top: 10, bottom: 10, left: 40, right: 40 } },
            plugins: {
              legend: { display: false },
              tooltip: {
                enabled: false,
                mode: 'index',
                intersect: false,
                external: function(context) {
                  var el = unitModeTooltipEl || document.getElementById('unitModeTooltipEl');
                  if (!el) return;
                  if (context.tooltip.opacity === 0) {
                    el.style.display = 'none';
                    return;
                  }
                  var idx = context.tooltip.dataPoints && context.tooltip.dataPoints[0] ? context.tooltip.dataPoints[0].dataIndex : -1;
                  var chart = context.chart;
                  var tt = chart.data && chart.data.unitModeTooltip;
                  if (idx < 0 || !tt) {
                    el.style.display = 'none';
                    return;
                  }
                  var timeLabel = chart.data.labels[idx] || '';
                  var parts = [];
                  parts.push('<div class="tooltip-time">' + escapeHtml(String(timeLabel)) + '</div>');
                  if (tt.legacy && tt.legacy.unitIds && tt.legacy.modesPerHour && tt.legacy.modesPerHour[idx]) {
                    var legacyLines = buildUnitModeTooltipLines(tt.legacy.unitIds, tt.legacy.modesPerHour[idx]);
                    if (legacyLines.length) {
                      parts.push('<div class="tooltip-section"><div class="tooltip-section-title">\u5b9f\u7e3e</div>');
                      legacyLines.forEach(function(l) { parts.push('<div class="tooltip-line">' + escapeHtml(l) + '</div>'); });
                      parts.push('</div>');
                    }
                  }
                  if (tt.updated && tt.updated.unitIds && tt.updated.modesPerHour && tt.updated.modesPerHour[idx]) {
                    var updatedLines = buildUnitModeTooltipLines(tt.updated.unitIds, tt.updated.modesPerHour[idx]);
                    if (updatedLines.length) {
                      parts.push('<div class="tooltip-section"><div class="tooltip-section-title model">\u30e2\u30c7\u30eb\u51fa\u529b</div>');
                      updatedLines.forEach(function(l) { parts.push('<div class="tooltip-line">' + escapeHtml(l) + '</div>'); });
                      parts.push('</div>');
                    }
                  }
                  el.innerHTML = parts.join('');
                  var rect = chart.canvas.getBoundingClientRect();
                  var scaleX = chart.scales.x;
                  var dataIndex = context.tooltip.dataPoints[0].dataIndex;
                  var catWidth = (scaleX && scaleX.getPixelForValue) ? (scaleX.getPixelForValue(1) - scaleX.getPixelForValue(0)) : 0;
                  var barRight = context.tooltip.caretX + (catWidth * 0.25);
                  var gap = 6;
                  var x = rect.left + barRight + gap;
                  var y = rect.top + context.tooltip.caretY;
                  el.style.left = x + 'px';
                  el.style.top = y + 'px';
                  el.style.transform = 'translateY(-50%)';
                  el.style.display = 'block';
                }
              },
              unitModeTempLines: { outdoorTemp: [], indoorTemp: [], setTemp: [] }
            },
            scales: {
              x: { display: true, grid: { display: false }, ticks: { maxRotation: 45, minRotation: 30, font: { size: 10 }, callback: tickCallback } },
              y: { position: 'left', min: 0, max: maxUnits, ticks: { stepSize: 1, callback: function(v) { return Number.isInteger(v) ? v + '\u53f0' : ''; } }, title: { display: true, text: '\u7af6\u52d5\u53f0\u6570' }, grid: { display: false } },
              y1: { position: 'right', min: 0, max: 42, ticks: { stepSize: 5, callback: function(v) { return v + '\u00B0C'; } }, title: { display: true, text: '\u6e29\u5ea6 \u2103' }, grid: { display: false } }
            }
          }
        }));
      }
    }
  }

  function render(d) {
    var meta = d.meta || {};
    var zones = (meta.zones || []);
    var t = d.temperature || {};
    var hasTemp = t.labels && t.labels.length && (t.zones && t.zones.length);
    var facility = meta.facility || '';
    var titleText = facility ? (escapeHtml(facility) + ' \u8a2d\u65bd\uff1a\u5b9f\u7e3e\uff08\u904e\u53bb\uff09\u3068\u30e2\u30c7\u30eb\u51fa\u529b\u306e\u6bd4\u8f03') : '\u5b9f\u7e3e\uff08\u904e\u53bb\uff09\u3068\u30e2\u30c7\u30eb\u51fa\u529b\u306e\u6bd4\u8f03';
    var html = '<h1 style="max-width:1100px;">' + titleText + '</h1>';
    var labelsForDate = (d.temperature && d.temperature.labels) || (d.charts && d.charts.total_power && d.charts.total_power.labels) || [];
    var dateText = '';
    if (labelsForDate.length) {
      var first = new Date(labelsForDate[0]), last = new Date(labelsForDate[labelsForDate.length - 1]);
      if (!isNaN(first.getTime()) && !isNaN(last.getTime())) {
        var fmt = function(d) { return d.getFullYear() + '\u5e74' + (d.getMonth() + 1) + '\u6708' + d.getDate() + '\u65e5'; };
        var sameDay = first.getFullYear() === last.getFullYear() && first.getMonth() === last.getMonth() && first.getDate() === last.getDate();
        dateText = sameDay ? fmt(first) : fmt(first) + ' \u301c ' + fmt(last);
      }
    }
    html += '<div class="meta">\u6bd4\u8f03\u65e5: ' + escapeHtml(dateText || '\u2014') + '</div>';
    html += '<h2 style="max-width:1100px;">サマリー</h2>';
    html += renderTable(['項目', '値'], (d.summary || []).map(function(r) { return { '項目': r.metric, '値': r.value }; }));
    if (hasTemp) {
      html += '<p style="max-width:1100px;"><label>\u30be\u30fc\u30f3\uff08\u6e29\u5ea6\u63a8\u79fb\u30fb\u96fb\u529b\uff09: <select id="zoneSelect" class="zone-select">';
      zones.forEach(function(z) { html += '<option value="' + escapeHtml(z) + '">' + escapeHtml(z) + '</option>'; });
      html += '</select></label></p>';
    }
    html += '<div class="chart-container"><div class="title-section"><h2>温度推移 ℃</h2><div class="legend-sample">';
    html += '<span class="legend-item"><span class="legend-swatch" style="display:inline-block;width:24px;height:12px;background:rgba(46,204,113,0.2);box-sizing:border-box;"></span>営業時間</span>';
    html += ' <span class="legend-item"><span class="legend-swatch" style="display:inline-block;width:24px;height:12px;background:rgba(46,204,113,0.2);border:2px dashed #27ae60;box-sizing:border-box;"></span>許容範囲 <span id="comfortRangeLabel">24~26</span> ℃</span></div></div>';
    html += '<div class="chart-wrapper"><canvas id="tempChart"></canvas></div></div>';
    var hasUnitMode = (d.unit_mode_legacy && d.unit_mode_legacy.labels && d.unit_mode_legacy.labels.length) || (d.unit_mode_updated && d.unit_mode_updated.labels && d.unit_mode_updated.labels.length);
    if (hasUnitMode) {
      var legendUnitMode = '<div class="legend-sample"><span class="legend-item"><span class="color-dot" style="display:inline-block;background:rgba(0,163,224,0.45);"></span>冷房</span> <span class="legend-item"><span class="color-dot" style="display:inline-block;background:rgba(230,126,34,0.45);"></span>暖房</span> <span class="legend-item"><span class="color-dot" style="display:inline-block;background:rgba(120,120,120,0.45);"></span>送風</span></div>';
      html += '<div class="chart-container"><div class="title-section"><h2>台モード</h2>' + legendUnitMode + '</div><div class="chart-wrapper chart-wrapper-unit-mode"><canvas id="unitModeChart"></canvas><div id="unitModeTooltipEl" class="unit-mode-tooltip"></div></div></div>';
    }
    html += '<div class="chart-container"><div class="title-section"><h2>電力消費量 kWh</h2><div class="legend-sample"><span class="legend-item"><span class="color-dot legend-swatch" style="display:inline-block;background:#27ae60;"></span>実績</span> <span class="legend-item"><span class="color-dot legend-swatch" style="display:inline-block;background:rgba(46,204,113,0.25);"></span>モデル出力</span></div></div>';
    html += '<div class="chart-wrapper chart-wrapper-power"><canvas id="powerChart"></canvas></div></div>';
    document.getElementById('report').innerHTML = html;
    document.getElementById('report').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    var sel = document.getElementById('zoneSelect');
    var chosen = (sel && sel.options.length) ? (sel.options[sel.selectedIndex].value || zones[0]) : (zones[0] || null);
    drawCharts(d, chosen || null);
    if (sel && zones.length > 1) sel.addEventListener('change', function() { drawCharts(d, this.value || null); });
  }

  var dataEl = document.getElementById('comparison-data');
  var d = dataEl ? JSON.parse(dataEl.textContent) : null;
  if (d) { render(d); }
  else {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
    document.getElementById('error').textContent = 'No comparison data. Run compare_optimization_outputs.py to generate the report.';
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
    comfort_range_by_zone_month: dict[str, list[dict]],
    max_diff_table_rows: int = 200,
    zone_to_units: dict[str, list[str]] | None = None,
) -> Path:
    """Write comparison_report.html with data embedded (works when opened from file)."""
    zones = stats["zones"]
    chart_total_power = _chart_data_total_power(merged, zones)
    chart_power_by_zone = _chart_data_by_zone(merged, zones, "power")
    chart_units_by_zone = _chart_data_by_zone(merged, zones, "numb_units_on")
    chart_settemp_by_zone = _chart_data_by_zone(merged, zones, "set_temp")
    chart_temperature = _chart_data_temperature(merged, zones)

    unit_mode_legacy = None
    unit_mode_updated = None
    unit_legacy_df: pd.DataFrame | None = None
    unit_updated_df: pd.DataFrame | None = None
    unit_legacy_path = _resolve_unit_schedule_path(legacy_path)
    unit_updated_path = _resolve_unit_schedule_path(updated_path)
    if unit_legacy_path.exists():
        try:
            unit_legacy_df = pd.read_csv(unit_legacy_path)
            unit_mode_legacy = _unit_mode_from_schedule(unit_legacy_df)
            if unit_mode_legacy:
                _attach_temperature_to_unit_mode(unit_mode_legacy, chart_temperature, use_legacy=True)
        except Exception as e:
            print(f"Warning: Could not load unit schedule (legacy) {unit_legacy_path}: {e}", file=sys.stderr)
    if unit_updated_path.exists():
        try:
            unit_updated_df = pd.read_csv(unit_updated_path)
            unit_mode_updated = _unit_mode_from_schedule(unit_updated_df)
            if unit_mode_updated:
                _attach_temperature_to_unit_mode(unit_mode_updated, chart_temperature, use_legacy=False)
        except Exception as e:
            print(f"Warning: Could not load unit schedule (updated) {unit_updated_path}: {e}", file=sys.stderr)

    unit_mode_by_zone: dict[str, dict] | None = None
    if zone_to_units and (unit_legacy_df is not None or unit_updated_df is not None):
        unit_mode_by_zone = {}
        for zone in zones:
            uids = zone_to_units.get(zone) or []
            leg_z = _unit_mode_from_schedule(unit_legacy_df, uids) if unit_legacy_df is not None and uids else None
            upd_z = _unit_mode_from_schedule(unit_updated_df, uids) if unit_updated_df is not None and uids else None
            if leg_z is not None or upd_z is not None:
                unit_mode_by_zone[zone] = {"legacy": leg_z, "updated": upd_z}

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
        chart_temperature,
        comfort_range_by_zone_month,
        max_diff_table_rows,
        unit_mode_legacy=unit_mode_legacy,
        unit_mode_updated=unit_mode_updated,
        unit_mode_by_zone=unit_mode_by_zone,
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

    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve master Excel from legacy path (e.g. .../Clea/zone_schedule_... -> data/01_MasterData/MASTER_Clea.xlsx) and load 許容範囲 + zone→units
    comfort_range_by_zone_month: dict[str, list[dict]] = {}
    zone_to_units: dict[str, list[str]] = {}
    if stats["zones"]:
        store = legacy_path.parent.name
        for base in [Path.cwd(), Path(__file__).resolve().parent]:
            cand = base / "data" / "01_MasterData" / f"MASTER_{store}.xlsx"
            if cand.exists():
                comfort_range_by_zone_month = _load_comfort_range_from_master(cand, stats["zones"])
                zone_to_units = _load_zone_to_units_from_master(cand, stats["zones"])
                break

    try:
        html_path = _write_html_report(
            merged,
            diff_df,
            stats,
            legacy_path,
            updated_path,
            out_dir,
            comfort_range_by_zone_month,
            max_diff_table_rows=args.diff_table_rows,
            zone_to_units=zone_to_units,
        )
        print(f"Wrote HTML report: {html_path}")
    except Exception as e:
        print(f"Error generating HTML report: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
