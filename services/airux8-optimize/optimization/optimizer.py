"""
Zone-based HVAC optimization module.
Implements the algorithm from the image to find optimal AC settings per zone
by matching similar historical weather patterns and selecting the best-performing
settings that minimize power consumption while maintaining comfort.
"""

import json
import logging
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.utils import get_data_path
from processing.utilities.master_data_loader import (
    get_comfort_range,
    get_zone_operating_hours,
)


class Optimizer:
    """
    Zone-based HVAC optimization class.

    Two paths:
    - Model path (when data size OK and model available): wider candidates, score by model
      (predict power/temp/humidity under today's forecast), filter by comfort, select min predicted power.
    - Fallback path (when data size insufficient or model not available): similar days
      (temp + solar + humidity), rank by historical adjusted_power, filter by comfort
      (historical indoor temp), select best (lowest power within comfort).

    Supports whole-day or hour-block mode for the fallback path.
    """
    # Minimum historical data to use model path (otherwise fallback)
    MIN_HISTORICAL_ROWS = 500
    MIN_HISTORICAL_DAYS = 14

    # Zone order for wide format
    ZONE_ORDER = [
        "Area 1",
        "Area2_1",
        "Area2_2",
        "Area 3",
        "Area 4",
        "Meeting Room",
        "Break Room",
    ]
    
    def __init__(
        self,
        use_operating_hours: bool = False,
        hour_block_size: Optional[int] = 2,
        forecast_hour_range: Optional[Tuple[int, int]] = None,
        store_name: Optional[str] = None,
        use_mode_priority: bool = False,
        use_model_path: bool = False,
    ):
        """
        Initialize the Optimizer with configuration.

        Args:
            use_operating_hours: If True, filter by zone operating hours (default: False)
            hour_block_size: Number of consecutive hours to select (default 3).
                Use None for whole day mode (24 hours). Can be any positive integer >= 2.
            forecast_hour_range: Optional tuple (start_hour, end_hour) for forecast filtering
                (e.g., (8, 19) for 8 AM to 7 PM). If None, processes all 24 hours.
            store_name: Store name (e.g., "Clea") for loading operation type mapping from master data
            use_mode_priority: If True, prioritize target operation mode (COOL/HEAT) over FAN when selecting patterns.
                If False (default), simply select the pattern with lowest power consumption.
            use_model_path: If True, use model-based path when data and model are available.
                If False (default), always use fallback (similar-day + historical power).
        """
        # AC Mode mapping: operation type string to numeric value
        self.OPERATION_TYPE_TO_MODE = {"COOL": 1, "HEAT": 2, "FAN": 3, "OFF": 0}
        # Whether to use zone operating hours for optimization (default: False)
        self.use_operating_hours = use_operating_hours

        # Validate hour_block_size
        if hour_block_size is not None and (
            not isinstance(hour_block_size, int) or hour_block_size < 2
        ):
            raise ValueError(
                f"hour_block_size must be None (whole day mode) or an integer >= 2, "
                f"got {hour_block_size}"
            )
        self.hour_block_size = hour_block_size

        # Validate forecast_hour_range
        if forecast_hour_range is not None:
            if (
                not isinstance(forecast_hour_range, tuple)
                or len(forecast_hour_range) != 2
            ):
                raise ValueError(
                    f"forecast_hour_range must be a tuple of (start_hour, end_hour), "
                    f"got {forecast_hour_range}"
                )
            start_hour, end_hour = forecast_hour_range
            if not (
                0 <= start_hour < 24 and 0 < end_hour <= 24 and start_hour < end_hour
            ):
                raise ValueError(
                    f"forecast_hour_range must have start_hour in [0, 23] and end_hour in [1, 24] "
                    f"with start_hour < end_hour, got {forecast_hour_range}"
                )
        self.forecast_hour_range = forecast_hour_range

        # Store name for loading operation type mapping
        self.store_name = store_name

        # Whether to use mode priority when selecting patterns (eg. COOL/HEAT > FAN)
        self.use_mode_priority = use_mode_priority
        # When True, use model path when available; when False, always use fallback
        self.use_model_path = use_model_path

        self.category_mappings = self._load_category_mappings()
        # Load operation type mapping if store_name is provided
        self.operation_type_mapping = (
            self._load_operation_type_mapping(store_name) if store_name else {}
        )
        # When True, fallback path filters candidates by comfort range (historical indoor temp)
        self.use_comfort_filter = True
        # Lazy-loaded per-zone models for model path (store -> zone -> artifact)
        self._zone_models: Dict[str, dict] = {}
        self._models_store: Optional[str] = None

    def _ensure_models_loaded(self, store_name: Optional[str]) -> None:
        """Load per-zone models from 03_Models for the given store (once per store)."""
        if not store_name:
            return
        if self._models_store == store_name and self._zone_models:
            return
        try:
            from optimization.zone_model_trainer import load_all_zone_models_for_store

            self._zone_models = load_all_zone_models_for_store(store_name)
            self._models_store = store_name
            if self._zone_models:
                logging.info(
                    f"Loaded {len(self._zone_models)} zone model(s) for store {store_name}"
                )
        except Exception as e:
            logging.warning(f"Could not load zone models for {store_name}: {e}")
            self._zone_models = {}
            self._models_store = store_name

    def _model_available(self) -> bool:
        """Return True if model path is enabled and at least one zone model is loaded."""
        return self.use_model_path and len(self._zone_models) > 0

    def _filter_patterns_by_comfort(
        self,
        patterns_df: pd.DataFrame,
        zone: str,
        forecast_month: int,
        master_data: dict,
    ) -> pd.DataFrame:
        """
        Keep only patterns whose historical Indoor Temp. is within comfort range for zone and month.
        Returns a copy; if 'Indoor Temp.' is missing or comfort range cannot be loaded, returns full copy.
        """
        if patterns_df.empty:
            return patterns_df.copy()
        if "Indoor Temp." not in patterns_df.columns:
            return patterns_df.copy()
        try:
            comfort_min, comfort_max = get_comfort_range(master_data, zone, forecast_month)
        except Exception:
            return patterns_df.copy()
        in_comfort = (
            patterns_df["Indoor Temp."].notna()
            & (patterns_df["Indoor Temp."] >= comfort_min)
            & (patterns_df["Indoor Temp."] <= comfort_max)
        )
        return patterns_df.loc[in_comfort].copy()

    def _load_category_mappings(self) -> Dict:
        """Load category mappings from config file."""
        try:
            # Get the project root directory and navigate to config folder
            # __file__ is optimization/zone_optimizer.py, so go up 2 levels to project root
            project_root = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(project_root, "config", "category_mapping.json")
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load category mappings: {e}")
            return {}

    def _load_operation_type_mapping(
        self, store_name: str
    ) -> Dict[Tuple[str, int], str]:
        """
        Load operation type (運転区分) mapping from 制御マスタ sheet in Master Excel file.

        Args:
            store_name: Store name (e.g., "Clea")

        Returns:
            Dictionary mapping (zone_name, month_number) to operation type (e.g., "HEAT", "COOL")
        """
        try:
            from service.storage import get_storage_client

            storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()
            client = get_storage_client()

            if storage_backend == "gcs":
                excel_path = f"01_MasterData/MASTER_{store_name}.xlsx"
            else:
                master_dir = get_data_path("master_data_path")
                excel_path = os.path.join(master_dir, f"MASTER_{store_name}.xlsx")

            # Read 制御マスタ sheet
            control_master_df = client.read_excel(excel_path, sheet_name="制御マスタ")

            # Check if 運転区分 column exists
            if "運転区分" not in control_master_df.columns:
                logging.warning(
                    f"運転区分 column not found in 制御マスタ sheet for {store_name}. "
                    "Operation type filtering will be disabled."
                )
                return {}

            # Create mapping: (zone_name, month_number) -> operation_type
            operation_type_mapping = {}
            for _, row in control_master_df.iterrows():
                zone_name = row.get("制御区分")
                month_str = row.get("月")
                operation_type = row.get("運転区分")

                if pd.isna(zone_name) or pd.isna(month_str) or pd.isna(operation_type):
                    continue

                # Convert month string (e.g., "1月") to integer
                try:
                    month_number = int(month_str.replace("月", ""))
                    if 1 <= month_number <= 12:
                        # Normalize operation type to uppercase for consistent comparison
                        normalized_operation_type = str(operation_type).strip().upper()
                        operation_type_mapping[(str(zone_name), month_number)] = (
                            normalized_operation_type
                        )
                except (ValueError, AttributeError):
                    continue

            logging.info(
                f"Loaded operation type mapping: {len(operation_type_mapping)} entries for {store_name}"
            )
            return operation_type_mapping

        except Exception as e:
            logging.warning(
                f"Could not load operation type mapping for {store_name}: {e}. "
                "Operation type filtering will be disabled."
            )
            return {}

    def _get_forecast_operation_type(
        self, forecast_day_data: pd.DataFrame, zone: str
    ) -> Optional[str]:
        """
        Get operation type (運転区分) for the forecast day based on its month.

        Args:
            forecast_day_data: Forecast data for a single day
            zone: Zone name to filter by

        Returns:
            Operation type string ("HEAT", "COOL", etc.) or None if not found
        """
        if not self.operation_type_mapping:
            return None

        try:
            # Extract month from forecast day data
            forecast_datetime = pd.to_datetime(forecast_day_data["datetime"].iloc[0])
            forecast_month = forecast_datetime.month

            # Look up operation type for (zone, month)
            key = (zone, forecast_month)
            operation_type = self.operation_type_mapping.get(key)

            if operation_type:
                logging.debug(
                    f"Forecast day operation type for zone {zone}, month {forecast_month}: {operation_type}"
                )
            else:
                logging.debug(
                    f"No operation type found for zone {zone}, month {forecast_month}"
                )

            return operation_type

        except Exception as e:
            logging.warning(
                f"Error getting forecast operation type for zone {zone}: {e}"
            )
            return None

    def _get_allowed_ac_modes(self, operation_type: str) -> List[int]:
        """
        Get list of allowed AC Mode values for a given operation type.

        Relaxed filtering rules:
        - COOL: Allows COOL (1) and FAN (3), excludes HEAT (2)
        - HEAT: Allows HEAT (2) and FAN (3), excludes COOL (1)
        - FAN: Only FAN (3)
        - OFF: Only OFF (0)

        Args:
            operation_type: Operation type string ("HEAT", "COOL", "FAN", "OFF")

        Returns:
            List of allowed AC Mode numeric values
        """
        if not operation_type:
            return []

        operation_type_upper = operation_type.upper()
        target_mode = self.OPERATION_TYPE_TO_MODE.get(operation_type_upper)

        if target_mode is None:
            return []

        # Relaxed filtering: COOL and HEAT also allow FAN mode
        if operation_type_upper == "COOL":
            return [
                self.OPERATION_TYPE_TO_MODE["COOL"],
                self.OPERATION_TYPE_TO_MODE["FAN"],
            ]
        elif operation_type_upper == "HEAT":
            return [
                self.OPERATION_TYPE_TO_MODE["HEAT"],
                self.OPERATION_TYPE_TO_MODE["FAN"],
            ]
        else:
            # For FAN or OFF, only allow the exact mode
            return [target_mode]

    def _filter_days_by_operation_type(
        self,
        historical_df: pd.DataFrame,
        zone: str,
        operation_type: str,
        candidate_days: List[date],
    ) -> List[date]:
        """
        Filter candidate days to only include days that have AC Mode matching the operation type.

        Relaxed filtering: COOL allows both COOL and FAN modes, HEAT allows both HEAT and FAN modes.

        Note: Days can be from ANY month - we only check that the historical day's AC Mode
        matches the target operation type (e.g., if forecast is COOL, select days with COOL or FAN mode
        regardless of which month they're from).

        Uses vectorized operations for better performance with large candidate lists.

        Args:
            historical_df: Historical data DataFrame
            zone: Zone name to filter by
            operation_type: Target operation type (e.g., "HEAT", "COOL")
            candidate_days: List of candidate Date objects to filter

        Returns:
            Filtered list of Date objects with AC Mode matching the operation type (relaxed)
        """
        if not operation_type:
            # No filtering if operation type not available
            return candidate_days

        # Get allowed AC Mode values (relaxed: COOL allows FAN, HEAT allows FAN)
        allowed_modes = self._get_allowed_ac_modes(operation_type)

        if not allowed_modes:
            logging.warning(
                f"Unknown operation type {operation_type}, cannot filter by AC Mode"
            )
            return candidate_days

        # Filter zone data once
        zone_data = historical_df[historical_df["zone"] == zone].copy()
        if "A/C Mode" not in zone_data.columns:
            logging.warning(
                f"A/C Mode column not found in historical data, cannot filter by AC Mode"
            )
            return candidate_days

        # Filter to only candidate days first (vectorized operation)
        candidate_days_set = set(candidate_days)
        candidate_zone_data = zone_data[
            zone_data["Date"].isin(candidate_days_set)
        ].copy()

        if len(candidate_zone_data) == 0:
            # No data for any candidate days
            return []

        # Group by Date and check if any row has allowed AC Mode (vectorized with isin)
        days_with_allowed_mode_set = set(
            candidate_zone_data[candidate_zone_data["A/C Mode"].isin(allowed_modes)]
            .groupby("Date")["Date"]
            .first()
            .index.tolist()
        )

        # Convert back to list of date objects, preserving order from candidate_days
        # Using set for O(1) lookup instead of O(n) list lookup
        filtered_days = [
            day for day in candidate_days if day in days_with_allowed_mode_set
        ]

        if len(filtered_days) < len(candidate_days):
            mode_names = [self._map_ac_mode(mode) for mode in allowed_modes]
            logging.info(
                f"Filtered {len(candidate_days)} candidate days to {len(filtered_days)} days "
                f"with AC Mode matching operation type {operation_type} (allowed modes: {mode_names}) for zone {zone}"
            )

        return filtered_days

    def _get_weather_weights(self, hour: int) -> Dict[str, float]:
        """
        Get weather weights based on hour of the day.

        For hours 17:00 to 6:00 (next day): temperature 0.8, solar 0.0, humidity 0.2
        For other hours (7:00 to 16:59): temperature 0.5, solar 0.3, humidity 0.2
        Humidity is used when available in data (perceived/actual load).

        Args:
            hour: Hour of the day (0-23)

        Returns:
            Dictionary with temperature, solar_radiation, and humidity weights
        """
        # Hours 17:00 (17) to 6:00 (6) next day: temperature dominant, no solar
        if hour >= 17 or hour < 7:
            return {"temperature": 0.8, "solar_radiation": 0.0, "humidity": 0.2}
        # Hours 7:00 to 16:59: temperature, solar, and humidity
        else:
            return {"temperature": 0.5, "solar_radiation": 0.3, "humidity": 0.2}

    def _map_ac_mode(self, mode_value: int) -> str:
        """Map AC mode numeric value to string."""
        if not self.category_mappings or "A/C Mode" not in self.category_mappings:
            return str(mode_value)

        mode_mapping = self.category_mappings["A/C Mode"]
        # Reverse mapping: find key by value
        for mode_str, mode_num in mode_mapping.items():
            if mode_num == mode_value:
                return mode_str
        return str(mode_value)

    def _map_fan_speed(self, fan_speed_value: int) -> str:
        """Map fan speed numeric value to string."""
        if not self.category_mappings or "A/C Fan Speed" not in self.category_mappings:
            return str(fan_speed_value)

        fan_mapping = self.category_mappings["A/C Fan Speed"]
        # Reverse mapping: find key by value
        for fan_str, fan_num in fan_mapping.items():
            if fan_num == fan_speed_value:
                return fan_str
        return str(fan_speed_value)

    def _map_ac_on_off(self, units_count: float) -> str:
        """Map number of units to ON/OFF string."""
        return "ON" if units_count > 0 else "OFF"

    def load_historical_patterns(self, features_csv_path: str) -> pd.DataFrame:
        """
        Load features_processed_Clea.csv and filter valid records.

        Filters for records where:
        - Indoor Temp. is not null
        - adjusted_power > 0
        - Outdoor Temp. and Solar Radiation are not null
        - A/C ON/OFF > 0 (AC must be ON)

        Args:
            features_csv_path: Path to the features CSV file

        Returns:
            DataFrame ready for pattern matching (only AC ON records)
        """
        logging.info(f"Loading historical patterns from {features_csv_path}")

        # Load the CSV file
        df = pd.read_csv(features_csv_path)

        # Convert datetime column
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Add Date column for day-level grouping
        df["Date"] = df["Datetime"].dt.date

        # Filter valid records: non-null Indoor Temp., positive adjusted_power, AC ON only
        valid_mask = (
            df["Indoor Temp."].notna()
            & (df["adjusted_power"] > 0)
            & df["Outdoor Temp."].notna()
            & df["Solar Radiation"].notna()
            # & (df["A/C ON/OFF"] > 0)  # Only AC ON status
        )

        df_filtered = df[valid_mask].copy()

        logging.info(
            f"Loaded {len(df_filtered)} valid historical patterns from {len(df)} total records"
        )
        logging.info(
            f"Date range: {df_filtered['Datetime'].min()} to {df_filtered['Datetime'].max()}"
        )
        logging.info(f"Zones: {sorted(df_filtered['zone'].unique())}")

        return df_filtered

    def _find_similar_days(
        self,
        historical_df: pd.DataFrame,
        forecast_day_data: pd.DataFrame,
        zone: str,
        n_top: int = 20,
    ) -> List:
        """
        Find similar historical days for a given forecast day.

        Args:
            historical_df: Historical data DataFrame
            forecast_day_data: Forecast data for a single day
            zone: Zone name to filter by
            n_top: Number of top similar days to return

        Returns:
            List of Date objects for top N most similar historical days
        """
        # Filter by zone
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return []

        # Weather columns for similarity: temp and solar always; humidity when present
        weather_cols = ["Outdoor Temp.", "Solar Radiation"]
        use_humidity = (
            "Outdoor Humidity" in forecast_day_data.columns
            and "Outdoor Humidity" in zone_data.columns
        )
        if use_humidity:
            weather_cols = ["Outdoor Temp.", "Solar Radiation", "Outdoor Humidity"]

        # Calculate forecast day's mean weather
        f_means = {c: forecast_day_data[c].mean() for c in weather_cols}

        # Historical daily means for the zone
        daily_hist = (
            zone_data.groupby("Date")[weather_cols].mean().reset_index()
        )

        if daily_hist.empty:
            return []

        # Add day-of-week and weekend/holiday for similar-day (weather + day type per policy)
        forecast_dt = pd.to_datetime(forecast_day_data["datetime"].iloc[0])
        f_dow = forecast_dt.dayofweek  # 0=Mon, 6=Sun
        f_weekend = 1 if f_dow in (5, 6) else 0
        f_holiday = 0  # optional: set from calendar if available
        if "DayOfWeek" in zone_data.columns and "IsWeekend" in zone_data.columns:
            day_meta = (
                zone_data.groupby("Date")[["DayOfWeek", "IsWeekend"]]
                .first()
                .reset_index()
            )
            daily_hist = daily_hist.merge(day_meta, on="Date", how="left")
            # Weekend mismatch penalty (0 or 1)
            daily_hist["_weekend_diff"] = (
                daily_hist["IsWeekend"].fillna(0).astype(int) != f_weekend
            ).astype(int)
            # Day-of-week circular distance (0–3)
            h_dow = daily_hist["DayOfWeek"].fillna(0).astype(int)
            dow_diff = np.abs(h_dow - f_dow)
            daily_hist["_dow_diff"] = np.minimum(dow_diff, 7 - dow_diff)
            if "IsHoliday" in zone_data.columns:
                hol_meta = zone_data.groupby("Date")["IsHoliday"].first().reset_index()
                daily_hist = daily_hist.merge(hol_meta, on="Date", how="left")
                daily_hist["_holiday_diff"] = (
                    daily_hist["IsHoliday"].fillna(0).astype(int) != f_holiday
                ).astype(int)
            else:
                daily_hist["_holiday_diff"] = 0
        else:
            daily_hist["_weekend_diff"] = 0
            daily_hist["_dow_diff"] = 0
            daily_hist["_holiday_diff"] = 0

        # Z-score normalization per feature
        daily_hist = daily_hist.copy()
        forecast_zs = {}
        for c in weather_cols:
            hist_vals = daily_hist[c].dropna()
            h_mean, h_std = hist_vals.mean(), hist_vals.std()
            forecast_zs[c] = (
                (f_means[c] - h_mean) / h_std if h_std > 0 else 0
            )
            daily_hist[f"{c}_z"] = (
                (daily_hist[c] - h_mean) / h_std if h_std > 0 else 0
            )

        # Get weather weights based on first hour of forecast day
        forecast_first_hour = forecast_dt.hour
        weather_weights = self._get_weather_weights(forecast_first_hour)

        # Weighted day-level distance score (lower is better): weather + day type
        score = (
            weather_weights["temperature"]
            * abs(daily_hist["Outdoor Temp._z"] - forecast_zs["Outdoor Temp."])
            + weather_weights["solar_radiation"]
            * abs(daily_hist["Solar Radiation_z"] - forecast_zs["Solar Radiation"])
        )
        if use_humidity:
            score = score + weather_weights["humidity"] * abs(
                daily_hist["Outdoor Humidity_z"] - forecast_zs["Outdoor Humidity"]
            )
        # Day type terms (weekend > dow > holiday) so similar days prefer same weekday/weekend
        score = (
            score
            + 0.2 * daily_hist["_weekend_diff"]
            + 0.05 * daily_hist["_dow_diff"]
            + 0.05 * daily_hist["_holiday_diff"]
        )
        daily_hist["score"] = score

        # Select top N days based on day-level similarity
        top_days = daily_hist.nsmallest(n_top, "score")["Date"].tolist()

        return top_days

    def _select_best_complete_day(
        self,
        historical_df: pd.DataFrame,
        zone: str,
        top_days: List,
        forecast_day_data: pd.DataFrame,
        master_data: dict,
    ) -> Tuple[Optional[date], pd.DataFrame]:
        """
        Select the best complete historical day from top similar days and return its patterns.

        Uses a three-tier priority system:
        1. First priority: Select from complete days (all forecast hours available) - choose lowest power
        2. Second priority: If no complete days, select day with least missing hours (if tie, lowest power)

        The returned patterns DataFrame is reduced to one row per hour (lowest power pattern for each hour),
        ensuring efficient lookup and consistent pattern selection.

        Args:
            historical_df: Historical data DataFrame
            zone: Zone name to filter by
            top_days: List of Date objects for similar days
            forecast_day_data: Forecast data for the day (to match hours)
            master_data: Master data dictionary (currently unused, reserved for future comfort filtering)

        Returns:
            Tuple of (best_day Date object, patterns DataFrame with one row per hour) or (None, empty DataFrame) if no valid days
        """
        # Filter by zone
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return None, pd.DataFrame()

        # Filter top_days by operation type if available
        forecast_operation_type = self._get_forecast_operation_type(
            forecast_day_data, zone
        )
        if forecast_operation_type:
            filtered_top_days = self._filter_days_by_operation_type(
                historical_df, zone, forecast_operation_type, top_days
            )
            if len(filtered_top_days) > 0:
                top_days = filtered_top_days
                logging.info(
                    f"Zone {zone}: Filtered to {len(top_days)} days matching operation type {forecast_operation_type}"
                )
            else:
                logging.warning(
                    f"Zone {zone}: No days found matching operation type {forecast_operation_type}, "
                    f"falling back to original {len(top_days)} candidate days"
                )

        # Get forecast hours to match
        forecast_hours = pd.to_datetime(forecast_day_data["datetime"]).dt.hour.unique()
        required_hour_count = len(forecast_hours)

        # Evaluate each candidate day
        complete_days = []  # Days with all required hours
        incomplete_days = []  # Days with missing hours

        for day_date in top_days:
            day_data = zone_data[zone_data["Date"] == day_date].copy()
            if len(day_data) == 0:
                continue

            # Filter to only forecast hours (in case forecast doesn't have all 24 hours)
            day_data["hour"] = day_data["Datetime"].dt.hour
            day_data = day_data[day_data["hour"].isin(forecast_hours)]

            if len(day_data) == 0:
                continue

            # Calculate average power for the day (lower is better)
            avg_power = day_data["adjusted_power"].mean()
            available_hour_count = len(day_data["hour"].unique())
            missing_count = required_hour_count - available_hour_count
            coverage_rate = (
                available_hour_count / required_hour_count
                if required_hour_count > 0
                else 0.0
            )

            day_info = {
                "Date": day_date,
                "avg_power": avg_power,
                "available_hour_count": available_hour_count,
                "missing_count": missing_count,
                "coverage_rate": coverage_rate,
            }

            if missing_count == 0:
                # Complete day - all required hours available
                complete_days.append(day_info)
            else:
                # Incomplete day - some hours missing
                incomplete_days.append(day_info)

        # Select best day using priority system
        best_day = None

        if len(complete_days) > 0:
            # 第1優先：完全なデータがある日から最小電力の日を選択
            best_day = min(complete_days, key=lambda x: x["avg_power"])["Date"]
            logging.info(
                f"Zone {zone}: Selected complete day {best_day} "
                f"(power: {min(x['avg_power'] for x in complete_days):.0f}W)"
            )
        elif len(incomplete_days) > 0:
            # 第2優先：完全な日がない場合、欠損時間が最も少ない日を選択
            # 同じ欠損数の場合、最小電力の日を優先
            best_day_info = min(
                incomplete_days, key=lambda x: (x["missing_count"], x["avg_power"])
            )
            best_day = best_day_info["Date"]

            logging.warning(
                f"Zone {zone}: No complete days found. "
                f"第2優先適用: Selected day {best_day} "
                f"with {best_day_info['missing_count']} missing hours "
                f"(第3優先確認: coverage rate {best_day_info['coverage_rate']*100:.1f}%, "
                f"power: {best_day_info['avg_power']:.0f}W)"
            )
        else:
            # No valid days found
            logging.warning(
                f"Zone {zone}: No valid days found in top_days for forecast period"
            )
            return None, pd.DataFrame()

        # Get patterns for the best day (avoid redundant filtering)
        best_day_patterns = zone_data[zone_data["Date"] == best_day].copy()
        best_day_patterns["hour"] = best_day_patterns["Datetime"].dt.hour
        best_day_patterns = best_day_patterns[
            best_day_patterns["hour"].isin(forecast_hours)
        ].copy()

        # Filter by comfort range (historical indoor temp) when enabled
        if self.use_comfort_filter and not best_day_patterns.empty and master_data:
            forecast_month = pd.to_datetime(
                forecast_day_data["datetime"].iloc[0]
            ).month
            in_comfort = self._filter_patterns_by_comfort(
                best_day_patterns, zone, forecast_month, master_data
            )
            # For each hour: use in-comfort pattern if any, else fall back to best (lowest power) for that hour
            if not in_comfort.empty:
                best_day_patterns = best_day_patterns.sort_values(
                    "adjusted_power"
                )
                hours_with_comfort = in_comfort["hour"].unique()
                rows_list = []
                for h in forecast_hours:
                    in_comfort_h = in_comfort[in_comfort["hour"] == h]
                    if len(in_comfort_h) > 0:
                        rows_list.append(
                            in_comfort_h.sort_values("adjusted_power").iloc[
                                0
                            ]
                        )
                    else:
                        fallback_h = best_day_patterns[
                            best_day_patterns["hour"] == h
                        ]
                        if len(fallback_h) > 0:
                            rows_list.append(fallback_h.iloc[0])
                if rows_list:
                    best_day_patterns = pd.DataFrame(rows_list)

        # Filter patterns by AC Mode to match operation type if available (relaxed filtering)
        if forecast_operation_type and "A/C Mode" in best_day_patterns.columns:
            # Get allowed AC Mode values (relaxed: COOL allows FAN, HEAT allows FAN)
            allowed_modes = self._get_allowed_ac_modes(forecast_operation_type)

            if allowed_modes:
                # Only keep patterns with allowed AC Mode (vectorized with isin)
                before_count = len(best_day_patterns)
                best_day_patterns = best_day_patterns[
                    best_day_patterns["A/C Mode"].isin(allowed_modes)
                ].copy()
                after_count = len(best_day_patterns)

                if after_count < before_count:
                    mode_names = [self._map_ac_mode(mode) for mode in allowed_modes]
                    logging.info(
                        f"Zone {zone}: Filtered patterns from {before_count} to {after_count} "
                        f"matching operation type {forecast_operation_type} (allowed modes: {mode_names})"
                    )

        # Reduce to one pattern per hour (by default, lowest power; optionally prefer target mode)
        # This ensures the returned DataFrame has exactly one row per hour
        if len(best_day_patterns) == 0:
            return best_day, pd.DataFrame()

        # Sort patterns: by default, select lowest power
        # If use_mode_priority is True, prefer target operation mode, then by lowest power
        # For COOL: prefer COOL (1) over FAN (3), then by power
        # For HEAT: prefer HEAT (2) over FAN (3), then by power
        if (
            self.use_mode_priority
            and forecast_operation_type
            and "A/C Mode" in best_day_patterns.columns
        ):
            target_mode = self.OPERATION_TYPE_TO_MODE.get(
                forecast_operation_type.upper()
            )
            if target_mode is not None:
                # Create priority: target mode = 0 (highest priority), others = 1
                best_day_patterns = best_day_patterns.copy()
                best_day_patterns["mode_priority"] = (
                    best_day_patterns["A/C Mode"] != target_mode
                ).astype(int)
                # Sort by mode priority (target mode first), then by power
                best_day_patterns = best_day_patterns.sort_values(
                    ["mode_priority", "adjusted_power"]
                )
                best_day_patterns = best_day_patterns.drop(columns=["mode_priority"])
            else:
                # Fallback: sort by power only
                best_day_patterns = best_day_patterns.sort_values("adjusted_power")
        else:
            # Default: sort by power only (lowest power first)
            best_day_patterns = best_day_patterns.sort_values("adjusted_power")

        # Group by hour and select the first row (preferred mode, then lowest power)
        best_day_patterns = best_day_patterns.groupby("hour", as_index=False).first()

        return best_day, best_day_patterns

    def _calculate_hour_block_distance(
        self,
        forecast_block: pd.DataFrame,
        historical_block: pd.DataFrame,
    ) -> float:
        """
        Calculate distance between forecast hour block and historical hour block.

        IMPORTANT: This function expects that forecast_block and historical_block
        contain data for the same hours (e.g., if forecast is for hours [8, 9, 10],
        historical_block should also be for hours [8, 9, 10]). This ensures proper
        hour-to-hour comparison rather than arbitrary time matching.

        Args:
            forecast_block: DataFrame with forecast weather for specific hours
            historical_block: DataFrame with historical weather for the same hours

        Returns:
            Weather distance (lower is better, represents how similar the blocks are)
        """
        # Calculate mean weather values for each block
        forecast_temp_mean = forecast_block["Outdoor Temp."].mean()
        forecast_solar_mean = forecast_block["Solar Radiation"].mean()

        historical_temp_mean = historical_block["Outdoor Temp."].mean()
        historical_solar_mean = historical_block["Solar Radiation"].mean()

        # Normalized differences (z-style using historical std)
        hist_temp_std = historical_block["Outdoor Temp."].std()
        hist_solar_std = historical_block["Solar Radiation"].std()

        if hist_temp_std > 0:
            temp_diff = abs(forecast_temp_mean - historical_temp_mean) / hist_temp_std
        else:
            temp_diff = abs(forecast_temp_mean - historical_temp_mean)

        if hist_solar_std > 0:
            solar_diff = (
                abs(forecast_solar_mean - historical_solar_mean) / hist_solar_std
            )
        else:
            solar_diff = abs(forecast_solar_mean - historical_solar_mean)

        # Get weather weights
        forecast_first_hour = pd.to_datetime(forecast_block["datetime"].iloc[0]).hour
        weather_weights = self._get_weather_weights(forecast_first_hour)

        weather_distance = (
            weather_weights["temperature"] * temp_diff
            + weather_weights["solar_radiation"] * solar_diff
        )

        # Add humidity when present in both blocks
        if (
            "Outdoor Humidity" in forecast_block.columns
            and "Outdoor Humidity" in historical_block.columns
        ):
            f_hum_mean = forecast_block["Outdoor Humidity"].mean()
            h_hum_mean = historical_block["Outdoor Humidity"].mean()
            h_hum_std = historical_block["Outdoor Humidity"].std()
            hum_diff = (
                abs(f_hum_mean - h_hum_mean) / h_hum_std
                if h_hum_std > 0
                else abs(f_hum_mean - h_hum_mean)
            )
            weather_distance = (
                weather_distance + weather_weights["humidity"] * hum_diff
            )

        return weather_distance

    def _select_best_hour_blocks(
        self,
        historical_df: pd.DataFrame,
        zone: str,
        top_days: List,
        forecast_day_data: pd.DataFrame,
        master_data: dict,
    ) -> Dict[int, pd.Series]:
        """
        Select best N-hour blocks from candidate days for hour block mode.

        For each forecast hour block (consecutive N hours), finds the best matching
        N-hour block from all candidate historical days where AC is ON.

        CRITICAL: Historical blocks must have the exact same hours as forecast blocks.
        For example, if forecast is for hours [8, 9, 10], the historical block must
        also be for hours [8, 9, 10] (not [14, 15, 16] or any other hours).
        This ensures proper hour-to-hour matching (8 AM forecast → 8 AM historical).

        Args:
            historical_df: Historical data DataFrame (already filtered for AC ON)
            zone: Zone name to filter by
            top_days: List of Date objects for similar days
            forecast_day_data: Forecast data for the day
            master_data: Master data dictionary (unused, reserved for future use)

        Returns:
            Dictionary mapping forecast hour → best historical pattern row (pd.Series)
        """
        # Filter by zone (AC ON filtering already done in load_historical_patterns)
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return {}

        # Filter top_days by operation type if available
        forecast_operation_type = self._get_forecast_operation_type(
            forecast_day_data, zone
        )
        if forecast_operation_type:
            filtered_top_days = self._filter_days_by_operation_type(
                historical_df, zone, forecast_operation_type, top_days
            )
            if len(filtered_top_days) > 0:
                top_days = filtered_top_days
                logging.info(
                    f"Zone {zone}: Filtered to {len(top_days)} days matching operation type {forecast_operation_type}"
                )
            else:
                logging.warning(
                    f"Zone {zone}: No days found matching operation type {forecast_operation_type}, "
                    f"falling back to original {len(top_days)} candidate days"
                )

        # Optimize: Compute hour column once for all zone data
        zone_data["hour"] = zone_data["Datetime"].dt.hour

        # Extract forecast date for display
        forecast_date = pd.to_datetime(forecast_day_data["datetime"].iloc[0]).date()

        # Get forecast hours in order (preserve order from forecast_day_data)
        forecast_hours_ordered = pd.to_datetime(
            forecast_day_data["datetime"]
        ).dt.hour.tolist()
        # Get unique hours while preserving order
        seen = set()
        unique_forecast_hours = []
        for hour in forecast_hours_ordered:
            if hour not in seen:
                seen.add(hour)
                unique_forecast_hours.append(hour)

        print(
            f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Processing {len(unique_forecast_hours)} forecast hours: {unique_forecast_hours}"
        )

        # Group forecast hours into consecutive non-overlapping blocks of hour_block_size
        hour_blocks = []
        block_size = self.hour_block_size
        current_block = []

        for hour in unique_forecast_hours:
            if len(current_block) == 0:
                # Start new block
                current_block = [hour]
            elif hour == current_block[-1] + 1:  # consecutive hour
                # Consecutive hour, add to current block
                current_block.append(hour)
                if len(current_block) == block_size:
                    # Block is complete, add it and start new block
                    hour_blocks.append(current_block)
                    current_block = []
            else:
                # Not consecutive, save current block if it has any hours, then start new block
                if len(current_block) > 0:
                    hour_blocks.append(current_block)
                current_block = [hour]

        # Add remaining block if any
        if len(current_block) > 0:
            hour_blocks.append(current_block)

        print(
            f"[Zone: {zone}] Forecast Date: {forecast_date} | Grouped into {len(hour_blocks)} hour block(s) (block_size={block_size}): {hour_blocks}"
        )

        # Dictionary to store best pattern for each hour
        patterns_by_hour = {}

        # Process each forecast hour block
        for block_idx, hour_block in enumerate(hour_blocks, 1):
            # Extract forecast weather for this hour block
            forecast_block = forecast_day_data[
                pd.to_datetime(forecast_day_data["datetime"]).dt.hour.isin(hour_block)
            ].copy()

            if len(forecast_block) == 0:
                continue

            actual_block_size = len(
                hour_block
            )  # May be smaller than block_size for last incomplete block

            # Skip if block size is 0 (shouldn't happen, but defensive check)
            if actual_block_size == 0:
                continue

            print(
                f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Block {block_idx}/{len(hour_blocks)}: Forecast hours {hour_block} (size={actual_block_size})"
            )

            best_block_candidates = []

            # Evaluate each candidate day
            for day_date in top_days:
                day_data = zone_data[zone_data["Date"] == day_date].copy()
                if len(day_data) == 0:
                    continue

                # CRITICAL: Only consider historical blocks where hours exactly match forecast hours
                # For example, if forecast is [8, 9, 10], historical block must also be [8, 9, 10]
                day_hours = sorted(day_data["hour"].unique())

                # Check if all forecast hours exist in this day's historical data
                if not all(hour in day_hours for hour in hour_block):
                    continue

                # Extract historical block with exact same hours as forecast block
                candidate_block = day_data[day_data["hour"].isin(hour_block)].copy()

                if len(candidate_block) == 0:
                    continue

                # Filter by AC Mode to match operation type if available (relaxed filtering)
                if forecast_operation_type and "A/C Mode" in candidate_block.columns:
                    # Get allowed AC Mode values (relaxed: COOL allows FAN, HEAT allows FAN)
                    allowed_modes = self._get_allowed_ac_modes(forecast_operation_type)

                    if allowed_modes:
                        # Only keep patterns with allowed AC Mode (vectorized with isin)
                        candidate_block = candidate_block[
                            candidate_block["A/C Mode"].isin(allowed_modes)
                        ].copy()

                        if len(candidate_block) == 0:
                            continue

                # Verify we have data for all required hours
                candidate_hours = sorted(candidate_block["hour"].unique())
                if set(candidate_hours) != set(hour_block):
                    # Missing some hours, skip this candidate
                    continue

                # Calculate weather distance for this candidate block
                # Note: forecast_block and candidate_block now have matching hours
                weather_distance = self._calculate_hour_block_distance(
                    forecast_block, candidate_block
                )

                # Calculate average power for this block
                avg_power = candidate_block["adjusted_power"].mean()

                best_block_candidates.append(
                    {
                        "weather_distance": weather_distance,
                        "avg_power": avg_power,
                        "block_data": candidate_block,
                        "hours": candidate_hours,
                    }
                )

            # Select best block (lowest weather distance, then lowest power)
            if len(best_block_candidates) > 0:
                best_candidate = min(
                    best_block_candidates,
                    key=lambda x: (x["weather_distance"], x["avg_power"]),
                )

                print(
                    f"  ✓ Selected best block: hours {best_candidate['hours']} "
                    f"(weather_distance={best_candidate['weather_distance']:.3f}, "
                    f"avg_power={best_candidate['avg_power']:.0f}W) "
                    f"from {len(best_block_candidates)} candidates"
                )

                # Direct hour matching: forecast hour maps to same historical hour
                # Since we ensured hours match exactly, we can map directly
                historical_block_data = best_candidate["block_data"]

                print(
                    f"  → Direct hour matching: forecast hours {sorted(hour_block)} → historical hours {sorted(best_candidate['hours'])} (exact match)"
                )

                # Forecast month for comfort range
                forecast_month = (
                    pd.to_datetime(forecast_block["datetime"].iloc[0]).month
                    if "datetime" in forecast_block.columns
                    else 1
                )

                # Map each forecast hour to its corresponding historical hour (same hour value)
                for forecast_hour in hour_block:
                    # Get the row(s) for this historical hour (same hour as forecast)
                    hist_rows = historical_block_data[
                        historical_block_data["hour"] == forecast_hour
                    ]

                    if len(hist_rows) == 0:
                        continue

                    # Filter by comfort range (historical indoor temp) when enabled
                    if (
                        self.use_comfort_filter
                        and master_data
                        and "Indoor Temp." in hist_rows.columns
                    ):
                        in_comfort_rows = self._filter_patterns_by_comfort(
                            hist_rows, zone, forecast_month, master_data
                        )
                        if len(in_comfort_rows) > 0:
                            hist_rows = in_comfort_rows

                    # Filter by AC Mode to match operation type if available (relaxed filtering)
                    if forecast_operation_type and "A/C Mode" in hist_rows.columns:
                        # Get allowed AC Mode values (relaxed: COOL allows FAN, HEAT allows FAN)
                        allowed_modes = self._get_allowed_ac_modes(
                            forecast_operation_type
                        )

                        if allowed_modes:
                            # Only keep patterns with allowed AC Mode (vectorized with isin)
                            hist_rows = hist_rows[
                                hist_rows["A/C Mode"].isin(allowed_modes)
                            ]

                            if len(hist_rows) == 0:
                                # No matching AC Mode for this hour, skip it
                                mode_names = [
                                    self._map_ac_mode(mode) for mode in allowed_modes
                                ]
                                logging.debug(
                                    f"Zone {zone}: No patterns with allowed AC Modes {mode_names} "
                                    f"({forecast_operation_type}) for hour {forecast_hour}"
                                )
                                continue

                    # If multiple rows for same hour, prefer target mode, then select lowest power
                    if len(hist_rows) > 1:
                        target_mode = None
                        if forecast_operation_type:
                            target_mode = self.OPERATION_TYPE_TO_MODE.get(
                                forecast_operation_type.upper()
                            )
                        if target_mode is not None:
                            # Sort: prefer target mode, then by power
                            hist_rows = hist_rows.copy()
                            hist_rows["mode_priority"] = (
                                hist_rows["A/C Mode"] != target_mode
                            ).astype(int)
                            hist_rows = hist_rows.sort_values(
                                ["mode_priority", "adjusted_power"]
                            )
                            # Select first row and drop the temporary priority column
                            hist_row = hist_rows.iloc[0].drop("mode_priority")
                        else:
                            # Fallback: sort by power only
                            hist_row = hist_rows.sort_values("adjusted_power").iloc[0]
                    else:
                        hist_row = hist_rows.iloc[0]

                    patterns_by_hour[forecast_hour] = hist_row
            else:
                print(f"  ✗ No valid candidates found for forecast hours {hour_block}")

        print(
            f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Successfully mapped {len(patterns_by_hour)}/{len(unique_forecast_hours)} hours"
        )

        return patterns_by_hour

    def _optimize_zone_with_model(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        zone: str,
        master_data: dict,
    ) -> Optional[pd.DataFrame]:
        """
        Model path: wider candidates (unique historical setting patterns),
        score each by model (predict power and temp under forecast), filter by comfort,
        select pattern with minimum predicted power. Returns None when model not available.
        """
        artifact = self._zone_models.get(zone) if self._zone_models else None
        if artifact is None:
            return None

        power_pipe = artifact.get("power_model")
        temp_pipe = artifact.get("temp_model")
        feature_names = artifact.get("feature_names", [])
        impute_means = artifact.get("impute_means", {})
        if not power_pipe or not feature_names:
            return None

        # Operating hours and forecast range (mirror fallback)
        if self.use_operating_hours:
            start_hour, end_hour = get_zone_operating_hours(master_data, zone)
        else:
            start_hour, end_hour = 0, 24

        forecast_df_original = forecast_df.copy()
        forecast_df_working = forecast_df.copy()
        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_working["_hour"] = pd.to_datetime(
                forecast_df_working["datetime"]
            ).dt.hour
            forecast_df_working = forecast_df_working[
                (forecast_df_working["_hour"] >= range_start)
                & (forecast_df_working["_hour"] < range_end)
            ].copy()
            forecast_df_working = forecast_df_working.drop(columns=["_hour"])

        # Wider candidates: unique (Set Temp, Mode, Fan Speed, ON/OFF) from history for this zone
        zone_hist = historical_df[historical_df["zone"] == zone].copy()
        if zone_hist.empty:
            return None

        # Optional: filter by operation type for forecast month (use first forecast day's month)
        forecast_month = (
            pd.to_datetime(forecast_df_working["datetime"].iloc[0]).month
            if len(forecast_df_working) > 0
            else None
        )
        if forecast_month is not None and self.operation_type_mapping:
            key = (zone, forecast_month)
            op_type = self.operation_type_mapping.get(key)
            if op_type:
                allowed = self._get_allowed_ac_modes(op_type)
                if allowed:
                    zone_hist = zone_hist[
                        zone_hist["A/C Mode"].fillna(-1).astype(int).isin(allowed)
                    ]
        if zone_hist.empty:
            return None

        # One row per unique setting combination; ensure A/C Status
        if "A/C Status" not in zone_hist.columns:
            zone_hist["A/C Status"] = np.where(
                (zone_hist["A/C ON/OFF"].fillna(0) == 0) | (zone_hist["A/C ON/OFF"].isna()),
                0,
                zone_hist["A/C Mode"].fillna(0),
            )
        key_cols = [
            "A/C Set Temperature",
            "A/C Mode",
            "A/C Fan Speed",
            "A/C ON/OFF",
            "A/C Status",
        ]
        key_cols = [c for c in key_cols if c in zone_hist.columns]
        candidates_df = (
            zone_hist.drop_duplicates(subset=key_cols, keep="first")
            .reset_index(drop=True)
        )
        if candidates_df.empty:
            return None

        # Sort forecast by datetime for sequential Indoor Temp. Lag1
        forecast_sorted = forecast_df_working.sort_values("datetime").reset_index(drop=True)
        results = []
        last_predicted_temp = impute_means.get("Indoor Temp. Lag1", 25.0)
        if pd.isna(last_predicted_temp):
            last_predicted_temp = 25.0

        zone_info = master_data.get("zones", {}).get(zone, {})
        max_units = sum(
            len(ou.get("indoor_units", []))
            for ou in zone_info.get("outdoor_units", {}).values()
        )
        if max_units <= 0:
            max_units = 1

        for _, forecast_row in forecast_sorted.iterrows():
            forecast_datetime = pd.to_datetime(forecast_row["datetime"])
            hour = forecast_datetime.hour
            if self.use_operating_hours and not (start_hour <= hour < end_hour):
                continue

            month = forecast_datetime.month
            day_of_week = forecast_datetime.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 0  # Could be from master/calendar if needed

            # Build feature matrix: one row per candidate
            rows = []
            for _, pat in candidates_df.iterrows():
                set_temp = pat.get("A/C Set Temperature")
                ac_status = pat.get("A/C Status")
                if pd.isna(ac_status):
                    ac_status = 0 if (pat.get("A/C ON/OFF", 0) == 0) else pat.get("A/C Mode", 0)
                fan_speed = pat.get("A/C Fan Speed", 0)
                if pd.isna(fan_speed):
                    fan_speed = 0
                out_temp = forecast_row.get("Outdoor Temp.")
                out_hum = forecast_row.get("Outdoor Humidity")
                solar = forecast_row.get("Solar Radiation")
                if pd.isna(out_temp):
                    out_temp = impute_means.get("Outdoor Temp.", 25.0)
                if pd.isna(out_hum):
                    out_hum = impute_means.get("Outdoor Humidity", 50.0)
                if pd.isna(solar):
                    solar = impute_means.get("Solar Radiation", 0.0)
                if pd.isna(set_temp):
                    set_temp = impute_means.get("A/C Set Temperature", 26.0)

                row = {
                    "A/C Set Temperature": float(set_temp),
                    "Indoor Temp. Lag1": float(last_predicted_temp),
                    "A/C Status": int(ac_status),
                    "A/C Fan Speed": int(fan_speed),
                    "Outdoor Temp.": float(out_temp),
                    "Outdoor Humidity": float(out_hum),
                    "Solar Radiation": float(solar),
                    "DayOfWeek": day_of_week,
                    "Hour": hour,
                    "Month": month,
                    "IsWeekend": is_weekend,
                    "IsHoliday": is_holiday,
                }
                rows.append(row)

            X = pd.DataFrame(rows)
            # Align columns to model's feature_names order; fill missing with impute
            for c in feature_names:
                if c not in X.columns:
                    X[c] = impute_means.get(c, 0)
            X = X[feature_names].astype(float)

            pred_power = power_pipe.predict(X)
            pred_temp = temp_pipe.predict(X) if temp_pipe else np.full(len(X), last_predicted_temp)

            # Comfort filter
            try:
                comfort_min, comfort_max = get_comfort_range(master_data, zone, month)
            except Exception:
                comfort_min, comfort_max = 22.0, 28.0
            in_comfort = (pred_temp >= comfort_min) & (pred_temp <= comfort_max)
            if not in_comfort.any():
                in_comfort = np.ones(len(pred_temp), dtype=bool)

            valid_power = np.where(in_comfort, pred_power, np.inf)
            best_idx = int(np.argmin(valid_power))
            best = candidates_df.iloc[best_idx]
            last_predicted_temp = float(pred_temp[best_idx])

            units_count = min(
                int(best.get("A/C ON/OFF", 0) or 0),
                max_units,
            )
            ac_mode_value = best.get("A/C Mode")
            if pd.isna(ac_mode_value):
                ac_mode_value = 0
            else:
                ac_mode_value = int(ac_mode_value)
            fan_speed_value = best.get("A/C Fan Speed")
            if pd.isna(fan_speed_value):
                fan_speed_value = 0
            else:
                fan_speed_value = int(fan_speed_value)

            # When AC is OFF, power must be 0 (model may predict non-zero)
            power_value = int(round(float(pred_power[best_idx])))if units_count > 0 else 0.0

            results.append({
                "datetime": forecast_datetime,
                "zone": zone,
                "set_temp": best["A/C Set Temperature"],
                "mode": self._map_ac_mode(ac_mode_value),
                "fan_speed": self._map_fan_speed(fan_speed_value),
                "numb_units_on": units_count,
                "ac_on_off": self._map_ac_on_off(units_count),
                "power": power_value,
                "indoor_temp": f"{float(last_predicted_temp):.1f}",
                "hist_datetime_used": None,
                "forecast_outdoor_temp": forecast_row.get("Outdoor Temp."),
                "forecast_solar_radiation": forecast_row.get("Solar Radiation"),
                "hist_outdoor_temp": forecast_row.get("Outdoor Temp."),
                "hist_solar_radiation": forecast_row.get("Solar Radiation"),
                "hist_indoor_temp": f"{float(last_predicted_temp):.1f}",
            })

        if not results:
            return None

        result_df = pd.DataFrame(results).sort_values("datetime").reset_index(drop=True)

        # Hours outside forecast_hour_range: add empty rows like fallback
        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_original["_hour"] = pd.to_datetime(
                forecast_df_original["datetime"]
            ).dt.hour
            outside = forecast_df_original[
                (forecast_df_original["_hour"] < range_start)
                | (forecast_df_original["_hour"] >= range_end)
            ].copy()
            for _, row in outside.iterrows():
                dt = pd.to_datetime(row["datetime"])
                results.append({
                    "datetime": dt,
                    "zone": zone,
                    "set_temp": None,
                    "mode": None,
                    "fan_speed": None,
                    "numb_units_on": None,
                    "ac_on_off": None,
                    "power": None,
                    "indoor_temp": None,
                    "hist_datetime_used": None,
                    "forecast_outdoor_temp": row.get("Outdoor Temp."),
                    "forecast_solar_radiation": row.get("Solar Radiation"),
                    "hist_outdoor_temp": None,
                    "hist_solar_radiation": None,
                    "hist_indoor_temp": None,
                })
            result_df = pd.DataFrame(results).sort_values("datetime").reset_index(drop=True)

        print(f"\n[Zone: {zone}] Model path: {len(result_df)} hours scheduled")
        return result_df

    def _optimize_zone_for_forecast(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        zone: str,
        master_data: dict,
    ) -> pd.DataFrame:
        """
        Main optimization function for a single zone.

        Supports two modes:
        - Whole day mode (hour_block_size=None): Selects complete historical days
        - Hour block mode (hour_block_size is integer): Selects best N-hour blocks from candidate days

        Args:
            historical_df: Historical patterns DataFrame (already filtered for AC ON)
            forecast_df: Forecast weather DataFrame
            zone: Zone name to optimize
            master_data: Master data dictionary

        Returns:
            DataFrame with optimization results for the zone
        """
        # Get zone operating hours from master data
        if self.use_operating_hours:
            start_hour, end_hour = get_zone_operating_hours(master_data, zone)
        else:
            start_hour, end_hour = 0, 24  # All 24 hours

        # Apply forecast_hour_range filter if specified
        forecast_df_original = forecast_df.copy()
        forecast_df_working = forecast_df.copy()

        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_working["hour"] = pd.to_datetime(
                forecast_df_working["datetime"]
            ).dt.hour
            forecast_df_working = forecast_df_working[
                (forecast_df_working["hour"] >= range_start)
                & (forecast_df_working["hour"] < range_end)
            ].copy()
            forecast_df_working = forecast_df_working.drop(columns=["hour"])

        results = []
        stats = {
            "total_hours": 0,
            "outside_op_hours": 0,
            "no_similar_patterns": 0,
            "success": 0,
        }

        # Track detailed failures with timestamps and reasons
        detailed_failures = {
            "outside_op": [],
            "no_patterns": [],
        }

        # Group forecast by day for daily-level processing
        forecast_df_working["Date"] = pd.to_datetime(
            forecast_df_working["datetime"]
        ).dt.date

        forecast_days = forecast_df_working.groupby("Date")

        # Process each forecast day
        for forecast_date, forecast_day_data in forecast_days:
            # Find similar historical days for this forecast day
            top_days = self._find_similar_days(
                historical_df, forecast_day_data, zone, n_top=20
            )

            # Select patterns based on mode
            if self.hour_block_size is None:
                # Whole day mode: Select best complete day
                best_day, day_patterns = self._select_best_complete_day(
                    historical_df, zone, top_days, forecast_day_data, master_data
                )

                # Create a lookup dictionary from DataFrame for quick access by hour
                patterns_by_hour = {}
                if not day_patterns.empty and "hour" in day_patterns.columns:
                    for _, row in day_patterns.iterrows():
                        patterns_by_hour[row["hour"]] = row
            else:
                # Hour block mode: Select best N-hour blocks
                patterns_by_hour = self._select_best_hour_blocks(
                    historical_df, zone, top_days, forecast_day_data, master_data
                )

            # Process each hour in the forecast day
            for _, forecast_row in forecast_day_data.iterrows():
                forecast_datetime = pd.to_datetime(forecast_row["datetime"])
                hour = forecast_datetime.hour
                stats["total_hours"] += 1

                # Check if hour is within operating hours
                if self.use_operating_hours and not (start_hour <= hour < end_hour):
                    stats["outside_op_hours"] += 1
                    detailed_failures["outside_op"].append(
                        f"{forecast_datetime.strftime('%m/%d %H:00')}"
                    )
                    continue

                # Get pattern for this hour
                if hour not in patterns_by_hour:
                    stats["no_similar_patterns"] += 1
                    detailed_failures["no_patterns"].append(
                        f"{forecast_datetime.strftime('%m/%d %H:00')}"
                    )
                    continue

                # Use the pattern
                best_pattern = patterns_by_hour[hour]

                stats["success"] += 1

                # Extract recommended settings from the pattern
                # prevent physically impossible unit counts
                # getting number of indoor units from master
                zone_info = master_data["zones"].get(zone, {})
                max_units = sum(
                    len(ou.get("indoor_units", []))
                    for ou in zone_info.get("outdoor_units", {}).values()
                )

                units_count = min(
                    int(best_pattern.get("A/C ON/OFF", 0)),
                    max_units
                )                
                # Handle NaN values before converting to int
                ac_mode_value = best_pattern["A/C Mode"]
                if pd.isna(ac_mode_value):
                    # Default to OFF (0) if AC Mode is NaN
                    ac_mode_value = 0
                else:
                    ac_mode_value = int(ac_mode_value)
                
                fan_speed_value = best_pattern["A/C Fan Speed"]
                if pd.isna(fan_speed_value):
                    # Default to AUTO (typically 0 or 1) if Fan Speed is NaN
                    fan_speed_value = 0
                else:
                    fan_speed_value = int(fan_speed_value)
                
                result = {
                    "datetime": forecast_datetime,
                    "zone": zone,
                    "set_temp": best_pattern["A/C Set Temperature"],
                    "mode": self._map_ac_mode(ac_mode_value),
                    "fan_speed": self._map_fan_speed(fan_speed_value),
                    "numb_units_on": units_count,
                    "ac_on_off": self._map_ac_on_off(units_count),
                    "power": best_pattern["adjusted_power"],
                    "indoor_temp": best_pattern["Indoor Temp."],
                    "hist_datetime_used": best_pattern["Datetime"],
                    "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                    "forecast_solar_radiation": forecast_row["Solar Radiation"],
                    "hist_outdoor_temp": best_pattern["Outdoor Temp."],
                    "hist_solar_radiation": best_pattern["Solar Radiation"],
                    "hist_indoor_temp": best_pattern["Indoor Temp."],
                }

                results.append(result)

        # If forecast_hour_range is specified, add empty results for hours outside range
        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_original["hour"] = pd.to_datetime(
                forecast_df_original["datetime"]
            ).dt.hour
            outside_range_df = forecast_df_original[
                (forecast_df_original["hour"] < range_start)
                | (forecast_df_original["hour"] >= range_end)
            ].copy()

            for _, forecast_row in outside_range_df.iterrows():
                forecast_datetime = pd.to_datetime(forecast_row["datetime"])
                result = {
                    "datetime": forecast_datetime,
                    "zone": zone,
                    "set_temp": None,
                    "mode": None,
                    "fan_speed": None,
                    "numb_units_on": None,
                    "ac_on_off": None,
                    "power": None,
                    "indoor_temp": None,
                    "hist_datetime_used": None,
                    "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                    "forecast_solar_radiation": forecast_row["Solar Radiation"],
                    "hist_outdoor_temp": None,
                    "hist_solar_radiation": None,
                    "hist_indoor_temp": None,
                }
                results.append(result)

        result_df = pd.DataFrame(results)

        # ---------------------------------------
        # Calculate success rate and print detailed summary
        # ---------------------------------------
        valid_hours = stats["total_hours"] - stats["outside_op_hours"]
        success_rate = (stats["success"] / valid_hours * 100) if valid_hours > 0 else 0

        # Print detailed summary
        print(f"\n━━━━━ [{zone}] Summary ━━━━━")
        print(f"✅ 成功: {len(result_df)}時間 ({success_rate:.0f}%)")

        if stats["outside_op_hours"] > 0:
            print(f"⏰ 営業時間外: {stats['outside_op_hours']}時間")
            if len(detailed_failures["outside_op"]) <= 10:
                print(f"   {', '.join(detailed_failures['outside_op'])}")

        if stats["no_similar_patterns"] > 0:
            print(f"🔍 パターンなし: {stats['no_similar_patterns']}時間")
            if len(detailed_failures["no_patterns"]) <= 10:
                print(f"   {', '.join(detailed_failures['no_patterns'])}")

        print("━" * 30)

        return result_df

    def optimize_all_zones(
        self, forecast_df: pd.DataFrame, features_csv_path: str, master_data: dict
    ) -> pd.DataFrame:
        """
        Optimize all zones for the forecast period.

        Uses the configured optimization mode (whole day or hour block) and
        applies forecast hour range filtering if specified.

        Args:
            forecast_df: Forecast weather DataFrame
            features_csv_path: Path to features CSV file
            master_data: Master data dictionary

        Returns:
            Combined DataFrame with optimization results for all zones in wide format
        """
        # Load historical patterns
        historical_df = self.load_historical_patterns(features_csv_path)

        # 12 month filter
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        historical_df["Datetime"] = pd.to_datetime(historical_df["Datetime"])
        forecast_max_date = forecast_df["datetime"].max()
        twelve_months_ago = forecast_max_date - pd.DateOffset(months=12)
        historical_df = historical_df[
            ~(
                (historical_df["zone"].astype(str).str.strip().str.lower() == "area 1")
                & (historical_df["Datetime"] < twelve_months_ago)
            )
        ].copy()

        # Data size check: use model path only when enough data and model available
        n_rows = len(historical_df)
        n_days = historical_df["Date"].nunique() if "Date" in historical_df.columns else 0
        data_ok = (
            n_rows >= self.MIN_HISTORICAL_ROWS
            and n_days >= self.MIN_HISTORICAL_DAYS
        )

        if data_ok and self.store_name:
            self._ensure_models_loaded(self.store_name)
        use_model_path = data_ok and self._model_available()
        # use_model_path = data_ok and False
    
        if use_model_path:
            print(
                f"[Optimizer] Data size OK (rows={n_rows}, days={n_days}); using model path."
            )
        else:
            print(
                f"[Optimizer] Fallback path: data_ok={data_ok} (rows={n_rows}, days={n_days}), "
                f"model_available={self._model_available()}. "
                f"Similar-day + historical power + comfort filter."
            )
        # Get list of all zones from historical data
        zones = sorted(historical_df["zone"].unique())
        zones = [z for z in self.ZONE_ORDER if z in historical_df["zone"].unique()]
        all_results = []

        # Optimize each zone
        for zone in zones:
            zone_results = None
            if use_model_path:
                zone_results = self._optimize_zone_with_model(
                    historical_df, forecast_df, zone, master_data
                )
            if zone_results is None or len(zone_results) == 0:
                zone_results = self._optimize_zone_for_forecast(
                    historical_df, forecast_df, zone, master_data
                )
            if len(zone_results) > 0:
                all_results.append(zone_results)

        if not all_results:
            print("いずれのゾーンでも最適化結果は生成されませんでした")
            return pd.DataFrame()

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Sort by datetime and zone
        combined_results = combined_results.sort_values(
            ["datetime", "zone"]
        ).reset_index(drop=True)

        # Post-process results: replace COOL/HEAT with OFF status to FAN with AUTO
        combined_results = self._post_process_results(combined_results)

        print(f"\n=== 最適化サマリー ===")
        print(f"合計結果: {len(combined_results)}時間")
        for zone in zones:
            zone_count = len(combined_results[combined_results["zone"] == zone])
            print(f"  {zone}: {zone_count}時間")

        # Convert to wide format
        wide_df = self._convert_to_wide_format(combined_results)

        return wide_df

    def _post_process_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process optimization results to replace COOL/HEAT with OFF status
        with FAN mode, AUTO fan speed, and zero power.

        This ensures that if a COOL or HEAT mode is selected but AC is OFF,
        it is replaced with FAN mode (AUTO fan speed) with zero power consumption.

        Args:
            results_df: Long format DataFrame with optimization results

        Returns:
            Post-processed DataFrame with corrections applied
        """
        if results_df.empty:
            return results_df

        # Create a copy to avoid modifying the original
        processed_df = results_df.copy()

        # Identify rows where mode is COOL or HEAT AND status is OFF
        # Check both ac_on_off == "OFF" and numb_units_on == 0 or None
        is_cool_or_heat = processed_df["mode"].isin(["COOL", "HEAT"])
        is_off_status = (
            (processed_df["ac_on_off"] == "OFF")
            | (processed_df["numb_units_on"] == 0)
            | (processed_df["numb_units_on"].isna())
        )

        # Find rows that need correction
        needs_correction = is_cool_or_heat & is_off_status

        if needs_correction.any():
            correction_count = needs_correction.sum()
            logging.info(
                f"Post-processing: Replacing {correction_count} COOL/HEAT with OFF status "
                "to FAN mode with AUTO fan speed and zero power"
            )

            # Apply corrections using vectorized operations
            processed_df.loc[needs_correction, "mode"] = "FAN"
            processed_df.loc[needs_correction, "fan_speed"] = "AUTO"
            processed_df.loc[needs_correction, "power"] = 0.0

            # Log zone-wise statistics
            zone_stats = processed_df[needs_correction].groupby("zone").size().to_dict()
            for zone, count in zone_stats.items():
                logging.debug(f"  Zone {zone}: {count} corrections applied")

        return processed_df

    def _convert_to_wide_format(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long format optimization results to wide format.

        Args:
            long_df: Long format DataFrame with zone column

        Returns:
            Wide format DataFrame with zone-specific columns
        """
        if long_df.empty:
            return pd.DataFrame()

        # Create a base DataFrame with datetime and forecast data
        base_df = (
            long_df[["datetime", "forecast_outdoor_temp", "forecast_solar_radiation"]]
            .drop_duplicates()
            .copy()
        )
        base_df = base_df.sort_values("datetime").reset_index(drop=True)

        # Get unique zones
        zones = [
            z for z in self.ZONE_ORDER
            if z in long_df["zone"].unique()
        ]
        
        # Create zone-specific columns for each AC setting
        ac_settings = [
            "set_temp",
            "mode",
            "fan_speed",
            "numb_units_on",
            "ac_on_off",
            "power",
            "indoor_temp",
            # "similarity_score",  # Set to None for day-level selection
            "hist_outdoor_temp",
            "hist_solar_radiation",
            "hist_indoor_temp",
            "hist_datetime_used",
            # "hist_day_used",  # Track which historical day was used (same for all hours in forecast day)
        ]

        for zone in zones:
            zone_data = long_df[long_df["zone"] == zone].copy()

            # Merge zone data with base dataframe
            zone_data = zone_data[["datetime"] + ac_settings].copy()

            # Rename columns to include zone name
            rename_dict = {col: f"{zone}_{col}" for col in ac_settings}
            zone_data = zone_data.rename(columns=rename_dict)

            # Merge with base dataframe
            base_df = base_df.merge(zone_data, on="datetime", how="left")

        # Sort by datetime
        base_df = base_df.sort_values("datetime").reset_index(drop=True)

        logging.info(
            f"Converted to wide format: {len(base_df)} rows, {len(base_df.columns)} columns"
        )

        return base_df

    def get_long_format(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format optimization results to long format.

        This is a public method that can be called after optimize_all_zones to get
        long format output where each row represents one zone at one datetime.

        Args:
            wide_df: Wide format DataFrame (from optimize_all_zones)

        Returns:
            Long format DataFrame with zone column and one row per zone per datetime
        """
        return self._convert_to_long_format(wide_df)

    def _convert_to_long_format(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format optimization results to long format.

        This method converts zone-specific columns (e.g., "Area 1_set_temp", "Area 1_mode")
        back to a long format where each row represents one zone at one datetime.

        Args:
            wide_df: Wide format DataFrame with zone-specific columns

        Returns:
            Long format DataFrame with zone column and standardized column names
        """
        if wide_df.empty:
            return pd.DataFrame()

        # Base columns that are not zone-specific
        base_columns = ["datetime", "forecast_outdoor_temp", "forecast_solar_radiation"]
        
        # AC settings that have zone prefixes
        ac_settings = [
            "set_temp",
            "mode",
            "fan_speed",
            "numb_units_on",
            "ac_on_off",
            "power",
            "indoor_temp",
            "hist_outdoor_temp",
            "hist_solar_radiation",
            "hist_indoor_temp",
            "hist_datetime_used",
        ]

        # Find all zones by looking for columns with zone prefixes
        zone_columns = [col for col in wide_df.columns if any(col.startswith(f"{zone}_") for zone in self.ZONE_ORDER)]
        
        # Extract unique zones from column names
        zones_found = set()
        for col in zone_columns:
            for zone in self.ZONE_ORDER:
                if col.startswith(f"{zone}_"):
                    zones_found.add(zone)
                    break

        # Order zones according to ZONE_ORDER
        zones = [zone for zone in self.ZONE_ORDER if zone in zones_found]

        all_long_rows = []

        # For each datetime row in wide format, create a row for each zone
        for _, row in wide_df.iterrows():
            datetime_value = row["datetime"]
            
            for zone in zones:
                zone_prefix = f"{zone}_"
                
                # Build a row for this zone
                zone_row = {
                    "datetime": datetime_value,
                    "zone": zone,
                }
                
                # Add base columns if they exist in the wide DataFrame
                if "forecast_outdoor_temp" in wide_df.columns:
                    zone_row["forecast_outdoor_temp"] = row["forecast_outdoor_temp"]
                if "forecast_solar_radiation" in wide_df.columns:
                    zone_row["forecast_solar_radiation"] = row["forecast_solar_radiation"]
                
                # Extract zone-specific columns
                for setting in ac_settings:
                    zone_col = f"{zone_prefix}{setting}"
                    if zone_col in wide_df.columns:
                        zone_row[setting] = row[zone_col]
                    else:
                        zone_row[setting] = None
                
                all_long_rows.append(zone_row)

        long_df = pd.DataFrame(all_long_rows)

        # Sort by datetime and zone
        long_df = long_df.sort_values(["datetime", "zone"]).reset_index(drop=True)

        logging.info(
            f"Converted to long format: {len(long_df)} rows, {len(long_df.columns)} columns"
        )

        return long_df

    def get_unit_format(
        self, zone_wide_df: pd.DataFrame, master_data: dict
    ) -> pd.DataFrame:
        """
        Convert zone-level wide format DataFrame to unit-level wide format DataFrame.

        This is a public method that can be called after optimize_all_zones to get
        unit-level output format matching unit_schedule CSV format.

        Args:
            zone_wide_df: Zone-level wide format DataFrame (from optimize_all_zones)
            master_data: Master data dictionary containing zone-to-unit mappings

        Returns:
            Unit-level wide format DataFrame with unit-specific columns
        """
        return self._convert_to_unit_format(zone_wide_df, master_data)

    def _convert_to_unit_format(
        self, zone_wide_df: pd.DataFrame, master_data: dict
    ) -> pd.DataFrame:
        """
        Convert zone-level wide format DataFrame to unit-level wide format DataFrame.

        This method maps each zone to its indoor units from master data and creates
        unit-specific columns by duplicating zone data for each unit in that zone.

        The output format matches unit_schedule CSV format:
        - Date Time column in "YYYY/MM/DD HH:MM" format
        - For each unit: {unit_name}_OnOFF, {unit_name}_Mode, {unit_name}_SetTemp, {unit_name}_FanSpeed

        Args:
            zone_wide_df: Zone-level wide format DataFrame (from _convert_to_wide_format)
            master_data: Master data dictionary containing zone-to-unit mappings

        Returns:
            Unit-level wide format DataFrame with unit-specific columns
        """
        if zone_wide_df.empty:
            return pd.DataFrame()

        # Build zone-to-units mapping from master data
        zones = master_data.get("zones", {})
        zone_to_units: Dict[str, List[str]] = {}
        for zone_name, zone_info in zones.items():
            units = []
            for _, outdoor_unit_info in zone_info.get("outdoor_units", {}).items():
                units.extend(outdoor_unit_info.get("indoor_units", []))
            # Remove duplicates while preserving order
            zone_to_units[zone_name] = list(dict.fromkeys(units))

        if not zone_to_units:
            logging.warning(
                "No zone-to-unit mapping found in master data. Cannot convert to unit format."
            )
            return pd.DataFrame()

        # Create base DataFrame with Date Time column (matching unit_schedule format)
        base_df = zone_wide_df[["datetime"]].copy()
        base_df["Date Time"] = base_df["datetime"].dt.strftime("%Y/%m/%d %H:%M")
        base_df = base_df[["Date Time"]].copy()

        # For each zone, get its units and create unit-level columns
        for zone_name, units in zone_to_units.items():
            if not units:
                logging.warning(f"Zone {zone_name} has no indoor units, skipping")
                continue

            # Get zone data columns
            zone_prefix = f"{zone_name}_"
            zone_columns = [
                col for col in zone_wide_df.columns if col.startswith(zone_prefix)
            ]

            if not zone_columns:
                logging.warning(
                    f"No data found for zone {zone_name} in zone_wide_df, skipping"
                )
                continue

            # Extract zone data
            zone_data = zone_wide_df[["datetime"] + zone_columns].copy()

            # Get zone-specific values
            zone_set_temp_col = f"{zone_prefix}set_temp"
            zone_mode_col = f"{zone_prefix}mode"
            zone_fan_speed_col = f"{zone_prefix}fan_speed"
            zone_numb_units_on_col = f"{zone_prefix}numb_units_on"
            zone_ac_on_off_col = f"{zone_prefix}ac_on_off"

            # For each unit in the zone, create unit-specific columns
            for unit_name in units:
                # OnOFF: Use zone's ac_on_off value, or derive from numb_units_on
                if zone_ac_on_off_col in zone_wide_df.columns:
                    base_df[f"{unit_name}_OnOFF"] = zone_wide_df[
                        zone_ac_on_off_col
                    ].fillna("OFF")
                elif zone_numb_units_on_col in zone_wide_df.columns:
                    # Convert numb_units_on to OnOFF string
                    base_df[f"{unit_name}_OnOFF"] = zone_wide_df[
                        zone_numb_units_on_col
                    ].apply(lambda x: "ON" if pd.notna(x) and float(x) > 0 else "OFF")
                else:
                    base_df[f"{unit_name}_OnOFF"] = "OFF"

                # Mode: Use zone's mode value, or set to OFF if OnOFF is OFF
                if zone_mode_col in zone_wide_df.columns:
                    mode_values = zone_wide_df[zone_mode_col].copy()
                    # If OnOFF is OFF, set Mode to OFF
                    if f"{unit_name}_OnOFF" in base_df.columns:
                        mode_values = mode_values.where(
                            base_df[f"{unit_name}_OnOFF"] != "OFF", "OFF"
                        )
                    base_df[f"{unit_name}_Mode"] = mode_values.fillna("OFF")
                else:
                    base_df[f"{unit_name}_Mode"] = "OFF"

                # SetTemp: Use zone's set_temp value
                if zone_set_temp_col in zone_wide_df.columns:
                    base_df[f"{unit_name}_SetTemp"] = zone_wide_df[zone_set_temp_col]
                else:
                    base_df[f"{unit_name}_SetTemp"] = None

                # FanSpeed: Use zone's fan_speed value
                if zone_fan_speed_col in zone_wide_df.columns:
                    base_df[f"{unit_name}_FanSpeed"] = zone_wide_df[zone_fan_speed_col]
                else:
                    base_df[f"{unit_name}_FanSpeed"] = None

        # Sort by Date Time
        base_df = base_df.sort_values("Date Time").reset_index(drop=True)

        logging.info(
            f"Converted to unit format: {len(base_df)} rows, {len(base_df.columns)} columns"
        )

        return base_df
        