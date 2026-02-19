"""
Simplified optimization runner that orchestrates zone-based optimization.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

from config.config_gcp import GCPEnv
from config.utils import get_data_path, get_weather_forecast_path
from processing.utilities.master_data_loader import master_data_loader_runner
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher
from service.gdrive import GoogleDriveClient
from service.storage import get_storage_client

from .optimizer import Optimizer


class OptimizerRunner:
    """
    Main runner that coordinates zone-based optimization.
    Uses historical data and weather forecasts to determine optimal HVAC operation.
    """

    def __init__(self, store_name: str = "Clea"):
        """
        Initialize the optimization runner.

        Args:
            store_name: Store name (default: Clea)
        """
        self.store_name = store_name
        self.master_data: Dict[str, Any] = {}
        self.weather_data: pd.DataFrame = pd.DataFrame()
        self.results: Dict[str, Any] = {}

    def load_weather_data(self, start_date: str, end_date: str = None):
        """
        Load master data and fetch forecast weather data using weather API.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)
        """
        from datetime import datetime

        # Set end_date to start_date if not provided
        if end_date is None:
            end_date = start_date

        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError(
                    f"Start date {start_date} cannot be after end date {end_date}"
                )
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError(
                    f"Invalid date format. Use YYYY-MM-DD format. Error: {e}"
                )
            else:
                raise e

        # Load master data
        self.master_data = master_data_loader_runner(self.store_name)
        if not self.master_data:
            raise ValueError(f"Failed to load master data for store {self.store_name}")

        # Load forecast weather data using weather API
        coordinates = self.master_data["store_info"]["coordinates"]
        if not coordinates:
            raise ValueError(f"No coordinates found for store {self.store_name}")

        # Resolve API key: Secret Manager when STORAGE_BACKEND=gcs, else local config
        api_key = None
        try:
            backend = os.getenv("STORAGE_BACKEND", "local").lower()
            if backend == "gcs":
                from service.secretmanager import SecretManagerClient

                sm = SecretManagerClient()
                api_key = sm.get_secret_as_str("WEATHER_API_KEY")
        except Exception:
            api_key = None
        if not api_key:
            # for local development
            from config.private_information import WEATHER_API_KEY

            api_key = WEATHER_API_KEY

        # Check for cached weather forecast first (storage-backed)
        storage = get_storage_client()
        # Get weather forecast path from config
        cached_logical_path = get_weather_forecast_path(
            store_name=self.store_name, start_date=start_date, end_date=end_date
        )

        try:
            self.weather_data = storage.read_csv(cached_logical_path)
            if "datetime" in self.weather_data.columns:
                self.weather_data["datetime"] = pd.to_datetime(
                    self.weather_data["datetime"]
                )
            print(
                f"[OptimizerRunner] Cached weather data loaded. Shape: {self.weather_data.shape}"
            )
            return
        except Exception:
            pass

        print(
            f"[OptimizerRunner] Fetching weather forecast from {start_date} to {end_date}"
        )
        print(f"[OptimizerRunner] Using coordinates: {coordinates}")

        # Create weather API client and fetch forecast data
        weather_fetcher = VisualCrossingWeatherAPIDataFetcher(
            coordinates=coordinates,
            start_date=start_date,
            end_date=end_date,
            unit="metric",
            api_key=api_key,
            temperature_col_name="Outdoor Temp.",
            humidity_col_name="Outdoor Humidity",
            solar_col_name="Solar Radiation",
            batch_size_months=1,
            delay_between_requests=1.0,
        )

        # Fetch weather data
        self.weather_data = weather_fetcher.fetch()

        if self.weather_data is None or self.weather_data.empty:
            raise ValueError(
                f"Failed to fetch weather data for period {start_date} to {end_date}"
            )

        # Save to cache via storage client
        try:
            storage.write_csv(self.weather_data, cached_logical_path)
            print(
                f"[OptimizerRunner] Weather forecast cached to: {cached_logical_path}"
            )
        except Exception as e:
            print(f"[OptimizerRunner] Error saving weather forecast to cache: {e}")

        print(f"[OptimizerRunner] Loaded master data for store {self.store_name}")
        print(
            f"[OptimizerRunner] Fetched weather forecast for {start_date} to {end_date}: {len(self.weather_data)} hours"
        )

    def run_optimization(
        self, start_date: str = None, end_date: str = None, strategy: str = "hourly"
    ) -> Dict[str, Any]:
        """
        Run the zone-based optimization for the specified date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)

        Returns:
            Dictionary containing optimization results
        """
        # load processed features CSV (storage-backed). Optimizer needs a file path, so
        # for GCS we download to a temp file.
        storage = get_storage_client()
        features_logical_path = f"02_PreprocessedData/{self.store_name}/features_processed_{self.store_name}.csv"
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        if backend == "gcs":
            import tempfile

            tmp_fd, tmp_path = tempfile.mkstemp(prefix="features_", suffix=".csv")
            os.close(tmp_fd)
            # write bytes to tmp via storage.read_bytes
            features_bytes = storage.read_bytes(features_logical_path)
            with open(tmp_path, "wb") as f:
                f.write(features_bytes)
            self.features_csv_path = tmp_path
        else:
            local_root = os.getenv("LOCAL_DATA_ROOT", os.path.join(os.getcwd(), "data"))
            self.features_csv_path = os.path.join(local_root, features_logical_path)
            if not os.path.exists(self.features_csv_path):
                raise FileNotFoundError(
                    f"Features CSV not found: {self.features_csv_path}"
                )

        # Set default dates if not provided (today + 3 days)
        if start_date is None:
            today = datetime.now().date()
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + timedelta(days=3)).strftime("%Y-%m-%d")
            print(
                f"[OptimizerRunner] No dates provided, using default period: {start_date} to {end_date}"
            )
        elif end_date is None:
            end_date = start_date

        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError(
                    f"Start date {start_date} cannot be after end date {end_date}"
                )
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError(
                    f"Invalid date format. Use YYYY-MM-DD format. Error: {e}"
                )
            else:
                raise e

        print(
            f"[OptimizerRunner] Running zone-based optimization for {start_date} to {end_date}..."
        )
        try:
            # Load data (master data and weather forecast)
            self.load_weather_data(start_date, end_date)

            # Run optimization twice: model-based and legacy fallback (for comparison)
            optimizer_model = Optimizer(
                store_name=self.store_name,
                use_operating_hours=False,
                use_model_path=True,
            )
            result_model = optimizer_model.optimize_all_zones(
                forecast_df=self.weather_data,
                features_csv_path=self.features_csv_path,
                master_data=self.master_data,
            )

            optimizer_fallback = Optimizer(
                store_name=self.store_name,
                use_operating_hours=False,
                use_model_path=False,
            )
            result_fallback = optimizer_fallback.optimize_all_zones(
                forecast_df=self.weather_data,
                features_csv_path=self.features_csv_path,
                master_data=self.master_data,
            )

            if result_fallback.empty and result_model.empty:
                print("[OptimizerRunner] Zone optimization returned no data")
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": "No optimization data",
                }

            self.optimizer = optimizer_fallback
            self.results["optimization_result"] = result_fallback
            self.results["optimization_result_model"] = result_model
            self.results["optimization_result_fallback"] = result_fallback
            self.results["status"] = "success"
            self.results["start_date"] = start_date
            self.results["end_date"] = end_date

            # Unit-level format from fallback (backward compatibility)
            try:
                unit_result_df = self.optimizer.get_unit_format(
                    result_fallback, self.master_data
                )
                if not unit_result_df.empty:
                    self.results["optimization_result_units"] = unit_result_df
                    print(
                        f"[OptimizerRunner] Unit-level optimization result generated: {len(unit_result_df)} results"
                    )
                else:
                    print(
                        "[OptimizerRunner] Warning: Unit-level optimization result is empty"
                    )
            except Exception as e:
                print(
                    f"[OptimizerRunner] Warning: Failed to generate unit-level format: {e}"
                )

            print(
                f"[OptimizerRunner] Zone optimization completed: model={len(result_model)} rows, fallback={len(result_fallback)} rows"
            )
            return self.results

        except Exception as e:
            print(f"[OptimizerRunner] Zone optimization failed: {e}")
            return {
                "start_date": start_date,
                "end_date": end_date,
                "error": f"Optimization failed: {str(e)}",
            }

    def _remove_hist_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that contain '_hist_' in their name.
        Used for Google Drive upload to exclude historical columns.

        Args:
            dataframe: DataFrame to filter

        Returns:
            DataFrame with '_hist_' columns removed
        """
        columns_to_keep = [
            column
            for column in dataframe.columns
            if not any(substring in column for substring in ["_hist_", "_power", "_indoor_temp"])
        ]

        filtered_dataframe = dataframe[columns_to_keep].copy()
        removed_count = len(dataframe.columns) - len(filtered_dataframe.columns)
        if removed_count > 0:
            print(
                f"[OptimizerRunner] Removed {removed_count} column(s) containing '_hist_' for Google Drive upload"
            )
        return filtered_dataframe

    def _upload_to_google_drive(self, dataframe: pd.DataFrame, filename: str):
        """
        Upload optimization results DataFrame to Google Drive.

        Args:
            dataframe: DataFrame to upload
            filename: Filename for the uploaded file
        """
        # Resolve service account JSON: Secret Manager when STORAGE_BACKEND=gcs, else local file
        service_account_json = None
        backend = os.getenv("STORAGE_BACKEND", "local").lower()

        if backend == "gcs":
            try:
                from service.secretmanager import SecretManagerClient

                sm = SecretManagerClient()
                secret_name = GCPEnv.SERVICE_ACCOUNT_SECRET_NAME
                service_account_json = sm.get_secret_as_str(secret_name)
                if not service_account_json:
                    raise ValueError(
                        f"{secret_name} secret not found in Secret Manager"
                    )
            except Exception as e:
                secret_name = GCPEnv.SERVICE_ACCOUNT_SECRET_NAME
                raise ValueError(
                    f"Failed to load Google Drive service account from Secret Manager: {e}. "
                    f"Ensure {secret_name} secret exists in Secret Manager."
                )

        # Fallback to local file only for local backend
        if not service_account_json and backend != "gcs":
            service_account_path = GCPEnv.SERVICE_ACCOUNT_JSON
            # Handle both relative and absolute paths
            if not os.path.isabs(service_account_path):
                # Get project root (parent of optimization directory)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_file_dir)
                service_account_path = os.path.join(project_root, service_account_path)
            try:
                with open(service_account_path, "r") as f:
                    service_account_json = f.read()
            except Exception as e:
                raise ValueError(
                    f"Failed to load service account JSON from {service_account_path}: {e}"
                )

        # Get folder ID and encoding from config
        folder_id = GCPEnv.CLEA_OUT_GDRIVE_FOLDER_ID
        encoding = GCPEnv.CSV_ENCODING

        # Remove _hist_ columns for Google Drive upload (original CSV remains unchanged)
        filtered_dataframe = self._remove_hist_columns(dataframe)

        # Upload to Google Drive
        print(
            f"[OptimizerRunner] Uploading {filename} to Google Drive folder: {folder_id}"
        )
        drive_client = GoogleDriveClient(
            service_account_json=service_account_json,
            folder_id=folder_id,
            encoding=encoding,
        )
        file_id = drive_client.upload_file(
            filtered_dataframe, filename, encoding=encoding
        )
        print(
            f"[OptimizerRunner] Successfully uploaded to Google Drive. File ID: {file_id}"
        )

    def save_results_to_csv(
        self, start_date: str = None, end_date: str = None, filename: str = None
    ) -> str:
        """
        Save optimization results to CSV (both zone-level and unit-level if available).

        Args:
            start_date: Start date in YYYY-MM-DD format (uses result data if None)
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)
            filename: Optional custom CSV filename (for zone-level output)

        Returns:
            Path to the saved zone-level CSV file
        """
        if "optimization_result" not in self.results:
            raise ValueError("No optimization results to save. Run optimization first.")

        # Use dates from results if not provided
        if start_date is None:
            start_date = self.results.get("start_date", "unknown")
        if end_date is None:
            end_date = self.results.get("end_date", start_date)

        # Output path (storage-backed)
        storage = get_storage_client()

        start_date_formatted = start_date.replace("-", "")
        end_date_formatted = end_date.replace("-", "")
        date_suffix = f"{start_date_formatted}_{end_date_formatted}"

        if filename is None:
            filename = f"zone_schedule_{date_suffix}.csv"
        output_logical_path = f"04_PlanningData/{self.store_name}/{filename}"

        print(
            f"[OptimizerRunner] Saving zone-level optimization results to storage path: {output_logical_path}"
        )
        try:
            storage.write_csv(self.results["optimization_result"], output_logical_path)
        except Exception as error:
            print(
                f"[OptimizerRunner] Failed to save optimization results to {output_logical_path}: {error}"
            )
            raise
        print(
            f"[OptimizerRunner] Zone-level optimization results saved to: {output_logical_path}"
        )

        # Save model and fallback outputs for comparison (2 outputs for compare_optimization_outputs.py)
        if "optimization_result_model" in self.results and "optimization_result_fallback" in self.results:
            model_path = f"04_PlanningData/{self.store_name}/zone_schedule_{date_suffix}_model.csv"
            fallback_path = f"04_PlanningData/{self.store_name}/zone_schedule_{date_suffix}_fallback.csv"
            try:
                storage.write_csv(
                    self.results["optimization_result_model"],
                    model_path,
                )
                print(
                    f"[OptimizerRunner] Model output saved: zone_schedule_{date_suffix}_model.csv"
                )
            except Exception as error:
                print(f"[OptimizerRunner] Warning: Failed to save model output: {error}")
            try:
                storage.write_csv(
                    self.results["optimization_result_fallback"],
                    fallback_path,
                )
                print(
                    f"[OptimizerRunner] Fallback output saved: zone_schedule_{date_suffix}_fallback.csv"
                )
            except Exception as error:
                print(f"[OptimizerRunner] Warning: Failed to save fallback output: {error}")

        # Also save long format results
        try:
            long_result_df = self.optimizer.get_long_format(self.results["optimization_result"])
            if not long_result_df.empty:
                # Generate long format filename
                start_date_formatted = start_date.replace("-", "")
                end_date_formatted = end_date.replace("-", "")
                long_filename = f"zone_schedule_long_{start_date_formatted}_{end_date_formatted}.csv"
                long_output_logical_path = f"04_PlanningData/{self.store_name}/{long_filename}"

                print(
                    f"[OptimizerRunner] Saving long format optimization results to storage path: {long_output_logical_path}"
                )
                storage.write_csv(long_result_df, long_output_logical_path)
                print(
                    f"[OptimizerRunner] Long format optimization results saved to: {long_output_logical_path}"
                )
        except Exception as error:
            print(
                f"[OptimizerRunner] Warning: Failed to save long format results: {error}"
            )
            # Don't raise - wide format results are already saved

        # Also save unit-level results if available
        if "optimization_result_units" in self.results:
            # Generate unit-level filename
            start_date_formatted = start_date.replace("-", "")
            end_date_formatted = end_date.replace("-", "")
            unit_filename = (
                f"unit_schedule_{start_date_formatted}_{end_date_formatted}.csv"
            )
            unit_output_logical_path = (
                f"04_PlanningData/{self.store_name}/{unit_filename}"
            )

            print(
                f"[OptimizerRunner] Saving unit-level optimization results to storage path: {unit_output_logical_path}"
            )
            try:
                storage.write_csv(
                    self.results["optimization_result_units"], unit_output_logical_path
                )
                print(
                    f"[OptimizerRunner] Unit-level optimization results saved to: {unit_output_logical_path}"
                )
            except Exception as error:
                print(
                    f"[OptimizerRunner] Warning: Failed to save unit-level results to {unit_output_logical_path}: {error}"
                )
                # Don't raise - zone-level results are already saved

        # Also upload to Google Drive (only when running on GCS/production)
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        if backend == "gcs":
            try:
                self._upload_to_google_drive(
                    self.results["optimization_result"], filename
                )
            except Exception as error:
                print(
                    f"[OptimizerRunner] Warning: Failed to upload zone-level schedule to Google Drive: {error}"
                )
                # Don't raise - we still want to return the storage path even if GDrive fails

            # Also upload unit-level schedule to Google Drive if available
            if "optimization_result_units" in self.results:
                try:
                    start_date_formatted = start_date.replace("-", "")
                    end_date_formatted = end_date.replace("-", "")
                    unit_filename = (
                        f"unit_schedule_{start_date_formatted}_{end_date_formatted}.csv"
                    )
                    self._upload_to_google_drive(
                        self.results["optimization_result_units"], unit_filename
                    )
                except Exception as error:
                    print(
                        f"[OptimizerRunner] Warning: Failed to upload unit-level schedule to Google Drive: {error}"
                    )
                    # Don't raise - zone-level upload may have succeeded
        else:
            print(
                f"[OptimizerRunner] Skipping Google Drive upload (running locally with STORAGE_BACKEND={backend})"
            )

        return output_logical_path
        