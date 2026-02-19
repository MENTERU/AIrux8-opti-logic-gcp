# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

# (e.g. services/svc-airux8-optimize-data-scraping on GCP Cloud Run)
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import pytz
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config.utils import get_data_path
from optimization.optimizer_runner import OptimizerRunner
from processing.aggregator import aggregation_runner
from processing.preprocessor import preprocessing_runner
from processing.utilities.master_data_loader import master_data_loader_runner
from service.secretmanager import SecretManagerClient

app = FastAPI()  # Initialize FastAPI app


def _resolve_weather_api_key() -> str:
    """Resolve weather API key by backend"""
    # 1) Environment variable
    env_key = os.getenv("WEATHER_API_KEY")
    if env_key:
        return env_key

    # 2) GCP Secret Manager (when running on GCP)
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    try:
        if backend == "gcs":
            sm = SecretManagerClient()
            key = sm.get_secret_as_str("WEATHER_API_KEY")
            if key:
                return key
    except Exception:
        pass

    # 3) Optional local fallback without hard import
    try:
        import importlib

        pi = importlib.import_module("config.private_information")
        local_key = getattr(pi, "WEATHER_API_KEY", None)
        if local_key:
            return local_key
    except Exception:
        pass

    raise RuntimeError("WEATHER_API_KEY not configured. Set env var or secret.")


@app.post("/execute_optimization_pipeline")
def execute_optimization_pipeline():
    """This endpoint is used to execute the optimization pipeline.
    It will be triggered by GCS event when preprocessed data is uploaded. 
    Or by Cloud Scheduler to run daily.

    Args:
        event: dict - GCS event
    Returns: 
        JSONResponse: JSON response
    """
    # TODO: Implement this check later. For now, we will use Cloud Scheduler to trigger
    # bucket_name = event.get("bucket")
    # file_name = event.get("name")

    # # Guard: must have bucket and name
    # if not bucket_name or not file_name:
    #     raise HTTPException(
    #         status_code=400, detail="Invalid GCS event: missing bucket or name"
    #     )

    # # Guard: only files under the allowed prefixes (avoid recursion on our own outputs)
    # if not any(file_name.startswith(prefix) for prefix in ALLOWED_TRIGGER_PREFIXES):
    #     return JSONResponse(
    #         status_code=201,
    #         content={
    #             "message": f"File {file_name} skipped - not in allowed prefixes",
    #             "allowed_prefixes": ALLOWED_TRIGGER_PREFIXES,
    #         },
    #     )

    # # Guard: only allow csv files
    # allowed_extensions = {"csv"}
    # file_extension = file_name.split(".")[-1].lower()
    # if file_extension not in allowed_extensions:
    #     return JSONResponse(
    #         status_code=201,
    #         content={
    #             "message": f"File {file_name} skipped - unsupported file type: {file_extension}",
    #         },
    #     )

    success = run_optimization_for_store(
        store_name="Clea",
        execution_mode="gcs",
    )
    status_code = 200 if success else 201
    return JSONResponse(
        status_code=status_code,
        content={
            "message": "full pipeline executed" if success else "pipeline failed",
            # "bucket": bucket_name,
            # "name": file_name,
        },
    )


def parse_arguments():
    """Parse command line arguments for local development"""
    parser = argparse.ArgumentParser(
        description="エアコン最適化システム - 実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        実行例:
        uv run run_optimization.py --preprocess-only
        uv run run_optimization.py --aggregate-only
                """,
    )

    # Store selection
    parser.add_argument(
        "--store", type=str, default="Clea", help="対象ストア(default:Clea)"
    )
    parser.add_argument("--preprocess", action="store_true", help="前処理のみ実行")
    parser.add_argument("--aggregate", action="store_true", help="集約のみ実行")
    parser.add_argument("--optimize", action="store_true", help="最適化のみ実行")
    parser.add_argument(
        "--strategy",
        type=str,
        default="hourly",
        choices=["hourly", "similar_day"],
        help="最適化戦略の選択 (hourly|similar_day)",
    )
    parser.add_argument("--start-date", type=str, help="最適化開始日 (YYYY-MM-DD形式)")
    parser.add_argument("--end-date", type=str, help="最適化終了日 (YYYY-MM-DD形式)")

    return parser.parse_args()


def run_optimization_for_store(
    store_name,
    execution_mode: str = "full",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy: str = "hourly",
):
    """
    指定されたストアの最適化を実行 (Cloud Run or Local Development )

    Args:
        store_name: ストア名
        execution_mode: 実行モード ("preprocess", "aggregate", "optimize", "gcs", "full")
        start_date: 開始日 (オプション)
        end_date: 終了日 (オプション)

    Returns:
        bool: 実行成功時True、失敗時False
    """
    print(f"🚀 {store_name}の最適化パイプライン開始 (モード: {execution_mode})")
    store_master_file = master_data_loader_runner(store_name)
    if store_master_file is None:
        print(f"❌ エラーが発生しました: ストアマスタデータの読み込みに失敗しました")
        return False

    try:
        if execution_mode == "preprocess":
            print("📊 前処理のみ実行")
            preprocessing_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                weather_api_key=_resolve_weather_api_key(),
                temperature_std_multiplier=5.0,
                power_std_multiplier=5.0,
                export_temp_range_stats=False,
            )
            print("✅ 前処理完了")
            return True

        elif execution_mode == "aggregate":
            logging.info("=" * 50)
            print("🔄 集約のみ実行")
            aggregation_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                freq="1H",
            )
            print("✅ 集約完了")
            return True
        elif execution_mode == "optimize":
            print("=" * 50)
            print("🔄 最適化のみ実行")

            # Use provided dates or default to today -> +3 days
            if start_date is None:
                start_date = datetime.now(pytz.timezone("Asia/Tokyo")).strftime(
                    "%Y-%m-%d"
                )
                # start_date = "2024-12-01" # for sepcified date for test
            if end_date is None:
                end_date = (
                    datetime.now(pytz.timezone("Asia/Tokyo")) + timedelta(days=3)
                ).strftime("%Y-%m-%d")
                # end_date = "2024-12-03" # for sepcified date for test
            logging.info(f"Optimizing for period: {start_date} to {end_date}")
            logging.info(f"Strategy: {strategy}")

            try:
                # Initialize optimization runner
                logging.info("Initializing Optimization Runner...")
                runner = OptimizerRunner(store_name=store_name)

                # Run optimization (this will load data and run optimization)
                logging.info(f"Running optimization for {start_date} to {end_date}...")
                results = runner.run_optimization(
                    start_date, end_date, strategy=strategy
                )

                if results.get("status") == "success":
                    # Save results (main CSV + zone_schedule_*_model.csv and zone_schedule_*_fallback.csv for comparison)
                    output_path = runner.save_results_to_csv(start_date, end_date)
                    print(f"✅ Optimization completed successfully!")
                    print(f"📁 Results saved to: {output_path}")
                    print(f"📁 Comparison outputs: zone_schedule_<dates>_model.csv and zone_schedule_<dates>_fallback.csv (same folder)")
                    return True
                else:
                    print(
                        f"❌ Optimization failed: {results.get('error', 'Unknown error')}"
                    )
                    return False

            except Exception as e:
                print(f"❌ Optimization failed: {e}")
                logging.error(f"Optimization error: {e}", exc_info=True)
                return False

        elif execution_mode == "gcs":
            print("🔄 フルパイプライン実行")

            opt_start_date = datetime.now(pytz.timezone("Asia/Tokyo")).strftime(
                "%Y-%m-%d"
            )
            opt_end_date = (
                datetime.now(pytz.timezone("Asia/Tokyo")) + timedelta(days=3)
            ).strftime("%Y-%m-%d")

            runner = OptimizerRunner(store_name=store_name)
            optimization_results = runner.run_optimization(opt_start_date, opt_end_date)
            if optimization_results.get("status") != "success":
                print(
                    f"❌ フルパイプライン失敗: {optimization_results.get('error', 'Unknown error')}"
                )
                return False
            # Save results to storage (GCS or local depending on backend)
            try:
                output_path = runner.save_results_to_csv(opt_start_date, opt_end_date)
                print("✅ フルパイプライン完了")
                print(f"📁 結果保存先: {output_path}")
                return True
            except Exception as e:
                print(f"❌ 結果の保存に失敗しました: {e}")
                return False

        else:  # full
            print("🔄 フルパイプライン実行")
            opt_start_date = start_date or end_date
            opt_end_date = end_date or start_date

            if opt_start_date is None and opt_end_date is None:
                now_tokyo = datetime.now(pytz.timezone("Asia/Tokyo"))
                opt_start_date = now_tokyo.strftime("%Y-%m-%d")
                opt_end_date = (now_tokyo + timedelta(days=3)).strftime("%Y-%m-%d")
            elif opt_start_date is None:
                opt_start_date = opt_end_date
            elif opt_end_date is None:
                opt_end_date = opt_start_date

            preprocess_success = preprocessing_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                weather_api_key=_resolve_weather_api_key(),
                temperature_std_multiplier=5.0,
                power_std_multiplier=5.0,
                export_temp_range_stats=False,
            )
            if not preprocess_success:
                print("❌ フルパイプライン失敗: 前処理に失敗しました")
                return False

            aggregated_df = aggregation_runner(
                store_name=store_name,
                store_master_file=store_master_file,
                freq="1H",
            )
            if aggregated_df is None or aggregated_df.empty:
                print("❌ フルパイプライン失敗: 集約結果が空です")
                return False

            runner = OptimizerRunner(store_name=store_name)
            optimization_results = runner.run_optimization(
                opt_start_date, opt_end_date, strategy=strategy
            )
            if optimization_results.get("status") != "success":
                print(
                    f"❌ フルパイプライン失敗: {optimization_results.get('error', 'Unknown error')}"
                )
                return False

            output_path = runner.save_results_to_csv(opt_start_date, opt_end_date)
            print("✅ フルパイプライン完了")
            print(f"📁 結果保存先: {output_path}")
            return True

    except Exception as error:
        print(f"❌ エラーが発生しました: {error}")
        return False


def main():
    """メイン実行関数"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # コマンドライン引数の解析
    args = parse_arguments()

    # 実行モードの決定 (複数のフラグを組み合わせ可能)
    execution_modes = []
    if args.preprocess:
        execution_modes.append("preprocess")
    if args.aggregate:
        execution_modes.append("aggregate")
    if args.optimize:
        execution_modes.append("optimize")

    # store_name
    if args.store is None:
        store_name = "Clea"  # default store name for development
    else:
        store_name = args.store

    if not execution_modes:
        execution_mode = "full"
        success = run_optimization_for_store(
            store_name=store_name,
            execution_mode=execution_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
        )
    elif len(execution_modes) == 1:
        execution_mode = execution_modes[0]
        success = run_optimization_for_store(
            store_name=store_name,
            execution_mode=execution_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
        )
    else:
        # Multiple modes specified - execute them in sequence
        print(f"🔄 Multiple execution modes specified: {execution_modes}")
        print(f"🔄 Will execute them in sequence")

        success = True
        # Execute each mode in sequence
        for i, mode in enumerate(execution_modes):
            print(f"\n{'='*70}")
            print(f"🏢 {store_name} - Step {i+1}/{len(execution_modes)}: {mode}")
            print(f"{'='*70}")

            step_success = run_optimization_for_store(
                store_name=store_name,
                execution_mode=mode,
                start_date=args.start_date,
                end_date=args.end_date,
                strategy=args.strategy,
            )

            if not step_success:
                print(f"❌ Step {i+1} ({mode}) failed. Stopping execution.")
                success = False
                break

    # 終了コードの設定
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
