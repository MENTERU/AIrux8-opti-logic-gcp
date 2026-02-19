"""
Train per-zone power and temp models and save to 03_Models.
Run from project root (services/airux8-optimize):  uv run python scripts/train_zone_models.py [--store Clea]
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.utils import get_data_path
from optimization.zone_model_trainer import train_and_save_all_zones


def main():
    parser = argparse.ArgumentParser(description="Train zone models and save to 03_Models")
    parser.add_argument("--store", default="Clea", help="Store name (default: Clea)")
    parser.add_argument("--min-samples", type=int, default=100, help="Min samples per zone (default: 100)")
    args = parser.parse_args()

    processed_dir = get_data_path("processed_data_path")
    features_path = os.path.join(processed_dir, args.store, f"features_processed_{args.store}.csv")

    if not os.path.isfile(features_path):
        print(f"Features file not found: {features_path}")
        sys.exit(1)

    print(f"Training zone models for store: {args.store}")
    print(f"Features: {features_path}")
    print(f"Min samples per zone: {args.min_samples}")

    saved = train_and_save_all_zones(
        features_csv_path=features_path,
        store=args.store,
        power_model_type="ridge",
        temp_model_type="ridge",
        min_samples=args.min_samples,
    )

    if saved:
        print(f"\nSaved {len(saved)} zone model(s):")
        for p in saved:
            print(f"  - {p}")
    else:
        print("\nNo zone models were saved (insufficient data or training failed).")
        sys.exit(1)


if __name__ == "__main__":
    main()
