import os
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"

# Required output columns in exact order
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation"
]

# Allowed values
AFFORDABILITY_STATUSES = [
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable"
]

RECOMMENDED_PAYMENT_METHODS = [
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended"
]

# Supported currencies
CURRENCIES = ["INR", "ZAR", "IDR", "USD", "EUR"]

# 90-day forecast horizon
FORECAST_DAYS = 90
