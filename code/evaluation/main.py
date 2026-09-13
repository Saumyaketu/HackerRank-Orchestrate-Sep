"""
Automated evaluation workflow for Buy or Wait.
Runs evaluation against sample_requests.csv and computes accuracy metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.config import DATASET_DIR
from code.main import run_agent

def evaluate_on_sample_requests(dataset_dir: Path = DATASET_DIR):
    print("=== Running Evaluation on sample_requests.csv ===")
    
    # 1. Run agent on sample_requests.csv
    sample_path = dataset_dir / "sample_requests.csv"
    ground_truth_df = pd.read_csv(sample_path)
    
    pred_df = run_agent(
        dataset_dir=dataset_dir,
        requests_file="sample_requests.csv",
        output_path=REPO_ROOT / "evaluation_sample_pred.csv",
        verbose=False
    )

    n_samples = len(ground_truth_df)
    print(f"Evaluated {n_samples} benchmark sample requests.\n")

    # 2. Compute metrics
    # A. Affordability Status Accuracy
    status_matches = (pred_df['affordability_status'] == ground_truth_df['affordability_status'])
    status_acc = status_matches.mean() * 100

    # B. Recommended Payment Method Accuracy
    method_matches = (pred_df['recommended_payment_method'] == ground_truth_df['recommended_payment_method'])
    method_acc = method_matches.mean() * 100

    # C. Payment Plan Match
    # For plans with floating numbers, compare structure or string
    plan_matches = (pred_df['payment_plan'] == ground_truth_df['payment_plan'])
    plan_acc = plan_matches.mean() * 100

    # D. Earliest Date for Full Payment Match
    gt_earliest = ground_truth_df['earliest_date_for_full_payment'].fillna('').astype(str)
    pred_earliest = pred_df['earliest_date_for_full_payment'].fillna('').astype(str)
    earliest_matches = (gt_earliest == pred_earliest)
    earliest_acc = earliest_matches.mean() * 100

    # E. Spending Changes Needed Match
    gt_changes = ground_truth_df['spending_changes_needed'].fillna('none')
    pred_changes = pred_df['spending_changes_needed'].fillna('none')
    changes_matches = (gt_changes == pred_changes)
    changes_acc = changes_matches.mean() * 100

    # F. Safe Amount Correlation and MAE
    gt_safe = ground_truth_df['amount_safe_to_pay'].astype(float)
    pred_safe = pred_df['amount_safe_to_pay'].astype(float)
    mae_safe = np.abs(gt_safe - pred_safe).mean()
    corr_safe = np.corrcoef(gt_safe, pred_safe)[0, 1]

    # Print summary table
    print("================================================================")
    print("                EVALUATION BENCHMARK RESULTS                    ")
    print("================================================================")
    print(f"Affordability Status Accuracy:       {status_acc:.1f}% ({status_matches.sum()}/{n_samples})")
    print(f"Recommended Payment Method Accuracy: {method_acc:.1f}% ({method_matches.sum()}/{n_samples})")
    print(f"Payment Plan Accuracy:               {plan_acc:.1f}% ({plan_matches.sum()}/{n_samples})")
    print(f"Earliest Full Payment Date Accuracy: {earliest_acc:.1f}% ({earliest_matches.sum()}/{n_samples})")
    print(f"Spending Changes Needed Accuracy:    {changes_acc:.1f}% ({changes_matches.sum()}/{n_samples})")
    print(f"Safe Amount Pearson Correlation:     {corr_safe:.4f}")
    print(f"Safe Amount Mean Absolute Error:     {mae_safe:.2f}")
    print("================================================================\n")

    # Detailed comparison
    print("Detailed Request Breakdown:")
    for i in range(n_samples):
        req_id = ground_truth_df.iloc[i]['request_id']
        s_ok = status_matches.iloc[i]
        m_ok = method_matches.iloc[i]
        p_ok = plan_matches.iloc[i]
        e_ok = earliest_matches.iloc[i]
        print(f"[{req_id}] Status: {pred_df.iloc[i]['affordability_status']:<22} | Method: {pred_df.iloc[i]['recommended_payment_method']:<18} | Status OK: {s_ok!s:<5} | Method OK: {m_ok!s:<5}")

    return {
        "status_acc": status_acc,
        "method_acc": method_acc,
        "plan_acc": plan_acc,
        "earliest_acc": earliest_acc,
        "changes_acc": changes_acc,
        "corr_safe": corr_safe,
        "mae_safe": mae_safe
    }

if __name__ == "__main__":
    evaluate_on_sample_requests()
