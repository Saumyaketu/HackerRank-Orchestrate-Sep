"""
Main entry point for the Buy or Wait AI financial decision agent.
Reads inputs from dataset/, runs 90-day simulation and plan optimization,
and writes predictions to output.csv matching the exact required schema.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import pandas as pd

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.config import DATASET_DIR, OUTPUT_COLUMNS
from code.evidence import resolve_image_events, parse_message_evidence
from code.optimizer import solve_financial_request
from code.explainer import generate_decision_explanation

def run_agent(dataset_dir: Path = DATASET_DIR, output_path: Path = None, 
              requests_file: str = "requests.csv", verbose: bool = True) -> pd.DataFrame:
    start_time = time.time()
    
    if output_path is None:
        output_path = dataset_dir / "output.csv"

    if verbose:
        print(f"=== Running Buy or Wait Financial Agent ===")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Input file: {requests_file}")
        print(f"Output file: {output_path}")

    # 1. Load input datasets
    requests_df = pd.read_csv(dataset_dir / requests_file)
    profiles_df = pd.read_csv(dataset_dir / "financial_profiles.csv")
    events_df = pd.read_csv(dataset_dir / "financial_events.csv")
    options_df = pd.read_csv(dataset_dir / "request_payment_options.csv")
    messages_df = pd.read_csv(dataset_dir / "messages.csv")
    rates_df = pd.read_csv(dataset_dir / "exchange_rates.csv")

    # 2. Resolve image receipts
    events_df = resolve_image_events(events_df)

    # 3. Parse messages evidence
    messages_by_user = {}
    for _, mrow in messages_df.iterrows():
        u = mrow['user_id']
        messages_by_user[u] = parse_message_evidence(mrow)

    # 4. Solve each request
    results = []
    total_reqs = len(requests_df)
    
    for idx, req_row in requests_df.iterrows():
        res = solve_financial_request(
            req_row, events_df, profiles_df, options_df, rates_df, messages_by_user
        )
        explanation = generate_decision_explanation(res, events_df)
        
        results.append({
            "request_id": res["request_id"],
            "amount_safe_to_pay": res["amount_safe_to_pay"],
            "affordability_status": res["affordability_status"],
            "recommended_payment_method": res["recommended_payment_method"],
            "payment_plan": res["payment_plan"],
            "earliest_date_for_full_payment": res["earliest_date_for_full_payment"],
            "spending_changes_needed": res["spending_changes_needed"],
            "decision_explanation": explanation
        })

        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total_reqs):
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{total_reqs} requests ({elapsed:.2f}s elapsed)...")

    # 5. Build output DataFrame with exact schema and order
    output_df = pd.DataFrame(results)[OUTPUT_COLUMNS]

    # Save to requested destination
    output_df.to_csv(output_path, index=False)
    
    # Also save to repo root output.csv for convenience
    root_output = REPO_ROOT / "output.csv"
    if root_output.resolve() != output_path.resolve():
        output_df.to_csv(root_output, index=False)

    if verbose:
        total_time = time.time() - start_time
        print(f"Successfully generated predictions for {total_reqs} requests in {total_time:.2f}s.")
        print(f"Saved to {output_path} and {root_output}")

    return output_df

def main():
    parser = argparse.ArgumentParser(description="Buy or Wait Financial Agent")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR), help="Path to dataset directory")
    parser.add_argument("--output-path", type=str, default=None, help="Path to write output.csv")
    parser.add_argument("--requests-file", type=str, default="requests.csv", help="Requests CSV filename")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    d_dir = Path(args.dataset_dir)
    o_path = Path(args.output_path) if args.output_path else None
    
    run_agent(dataset_dir=d_dir, output_path=o_path, requests_file=args.requests_file, verbose=not args.quiet)

if __name__ == "__main__":
    main()
