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
from code.agent import BuyOrWaitAgent

def run_agent(dataset_dir: Path = DATASET_DIR, output_path: Path = None, 
              requests_file: str = "requests.csv", model: str = "llava", verbose: bool = True) -> pd.DataFrame:
    start_time = time.time()
    
    if output_path is None:
        output_path = dataset_dir / "output.csv"

    if verbose:
        print(f"=== Running Buy or Wait Autonomous AI Financial Agent ===", flush=True)
        print(f"Model: {model} (Local VLM/LLM via Ollama)", flush=True)
        print(f"Dataset directory: {dataset_dir}", flush=True)
        print(f"Input file: {requests_file}", flush=True)
        print(f"Output file: {output_path}", flush=True)

    # Instantiate AI Agent with local model
    agent = BuyOrWaitAgent(dataset_dir=dataset_dir, model_name=model)
    output_df = agent.process_all_requests(requests_file=requests_file, verbose=verbose)

    # Save to requested destination
    output_df.to_csv(output_path, index=False)
    
    # Also save to repo root output.csv if processing official requests.csv
    root_output = REPO_ROOT / "output.csv"
    if requests_file == "requests.csv" and root_output.resolve() != output_path.resolve():
        output_df.to_csv(root_output, index=False)

    if verbose:
        total_time = time.time() - start_time
        print(f"Successfully generated predictions for {len(output_df)} requests in {total_time:.2f}s.", flush=True)
        print(f"Saved to {output_path} and {root_output}", flush=True)

    return output_df

def main():
    parser = argparse.ArgumentParser(description="Buy or Wait Financial Agent")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR), help="Path to dataset directory")
    parser.add_argument("--output-path", type=str, default=None, help="Path to write output.csv")
    parser.add_argument("--requests-file", type=str, default="requests.csv", help="Requests CSV filename")
    parser.add_argument("--model", type=str, default="llava", help="Local VLM/LLM model name (default: llava)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    d_dir = Path(args.dataset_dir)
    o_path = Path(args.output_path) if args.output_path else None
    
    run_agent(dataset_dir=d_dir, output_path=o_path, requests_file=args.requests_file, model=args.model, verbose=not args.quiet)

if __name__ == "__main__":
    main()
