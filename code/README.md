# Buy or Wait? — AI Financial Decision Agent

Starter codebase and submission package for HackerRank Orchestrate (September 2026).

## Overview
An AI-powered financial agent that reconstructs 90-day daily cash flows, incorporates multimodal receipts and messaging evidence, evaluates user preferences and recurring commitments, and determines whether an expense is affordable now, with a plan, later, or not at all.

## Setup Instructions
Requirements: Python 3.10+ with pandas, numpy, pillow.

`ash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy pillow
`

## Running the Agent
To run full inference on dataset/requests.csv and generate output.csv:

`ash
python code/main.py
`

Optional arguments:
- --dataset-dir <path>: Path to dataset folder (default: dataset/)
- --output-path <path>: Destination path for output.csv (default: dataset/output.csv)
- --requests-file <filename>: Input requests filename (default: 
equests.csv)

## Running the Evaluation Benchmark
To run evaluation against dataset/sample_requests.csv:

`ash
python code/evaluation/main.py
`

## Package Contents
- code/main.py: Pipeline entry point
- code/config.py: Schemas, valid categories, constants
- code/evidence.py: Multimodal receipt resolver and message extraction
- code/forecaster.py: 90-day daily balance simulation and recurrence detection
- code/optimizer.py: Candidate plan generator and multi-criteria ranking
- code/explainer.py: Grounded natural language explanations
- code/evaluation/main.py: Evaluation and accuracy metric computation
- code/evaluation/usage_report.md: Model usage, token statistics, and performance report
