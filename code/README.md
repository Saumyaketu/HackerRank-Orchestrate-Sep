# Buy or Wait? — Autonomous AI Financial Decision Agent

Starter codebase and submission package for HackerRank Orchestrate (September 2026).

## Overview
An Autonomous Tool-Augmented Neuro-Symbolic AI Financial Agent that:
- Performs multimodal vision reasoning on receipt/invoice images using local VLMs (LLaVA) or API vision endpoints.
- Analyzes multilingual natural language communications and defends against adversarial prompt injections (advance-fee scams).
- Runs conservative 90-day daily balance simulations enforcing the critical invariant: balance >= minimum_balance_to_keep.
- Formulates, optimizes, and ranks candidate payment strategies (immediate full payment, provider installment schedules, 2-part partial splits, flexible spending changes, or waiting until a confirmed payday) using the challenge's 6-tier preference hierarchy.
- Synthesizes transparent, grounded natural language decision explanations.

## Setup Instructions
Requirements: Python 3.10+ with pandas, 
umpy, pillow.

`ash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy pillow
`

### Local Open-Source Model (LLaVA / Ollama)
To run the multimodal vision and language understanding locally using Ollama:

`ash
# Ensure Ollama is running and model is downloaded
ollama run llava
`

The system automatically connects to http://localhost:11434 to query the local llava instance. If Ollama is offline, the agent gracefully falls back to resilient visual extractions so it never crashes.

## Running the Agent
To run full inference on dataset/requests.csv and generate output.csv:

`ash
python code/main.py --model llava
`

Optional arguments:
- --model <name>: Model identifier (default: llava)
- --dataset-dir <path>: Path to dataset folder (default: dataset/)
- --output-path <path>: Destination path for output.csv (default: dataset/output.csv)
- --requests-file <filename>: Input requests filename (default: equests.csv)

## Running the Evaluation Benchmark
To run evaluation against dataset/sample_requests.csv:

`ash
python code/evaluation/main.py
`

## Package Contents
- code/agent.py: Autonomous BuyOrWaitAgent orchestrator
- code/vlm_agent.py: Multimodal Vision-Language receipt perception agent (self.model_name = "llava")
- code/message_agent.py: Multilingual communications NLP agent (self.model_name = "llava")
- code/forecaster.py: 90-day daily balance simulation and statistical recurrence detection
- code/optimizer.py: Candidate plan generator and 6-level multi-criteria ranker
- code/explainer.py: Grounded natural language explanations
- code/prompts/: Structured prompt templates for VLM, NLP, reasoning, and explanation
- code/evaluation/main.py: Evaluation and accuracy metric benchmark
- code/evaluation/usage_report.md: Model usage, token statistics, and performance report
