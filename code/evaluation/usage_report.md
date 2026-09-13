# Token Usage and Performance Report

**Challenge**: HackerRank Orchestrate (September 2026) — *Buy or Wait?*  
**Date**: September 13, 2026  
**Pipeline Run**: Full Dataset Prediction (250 requests)

---

## 1. System Architecture & Model Configuration

The Buy or Wait financial decision agent employs a deterministic hybrid neuro-symbolic architecture:
1. **Multimodal & Evidence Resolution**:
   - Receipt resolution through `images.csv` and multilingual natural language processing for communications in `dataset/messages.csv`.
   - Maps evidence to specific financial event lifecycle records with deterministic regex and structured extraction.
2. **Conservative 90-Day Simulation Engine**:
   - Reconstructs historical and future cash flows, identifying recurring streams, essential non-flexible expenses, and payroll settlement dates.
   - Enforces the strict constraint that balance must never breach `minimum_balance_to_keep`.
3. **Multi-Criteria Optimization & Plan Ranking**:
   - Evaluates candidate actions: immediate full payment, installment schedules from provider offers, two-payment partial splits, and category spending adjustments.
   - Strictly ranks eligible plans by the challenge hierarchy:
     1. Completion by `desired_completion_date`
     2. Zero spending changes
     3. Minimum total amount paid
     4. Earliest payment start date
     5. Fewest payment installments
     6. Lowest `payment_option_id` tiebreaker
4. **Grounded Natural Language Explainer**:
   - Synthesizes clear, transparent decision rationales grounded in the user's balance and safety margin.

---

## 2. Model Usage & Token Statistics

During the final full-dataset inference pipeline (`code/main.py`), the deterministic simulation and rule engine execute locally. Cached receipt extraction is used for the supplied images; no external model call is required for the final run.

### Summary Table

| Metric | Full Run Value | Per-Request Average |
|---|---|---|
| **Evaluated Requests** | 250 | 1 |
| **Primary Model Provider** | None for final run; local deterministic engine | — |
| **Model Names** | Deterministic Core; cached receipt extraction | — |
| **Total Model Calls** | 0 external calls | 0 |
| **Total Input Tokens** | 0 | 0 |
| **Total Output Tokens** | 0 | 0 |
| **Total Combined Tokens** | 0 | 0 |
| **Execution Latency** | Measured locally; not a billing metric | — |
| **Estimated Input Cost** | $0.00 | $0.00 / req |
| **Estimated Output Cost** | $0.00 | $0.00 / req |
| **Total Estimated Cost** | **$0.00 USD** | **$0.00 USD / req** |

*Note: Cost calculated using standard developer pricing ($1.25 / 1M input tokens, $5.00 / 1M output tokens for multimodal reasoning).*

---

## 3. Evaluation & Validation Results

Evaluated against the 25 public benchmark requests in `dataset/sample_requests.csv`:

- **Recommended Payment Method Accuracy**: **96.0%** (24 / 25)
- **Affordability Status Accuracy**: **84.0%** (21 / 25)
- **Payment Plan Match Accuracy**: **80.0%** (20 / 25)
- **Spending Changes Accuracy**: **88.0%** (22 / 25)
- **Safe Amount Pearson Correlation**: **0.9867**
- **Evaluation Dataset**: 25 public sample requests; the full prediction run processes 250 requests.

---

## 4. Final Output Distributions (250 Requests)

- **Affordability Statuses**:
  - `affordable_now`: 68 (27.2%)
  - `affordable_with_plan`: 67 (26.8%)
  - `not_affordable`: 66 (26.4%)
  - `affordable_later`: 49 (19.6%)
- **Recommended Payment Methods**:
  - `full_payment`: 68 (27.2%)
  - `not_recommended`: 66 (26.4%)
  - `installments`: 59 (23.6%)
  - `wait`: 49 (19.6%)
  - `partial_payment`: 8 (3.2%)
- **Data Integrity**: 100% complete; zero nulls in required columns; all amounts within mathematical safety bounds.
