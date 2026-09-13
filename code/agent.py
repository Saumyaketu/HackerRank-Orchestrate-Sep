"""
Autonomous AI Financial Decision Agent for Buy or Wait.
Orchestrates Multimodal Perception, Message Intelligence, 90-Day Cash Flow Forecasting,
Neuro-Symbolic Constraint Verification, and Grounded Multi-Criteria Optimization.
"""

import os
from pathlib import Path
import pandas as pd

from code.vlm_agent import MultimodalVisionAgent, vlm_agent
from code.message_agent import FinancialMessageAgent, message_agent
from code.evidence import convert_currency
from code.forecaster import simulate_daily_balances
from code.optimizer import solve_financial_request
from code.explainer import generate_decision_explanation

class BuyOrWaitAgent:
    """
    End-to-End AI Financial Agent implementing:
    1. Multimodal Vision Receipt Perception
    2. Message Intent & Adversarial Injection Defense
    3. Conservative Statistical 90-Day Cash Flow Forecasting
    4. Tool-Augmented Optimization & Preference Hierarchy Ranking
    5. Grounded Transparent Rationale Synthesis
    """
    def __init__(self, dataset_dir: Path, model_name: str = "llava"):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.vlm_agent = MultimodalVisionAgent(model_name=self.model_name)
        self.message_agent = FinancialMessageAgent(model_name=self.model_name)

    def prepare_evidence(self) -> tuple:
        """
        Loads raw records and executes perception layers.
        """
        profiles_df = pd.read_csv(self.dataset_dir / "financial_profiles.csv")
        events_df = pd.read_csv(self.dataset_dir / "financial_events.csv")
        options_df = pd.read_csv(self.dataset_dir / "request_payment_options.csv")
        messages_df = pd.read_csv(self.dataset_dir / "messages.csv")
        images_df = pd.read_csv(self.dataset_dir / "images.csv")
        rates_df = pd.read_csv(self.dataset_dir / "exchange_rates.csv")

        # Multimodal Vision Layer: Resolve missing financial event amounts from media receipts
        media_dir = self.dataset_dir / "media" / "images"
        events_df = self.vlm_agent.resolve_missing_financial_events(events_df, media_dir, images_df)

        # Message Intelligence Layer: Parse communications & detect prompt injections
        messages_by_user = {}
        for _, mrow in messages_df.iterrows():
            u = mrow['user_id']
            msg_text = str(mrow.get('message_text', ''))
            ev_id = mrow.get('related_event_id') if pd.notna(mrow.get('related_event_id')) else None
            parsed = self.message_agent.parse_message(msg_text, u, ev_id)
            parsed['request_id'] = mrow.get('request_id') if pd.notna(mrow.get('request_id')) else None
            parsed['sent_at'] = mrow.get('sent_at') if pd.notna(mrow.get('sent_at')) else ''
            messages_by_user.setdefault(u, []).append(parsed)

        return profiles_df, events_df, options_df, messages_by_user, rates_df

    def evaluate_request(self, req_row: pd.Series, profiles_df: pd.DataFrame, 
                         events_df: pd.DataFrame, options_df: pd.DataFrame, 
                         rates_df: pd.DataFrame, messages_by_user: dict) -> dict:
        """
        Executes neuro-symbolic reasoning for an individual user request:
        1. Formulates candidate payment plans
        2. Invokes 90-day simulation oracle to verify mathematical safety
        3. Explores spending changes if headroom is deficient
        4. Applies 6-level multi-criteria ranking
        5. Synthesizes grounded natural language explanation
        """
        decision = solve_financial_request(
            req_row, events_df, profiles_df, options_df, rates_df, messages_by_user
        )
        explanation = generate_decision_explanation(decision, events_df)
        decision['decision_explanation'] = explanation
        return decision

    def process_all_requests(self, requests_file: str = "requests.csv", verbose: bool = True) -> pd.DataFrame:
        """
        Processes all evaluation requests and compiles predictions.
        """
        profiles_df, events_df, options_df, messages_by_user, rates_df = self.prepare_evidence()
        requests_df = pd.read_csv(self.dataset_dir / requests_file)

        results = []
        total = len(requests_df)
        for idx, req_row in requests_df.iterrows():
            res = self.evaluate_request(
                req_row, profiles_df, events_df, options_df, rates_df, messages_by_user
            )
            results.append({
                "request_id": res["request_id"],
                "amount_safe_to_pay": res["amount_safe_to_pay"],
                "affordability_status": res["affordability_status"],
                "recommended_payment_method": res["recommended_payment_method"],
                "payment_plan": res["payment_plan"],
                "earliest_date_for_full_payment": res["earliest_date_for_full_payment"],
                "spending_changes_needed": res["spending_changes_needed"],
                "decision_explanation": res["decision_explanation"]
            })
            if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total):
                print(f"Agent evaluated {idx + 1}/{total} requests...", flush=True)

        from code.config import OUTPUT_COLUMNS
        return pd.DataFrame(results)[OUTPUT_COLUMNS]
