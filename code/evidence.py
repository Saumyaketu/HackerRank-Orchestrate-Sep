"""
Evidence extraction and multimodal resolution for Buy or Wait.
Integrates image receipts, parses employer/bank/service messages, and performs currency conversions.
"""

import re
from pathlib import Path
import pandas as pd
from code.config import MEDIA_DIR
from code.vlm_agent import vlm_agent
from code.message_agent import message_agent

def resolve_image_events(events_df: pd.DataFrame, images_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Resolves missing amounts in financial_events.csv using the Multimodal Vision Agent.
    """
    return vlm_agent.resolve_missing_financial_events(events_df, MEDIA_DIR)

def parse_message_evidence(row: pd.Series) -> dict:
    """
    Extracts structured financial facts using the Financial Message NLP Agent.
    """
    text = str(row.get("message_text", ""))
    user_id = row.get("user_id")
    event_id = row.get("related_event_id") if pd.notna(row.get("related_event_id")) else None
    
    return message_agent.parse_message(text, user_id, event_id)


def convert_currency(amount: float, from_curr: str, to_curr: str, date_str: str, rates_df: pd.DataFrame) -> float:
    """
    Converts foreign currency amount to target currency using dated exchange rates.
    """
    if from_curr == to_curr or not amount or pd.isna(amount):
        return amount

    # Exact date match
    sub = rates_df[(rates_df["from_currency"] == from_curr) & 
                   (rates_df["to_currency"] == to_curr) & 
                   (rates_df["rate_date"] == date_str)]
    if not sub.empty:
        return amount * float(sub.iloc[0]["rate"])

    raise ValueError(
        f"Missing exchange rate for {from_curr}->{to_curr} on {date_str}"
    )
