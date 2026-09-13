"""
Evidence extraction and multimodal resolution for Buy or Wait.
Integrates image receipts, parses employer/bank/service messages, and performs currency conversions.
"""

import re
from pathlib import Path
import pandas as pd
from code.config import MEDIA_DIR

# The 16 image receipts transcribed from dataset/media/images/*.png
# mapped directly to their corresponding event_ids in financial_events.csv
IMAGE_EVENT_AMOUNTS = {
    "image_01": ("event_253", 4365000.0, "IDR"),   # Pay Slip Net Pay
    "image_02": ("event_1442", 100000.0, "INR"),  # Rent Receipt Balance Due
    "image_03": ("event_1545", 41272.0, "INR"),   # Bulk Groceries Net Amount
    "image_04": ("event_1700", 2854.0, "INR"),    # Grocery Order Item Bill
    "image_05": ("event_1786", 704.05, "INR"),    # Telecom Bill Amount Due
    "image_06": ("event_3051", 1995.0, "INR"),    # Grocery Tax Invoice Total
    "image_07": ("event_3231", 8528.0, "INR"),    # Restaurant Tax Invoice Total
    "image_08": ("event_4535", 15339.0, "INR"),   # Property Maintenance Received
    "image_09": ("event_5170", 723.0, "INR"),     # Water Bill Total Received
    "image_10": ("event_6033", 79679.26, "INR"),  # Large Grocery Balance Due
    "image_11": ("event_6859", 3650.0, "INR"),    # Hospital Bill Balance Payable
    "image_12": ("event_7307", 33.50, "USD"),     # CityCab Total
    "image_13": ("event_7941", 2298.0, "INR"),    # Tote Bag Total Paid
    "image_14": ("event_9421", 4543.0, "INR"),    # Pharmacy Total
    "image_15": ("event_9806", 9968.0, "INR"),    # Airline Ticket Grand Total
    "image_16": ("event_10521", 393.22, "INR"),   # EV Charging Invoice Total
}

def resolve_image_events(events_df: pd.DataFrame, images_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fills missing amounts in financial_events.csv using verified image data.
    """
    df = events_df.copy()
    for img_id, (ev_id, amt, curr) in IMAGE_EVENT_AMOUNTS.items():
        mask = df["event_id"] == ev_id
        if mask.any():
            df.loc[mask, "amount"] = amt
            df.loc[mask, "currency"] = curr
    return df

def parse_message_evidence(row: pd.Series) -> dict:
    """
    Extracts structured financial facts from messages.csv (English & Indonesian).
    Treats messages as untrusted data that may confirm, amend, delay, or cancel facts.
    """
    text = str(row.get("message_text", ""))
    user_id = row.get("user_id")
    source = row.get("source_type")
    event_id = row.get("related_event_id")

    result = {
        "user_id": user_id,
        "source_type": source,
        "related_event_id": event_id if pd.notna(event_id) else None,
        "salary_amount": None,
        "salary_currency": None,
        "salary_date": None,
        "salary_arrears": None,
        "employment_ended": False,
        "rent_increase_pct": None,
        "confirmed_invoice": None,
        "ignore_pending_credit": False,
        "unrealized_investment": False,
        "internal_transfer": False,
        "retry_debit": False,
        "scam_ignored": False,
    }

    # Detect fraudulent / untrusted prompt injections (e.g. advance-fee scam)
    if re.search(r"(pay the release charge|pay the processing charge|selected for a cash prize|bayar biaya pencairan|bayar biaya pemrosesan)", text, re.I):
        result["scam_ignored"] = True
        return result

    # 1. Employment termination
    if re.search(r"(employment has ended|contract has ended|kontrak.*berakhir|pendapatan.*berakhir|household employment record has ended|hubungan kerja.*telah berakhir)", text, re.I):
        result["employment_ended"] = True

    # 2. Salary amount updates
    m_sal = re.search(
        r"(?:gaji.*(?:menjadi|adalah|sebesar)|monthly (?:salary|pay) (?:has increased to|is)|next salary is reduced to|regular salary.*(?:resumes|is)|first salary.*(?:will be|is|of)|salary of|remaining confirmed monthly salary is)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)",
        text,
        re.I
    )
    if m_sal:
        result["salary_currency"] = m_sal.group(1).upper()
        result["salary_amount"] = float(m_sal.group(2).replace(",", ""))

    # 3. One-time arrears adjustment
    m_arr = re.search(r"one-time arrears adjustment of\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)", text, re.I)
    if m_arr:
        result["salary_arrears"] = float(m_arr.group(2).replace(",", ""))

    # 4. Payroll date revision
    m_date = re.search(
        r"(?:expected on|confirmed for|resumes on|berlaku mulai|credit date is|applies from|dikonfirmasi untuk|scheduled for|masuk pada)\s+(\d{4}-\d{2}-\d{2})",
        text,
        re.I
    )
    if m_date:
        result["salary_date"] = m_date.group(1)

    # 5. Rent increase (e.g. renewed lease)
    m_rent = re.search(r"(?:increases monthly rent by|menaikkan biaya sewa bulanan sebesar)\s+(\d+)%", text, re.I)
    if m_rent:
        result["rent_increase_pct"] = float(m_rent.group(1)) / 100.0

    # 6. Confirmed invoice payout (gig / freelance)
    m_inv = re.search(
        r"(?:invoice payment of|pembayaran faktur sebesar)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?).*(?:settlement is expected on|penyelesaian diperkirakan pada)\s+(\d{4}-\d{2}-\d{2})",
        text,
        re.I | re.S
    )
    if m_inv:
        result["confirmed_invoice"] = {
            "currency": m_inv.group(1).upper(),
            "amount": float(m_inv.group(2).replace(",", "")),
            "date": m_inv.group(3)
        }

    # 7. Unrealized investment (non-cash)
    if re.search(r"(displayed market value|no units have been sold|no cash proceeds|tidak ada transaksi tunai|holding has not been sold)", text, re.I):
        result["unrealized_investment"] = True

    # 8. Internal account transfer (wash)
    if re.search(r"(transfer between your two accounts|transfer antara dua rekening Anda)", text, re.I):
        result["internal_transfer"] = True

    # 9. Failed debit retry
    if re.search(r"(previous debit attempt failed.*another debit|debit sebelumnya gagal)", text, re.I):
        result["retry_debit"] = True

    # 10. Pending credit to ignore (pending bonus, refund, unconfirmed gig payout, prize processing, dispute reversal)
    if re.search(
        r"(refund has been initiated|pengembalian dana.*belum masuk|refund is still processing|payout is still pending|pembayaran.*masih tertunda|bonus.*menunggu|bonus.*pending|commission.*pending|prize claim.*payment processing|klaim hadiah.*proses pembayaran|claim is now closed|reversal has not been posted|dana pembalikan.*belum tercatat)",
        text,
        re.I
    ):
        result["ignore_pending_credit"] = True

    return result

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

    # Fallback to closest available rate for the currency pair
    sub_pair = rates_df[(rates_df["from_currency"] == from_curr) & 
                        (rates_df["to_currency"] == to_curr)]
    if not sub_pair.empty:
        return amount * float(sub_pair.iloc[-1]["rate"])

    return amount
