"""
Concise, grounded financial decision explanation builder for Buy or Wait.
Produces transparent justifications matching evaluation benchmarks.
"""

from datetime import datetime
import pandas as pd

def format_date_natural(date_str: str) -> str:
    """Formats '2025-08-08' as '8 August 2025'"""
    if not date_str or date_str == 'none':
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.day} {dt.strftime('%B')} {dt.year}"
    except Exception:
        return date_str

def format_amount(amt: float, curr: str) -> str:
    """Formats 15952906.67 as 'IDR 15,952,906.67' or 25256 as 'ZAR 25,256'"""
    if amt == int(amt):
        return f"{curr} {int(amt):,}"
    return f"{curr} {amt:,.2f}"

def format_amount_clean(amt: float) -> str:
    if amt == int(amt):
        return f"{int(amt):,}"
    return f"{amt:,.2f}"

def get_event_description(event_id: str, events_df: pd.DataFrame) -> str:
    sub = events_df[events_df['event_id'] == event_id]
    if not sub.empty:
        return str(sub.iloc[0]['description']).lower()
    return "subscription"

def generate_decision_explanation(res: dict, events_df: pd.DataFrame) -> str:
    status = res['affordability_status']
    method = res['recommended_payment_method']
    curr = res['home_currency']
    req_amt = res['requested_amount']
    min_bal = res['min_bal_to_keep']
    safe_to_pay = res['amount_safe_to_pay']
    comp_date_str = res['desired_completion_date']
    comp_date_nat = format_date_natural(comp_date_str)
    
    req_amt_str = format_amount(req_amt, curr)
    min_bal_str = format_amount(min_bal, curr)
    safe_amt_str = format_amount(safe_to_pay, curr)

    # 1. Affordable Now
    if status == 'affordable_now' and method == 'full_payment':
        return f"Pay {req_amt_str} today. This leaves at least {min_bal_str} available over the next 90 days."

    # 2. Affordable With Plan - Installments
    if status == 'affordable_with_plan' and method == 'installments':
        plan = res['payment_plan']
        payments = plan.split('|')
        n_payments = len(payments)
        first_part = payments[0].split(':')
        first_date_nat = format_date_natural(first_part[0])
        p_amt = float(first_part[1])
        p_amt_str = format_amount(p_amt, curr)
        return f"Use {n_payments} installments of {p_amt_str}, starting {first_date_nat}. This leaves at least {min_bal_str} available."

    # 3. Affordable With Plan - Partial Payment
    if status == 'affordable_with_plan' and method == 'partial_payment':
        plan = res['payment_plan']
        payments = plan.split('|')
        p1_amt = float(payments[0].split(':')[1])
        p2_date_nat = format_date_natural(payments[1].split(':')[0])
        p2_amt = float(payments[1].split(':')[1])
        p1_str = format_amount(p1_amt, curr)
        p2_str = format_amount(p2_amt, curr)
        return f"Pay {p1_str} today and the remaining {p2_str} on {p2_date_nat}. This completes the full request and keeps the {min_bal_str} minimum protected."

    # 4. Affordable With Plan - Spending Changes
    if status == 'affordable_with_plan' and method == 'full_payment':
        changes = res['spending_changes_needed'].split('|')
        actions = []
        for ch in changes:
            if ch.startswith('stop:'):
                ev_id = ch.split(':')[1]
                desc = get_event_description(ev_id, events_df)
                actions.append(f"stop the {desc}")
            elif ch.startswith('reduce_to:'):
                parts = ch.split(':')
                ev_id = parts[1]
                red_amt = float(parts[2])
                desc = get_event_description(ev_id, events_df)
                red_amt_str = format_amount(red_amt, curr)
                actions.append(f"reduce the {desc} to {red_amt_str}")

        if len(actions) == 1:
            actions_text = actions[0].capitalize()
        else:
            actions_text = actions[0].capitalize() + " and " + actions[1]

        return f"{actions_text}, then pay {req_amt_str} today. This leaves at least {min_bal_str} available."

    # 5. Affordable Later - Wait
    if status == 'affordable_later' and method == 'wait':
        earliest_date_str = res['earliest_date_for_full_payment']
        earliest_date_nat = format_date_natural(earliest_date_str)
        return f"Pay {req_amt_str} in full on {earliest_date_nat}. Paying earlier would take the balance below the {min_bal_str} minimum."

    # 6. Not Affordable - Not Recommended
    if status == 'not_affordable' or method == 'not_recommended':
        # Check if user had some safe amount today
        if safe_to_pay > 0:
            return f"Do not proceed with the {req_amt_str} request. Although {safe_amt_str} is available today, the full amount cannot be completed safely within 90 days."
        else:
            return f"Do not make this payment by {comp_date_nat}. None of the available options keeps the {min_bal_str} minimum protected."

    return f"Do not proceed with {req_amt_str}. Keeps the {min_bal_str} minimum protected."
