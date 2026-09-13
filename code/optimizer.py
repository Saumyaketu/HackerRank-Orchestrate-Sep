"""
Plan optimization, spending adjustments, and multi-criteria plan ranking for Buy or Wait.
Enforces all challenge constraints from problem_statement.md.
"""

from datetime import datetime, timedelta
import calendar
import pandas as pd
from code.config import FORECAST_DAYS
from code.forecaster import simulate_daily_balances

def format_plan_amt(amt: float) -> str:
    if abs(amt - round(amt)) < 1e-5:
        return str(int(round(amt)))
    return f"{amt:.2f}"

def add_months(date_value, months: int):
    month_index = date_value.year * 12 + date_value.month - 1 + months
    year, month_index = divmod(month_index, 12)
    month = month_index + 1
    day = min(date_value.day, calendar.monthrange(year, month)[1])
    return date_value.replace(year=year, month=month, day=day)

def select_request_messages(messages_by_user: dict, user_id: str, request_id: str) -> dict:
    messages = messages_by_user.get(user_id, [])
    if isinstance(messages, dict):
        return messages

    relevant = [message for message in messages
                if message.get('request_id') in (None, request_id)]
    relevant.sort(key=lambda message: str(message.get('sent_at', '')))
    merged = {}
    for message in relevant:
        for key, value in message.items():
            if key in {'user_id', 'request_id', 'related_event_id', 'sent_at'}:
                continue
            if value is not None and value is not False:
                merged[key] = value
    return merged

def solve_financial_request(req_row: pd.Series, events_df: pd.DataFrame, 
                            profiles_df: pd.DataFrame, options_df: pd.DataFrame, 
                            rates_df: pd.DataFrame, messages_by_user: dict) -> dict:
    req_id = req_row['request_id']
    user_id = req_row['user_id']
    req_date_str = req_row['request_date']
    req_date = datetime.strptime(req_date_str, '%Y-%m-%d').date()
    req_amt = float(req_row['requested_amount'])
    comp_date_str = req_row['desired_completion_date']
    comp_date = datetime.strptime(comp_date_str, '%Y-%m-%d').date()
    allows_partial = str(req_row.get('allows_partial_payment', '')).lower() in ['true', '1', 'yes']

    prof = profiles_df[profiles_df['user_id'] == user_id].iloc[0]
    home_curr = prof['home_currency']
    min_bal = float(prof['minimum_balance_to_keep'])
    user_methods = str(prof['payment_methods_user_will_consider']).split('|')
    
    max_inst_months = prof['max_installment_months']
    if pd.notna(max_inst_months) and str(max_inst_months).strip() != '':
        max_inst_months = float(max_inst_months)
    else:
        max_inst_months = 0

    willing_stop_cats = set(str(prof.get('expense_categories_user_is_willing_to_stop', '')).split('|'))
    willing_red_cats = set(str(prof.get('expense_categories_user_is_willing_to_reduce', '')).split('|'))
    protect_cats = set(str(prof.get('expense_categories_to_protect', '')).split('|'))

    request_messages = select_request_messages(messages_by_user, user_id, req_id)

    # 1. Baseline 90-day simulation
    sim_res = simulate_daily_balances(user_id, req_date_str, events_df, profiles_df, rates_df, {user_id: request_messages})
    daily_balances = sim_res['daily_balances']
    base_sal_day = sim_res['base_sal_day']
    fixed_streams = sim_res['fixed_streams']
    emp_ended = sim_res['emp_ended']

    # Calculate amount_safe_to_pay on request_date
    min_bal_over_90 = min(daily_balances[req_date + timedelta(days=d)] for d in range(FORECAST_DAYS))
    amount_safe_to_pay = max(0.0, min_bal_over_90 - min_bal)
    amount_safe_to_pay = min(req_amt, round(amount_safe_to_pay, 2))

    # Calculate earliest_date_for_full_payment independently of payment preferences
    earliest_date_for_full = None
    if amount_safe_to_pay >= req_amt:
        earliest_date_for_full = req_date_str
    else:
        # Find the first date on which a full payment remains safe for the rest
        # of the forecast, including confirmed future income and expenses.
        for d in range(1, FORECAST_DAYS):
            test_dt = req_date + timedelta(days=d)
            is_safe = all(
                daily_balances[req_date + timedelta(days=fut_d)] - req_amt >= min_bal - 0.05
                for fut_d in range(d, FORECAST_DAYS)
            )
            if is_safe:
                earliest_date_for_full = str(test_dt)
                break

    candidates = []

    # Option 1: Full payment today
    if 'full_payment' in user_methods and amount_safe_to_pay >= req_amt:
        candidates.append({
            'status': 'affordable_now',
            'method': 'full_payment',
            'plan': f"{req_date_str}:{format_plan_amt(req_amt)}",
            'earliest_date': req_date_str,
            'spending_changes': 'none',
            'total_amount': req_amt,
            'completes_by_deadline': True,
            'requires_spending_changes': False,
            'start_date': req_date,
            'num_payments': 1,
            'option_id': 0
        })

    # Option 2: Installments from request_payment_options.csv
    if 'installments' in user_methods and max_inst_months > 0:
        opts = options_df[(options_df['request_id'] == req_id) & 
                          (options_df['payment_method'] == 'installments')]
        for _, opt in opts.iterrows():
            n_payments = int(opt['number_of_payments'])
            p_amt = float(opt['payment_amount'])
            freq = int(opt['payment_frequency_days'])
            first_dt = datetime.strptime(opt['first_payment_date'], '%Y-%m-%d').date()
            tot_amt = float(opt['total_payable_amount'])

            if first_dt < req_date:
                continue
            last_dt = first_dt + timedelta(days=(n_payments - 1) * freq)
            if last_dt > add_months(first_dt, int(max_inst_months)):
                continue

            pay_dates = [first_dt + timedelta(days=i * freq) for i in range(n_payments)]
            completes_on_time = pay_dates[-1] <= comp_date

            # Simulate daily balance safety with installments deducted
            is_plan_safe = True
            for d in range(FORECAST_DAYS):
                cur_dt = req_date + timedelta(days=d)
                cum_paid = sum(p_amt for pdt in pay_dates if pdt <= cur_dt)
                if daily_balances[cur_dt] - cum_paid < min_bal - 0.05:
                    is_plan_safe = False
                    break

            if is_plan_safe:
                plan_str = "|".join(f"{str(pdt)}:{format_plan_amt(p_amt)}" for pdt in pay_dates)
                candidates.append({
                    'status': 'affordable_with_plan',
                    'method': 'installments',
                    'plan': plan_str,
                    'earliest_date': earliest_date_for_full or '',
                    'spending_changes': 'none',
                    'total_amount': tot_amt,
                    'completes_by_deadline': completes_on_time,
                    'requires_spending_changes': False,
                    'start_date': first_dt,
                    'num_payments': n_payments,
                    'option_id': int(opt['payment_option_id'].split('_')[-1]),
                    'opt_row': opt
                })

    # Option 3: Partial payment
    if 'partial_payment' in user_methods and allows_partial and amount_safe_to_pay > 0 and amount_safe_to_pay < req_amt:
        if earliest_date_for_full and earliest_date_for_full <= comp_date_str:
            p2_date = datetime.strptime(earliest_date_for_full, '%Y-%m-%d').date()
            rem_amt = round(req_amt - amount_safe_to_pay, 2)
            is_plan_safe = True
            for d in range(FORECAST_DAYS):
                cur_dt = req_date + timedelta(days=d)
                cum_paid = amount_safe_to_pay if cur_dt >= req_date else 0.0
                if cur_dt >= p2_date:
                    cum_paid += rem_amt
                if daily_balances[cur_dt] - cum_paid < min_bal - 0.05:
                    is_plan_safe = False
                    break
            if is_plan_safe:
                plan_str = f"{req_date_str}:{format_plan_amt(amount_safe_to_pay)}|{earliest_date_for_full}:{format_plan_amt(rem_amt)}"
                candidates.append({
                    'status': 'affordable_with_plan',
                    'method': 'partial_payment',
                    'plan': plan_str,
                    'earliest_date': earliest_date_for_full,
                    'spending_changes': 'none',
                    'total_amount': req_amt,
                    'completes_by_deadline': True,
                    'requires_spending_changes': False,
                    'start_date': req_date,
                    'num_payments': 2,
                    'option_id': 999
                })

    # Option 4: Spending changes (if needed to enable payment by deadline)
    if ('full_payment' in user_methods) and amount_safe_to_pay < req_amt:
        stoppable_candidates = [fs for fs in fixed_streams if fs['category'] in willing_stop_cats and fs['category'] not in protect_cats and fs['flexibility'] in ['stoppable', 'reducible_or_stoppable']]
        reducible_candidates = [fs for fs in fixed_streams if fs['category'] in willing_red_cats and fs['category'] not in protect_cats and fs['flexibility'] in ['reducible', 'reducible_or_stoppable']]

        best_change_combo = None
        # Try 1 stop
        for sc in stoppable_candidates:
            change = [f"stop:{sc['event_id']}"]
            b_res = simulate_daily_balances(user_id, req_date_str, events_df, profiles_df, rates_df, {user_id: request_messages}, spending_changes=change)
            min_b = min(b_res['daily_balances'][req_date + timedelta(days=d)] for d in range(FORECAST_DAYS))
            if min_b - min_bal >= req_amt - 1e-4:
                best_change_combo = change
                break

        # Try 1 reduce
        if not best_change_combo:
            for rc in reducible_candidates:
                if rc['min_allowed'] is not None:
                    change = [f"reduce_to:{rc['event_id']}:{format_plan_amt(rc['min_allowed'])}"]
                    b_res = simulate_daily_balances(user_id, req_date_str, events_df, profiles_df, rates_df, {user_id: request_messages}, spending_changes=change)
                    min_b = min(b_res['daily_balances'][req_date + timedelta(days=d)] for d in range(FORECAST_DAYS))
                    if min_b - min_bal >= req_amt - 1e-4:
                        best_change_combo = change
                        break

        # Try 1 stop + 1 reduce
        if not best_change_combo:
            for sc in stoppable_candidates:
                for rc in reducible_candidates:
                    if sc['event_id'] == rc['event_id'] or rc['min_allowed'] is None:
                        continue
                    change = [f"stop:{sc['event_id']}", f"reduce_to:{rc['event_id']}:{format_plan_amt(rc['min_allowed'])}"]
                    b_res = simulate_daily_balances(user_id, req_date_str, events_df, profiles_df, rates_df, {user_id: request_messages}, spending_changes=change)
                    min_b = min(b_res['daily_balances'][req_date + timedelta(days=d)] for d in range(FORECAST_DAYS))
                    if min_b - min_bal >= req_amt - 1e-4:
                        best_change_combo = change
                        break
                if best_change_combo:
                    break

        if best_change_combo:
            candidates.append({
                'status': 'affordable_with_plan',
                'method': 'full_payment',
                'plan': f"{req_date_str}:{format_plan_amt(req_amt)}",
                'earliest_date': earliest_date_for_full or '',
                'spending_changes': "|".join(best_change_combo),
                'total_amount': req_amt,
                'completes_by_deadline': True,
                'requires_spending_changes': True,
                'start_date': req_date,
                'num_payments': 1,
                'option_id': 1000
            })

    # Option 5: Wait
    if 'full_payment' in user_methods and earliest_date_for_full and earliest_date_for_full != req_date_str:
        wait_completes_on_time = earliest_date_for_full <= comp_date_str
        candidates.append({
            'status': 'affordable_later',
            'method': 'wait',
            'plan': f"{earliest_date_for_full}:{format_plan_amt(req_amt)}",
            'earliest_date': earliest_date_for_full,
            'spending_changes': 'none',
            'total_amount': req_amt,
            'completes_by_deadline': wait_completes_on_time,
            'requires_spending_changes': False,
            'start_date': datetime.strptime(earliest_date_for_full, '%Y-%m-%d').date(),
            'num_payments': 1,
            'option_id': 1001
        })

    # Plan ranking according to §6.3:
    # 1. Complete full request by desired_completion_date
    # 2. Require no spending changes
    # 3. Minimize total amount paid
    # 4. Start payment earlier
    # 5. Use fewer payments
    # 6. Lowest option_id
    if candidates:
        candidates.sort(key=lambda c: (
            not c['completes_by_deadline'],
            c['requires_spending_changes'],
            c['total_amount'],
            c['start_date'],
            c['num_payments'],
            c['option_id']
        ))
        best = candidates[0]
    else:
        best = {
            'status': 'not_affordable',
            'method': 'not_recommended',
            'plan': 'none',
            'earliest_date': '',
            'spending_changes': 'none',
            'total_amount': req_amt,
        }

    return {
        'request_id': req_id,
        'user_id': user_id,
        'home_currency': home_curr,
        'requested_amount': req_amt,
        'desired_completion_date': comp_date_str,
        'amount_safe_to_pay': amount_safe_to_pay,
        'affordability_status': best['status'],
        'recommended_payment_method': best['method'],
        'payment_plan': best['plan'],
        'earliest_date_for_full_payment': best['earliest_date'],
        'spending_changes_needed': best['spending_changes'],
        'best_candidate': best,
        'min_bal_to_keep': min_bal,
    }
