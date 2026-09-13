"""
90-day cash flow simulation and recurrence detection engine.
Projects daily balances conservative to ensure minimum_balance_to_keep is protected.
"""

from datetime import datetime, timedelta
import calendar
import numpy as np
import pandas as pd
from code.config import FORECAST_DAYS
from code.evidence import convert_currency

FIXED_BILL_CATEGORIES = {
    'rent', 'housing', 'utilities', 'debt_repayment', 'education', 
    'insurance', 'healthcare', 'cloud_storage', 'streaming', 
    'music_subscription', 'delivery_membership', 'gym', 'family_support'
}

def extract_recurring_streams(user_id: str, request_date_str: str, events_df: pd.DataFrame, 
                              profiles_df: pd.DataFrame, msg: dict, spending_changes: list = None):
    prof = profiles_df[profiles_df['user_id'] == user_id].iloc[0]
    protect_cats = set(str(prof.get('expense_categories_to_protect', '')).split('|'))
    
    ev_u = events_df[events_df['user_id'] == user_id].copy()
    hist_debits = ev_u[(ev_u['direction'] == 'debit') & 
                       (ev_u['settlement_date'] < request_date_str) & 
                       (ev_u['status'] == 'settled')]
                       
    stopped_events = set()
    reduced_events = {}
    if spending_changes:
        for sc in spending_changes:
            if sc.startswith('stop:'):
                stopped_events.add(sc.split(':')[1])
            elif sc.startswith('reduce_to:'):
                parts = sc.split(':')
                reduced_events[parts[1]] = float(parts[2])

    fixed_streams = []
    for (cat, desc, flex), grp in hist_debits.groupby(['category', 'description', 'flexibility']):
        if cat in FIXED_BILL_CATEGORIES:
            dates = sorted([datetime.strptime(d, '%Y-%m-%d').date() for d in grp['settlement_date']])
            if len(dates) >= 2:
                last_row = grp.sort_values('settlement_date').iloc[-1]
                last_date = dates[-1]
                last_amt = float(last_row['amount'])
                last_ev_id = last_row['event_id']
                min_allowed = float(last_row['minimum_allowed_amount']) if pd.notna(last_row['minimum_allowed_amount']) else None
                
                if last_ev_id in stopped_events:
                    continue
                if last_ev_id in reduced_events:
                    last_amt = reduced_events[last_ev_id]
                elif cat in ['rent', 'housing'] and msg.get('rent_increase_pct'):
                    last_amt *= (1.0 + msg['rent_increase_pct'])
                    
                fixed_streams.append({
                    'category': cat,
                    'description': desc,
                    'flexibility': flex,
                    'amount': last_amt,
                    'due_day': last_date.day,
                    'last_date': last_date,
                    'event_id': last_ev_id,
                    'min_allowed': min_allowed
                })
                
    var_streams = []
    # Essential variable spending: groceries and transport
    for cat in ['groceries', 'transport']:
        sub = hist_debits[hist_debits['category'] == cat]
        if not sub.empty:
            dates = sorted([datetime.strptime(d, '%Y-%m-%d').date() for d in sub['settlement_date']])
            if len(dates) >= 3:
                diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                med_diff = max(1, int(round(np.median(diffs))))
                avg_amt = float(sub['amount'].mean())
                var_streams.append({
                    'category': cat,
                    'interval_days': med_diff,
                    'last_date': dates[-1],
                    'amount': avg_amt
                })
                
    return fixed_streams, var_streams

def simulate_daily_balances(user_id: str, request_date_str: str, events_df: pd.DataFrame, 
                            profiles_df: pd.DataFrame, rates_df: pd.DataFrame, 
                            messages_by_user: dict, spending_changes: list = None, 
                            horizon_days: int = FORECAST_DAYS):
    prof = profiles_df[profiles_df['user_id'] == user_id].iloc[0]
    home_curr = prof['home_currency']
    min_bal = float(prof['minimum_balance_to_keep'])
    init_bal = float(prof['current_available_balance'])
    
    req_date = datetime.strptime(request_date_str, '%Y-%m-%d').date()
    msg = messages_by_user.get(user_id, {})
    ev_u = events_df[events_df['user_id'] == user_id].copy()
    
    hist_salaries = ev_u[(ev_u['category'] == 'salary') & 
                         (ev_u['direction'] == 'credit') & 
                         (ev_u['status'] == 'settled')]
                         
    base_sal_amt = None
    base_sal_day = 15
    if not hist_salaries.empty:
        reg_sal = hist_salaries[hist_salaries['description'].str.contains('payroll|regular salary', case=False, na=False)]
        sal_to_use = reg_sal if not reg_sal.empty else hist_salaries
        days = [datetime.strptime(d, '%Y-%m-%d').date().day for d in sal_to_use['settlement_date']]
        base_sal_day = int(pd.Series(days).mode()[0])
        matching_sal = sal_to_use[sal_to_use['settlement_date'].apply(lambda d: datetime.strptime(d, '%Y-%m-%d').date().day == base_sal_day)]
        if not matching_sal.empty:
            base_sal_amt = float(matching_sal.sort_values('settlement_date').iloc[-1]['amount'])
        else:
            base_sal_amt = float(sal_to_use.sort_values('settlement_date').iloc[-1]['amount'])
        
    future_sal = ev_u[(ev_u['category'] == 'salary') & 
                      (ev_u['direction'] == 'credit') & 
                      (ev_u['settlement_date'] >= request_date_str) & 
                      (ev_u['status'].isin(['scheduled', 'settled']))]
    if not future_sal.empty:
        fs = future_sal.sort_values('settlement_date').iloc[0]
        base_sal_amt = float(fs['amount'])
        base_sal_day = datetime.strptime(fs['settlement_date'], '%Y-%m-%d').date().day

    # Check for "final employer payroll" in description (employment ended)
    has_final_sal = ev_u['description'].str.contains('final employer payroll', case=False, na=False).any()
    emp_ended = msg.get('employment_ended', False) or has_final_sal

    if msg.get('salary_amount'):
        base_sal_amt = msg['salary_amount']
    if msg.get('salary_date'):
        msg_sdt = datetime.strptime(msg['salary_date'], '%Y-%m-%d').date()
        base_sal_day = msg_sdt.day

    fixed_streams, var_streams = extract_recurring_streams(user_id, request_date_str, events_df, profiles_df, msg, spending_changes)

    daily_balances = {}
    curr_bal = init_bal
    
    explicit_future = ev_u[(ev_u['settlement_date'] >= request_date_str) & 
                           (ev_u['status'].isin(['pending', 'scheduled', 'settled']))]
                           
    first_salary_done = False
    
    for day_idx in range(horizon_days + 1):
        day_date = req_date + timedelta(days=day_idx)
        day_str = str(day_date)
        net_change = 0.0
        
        day_events = explicit_future[explicit_future['settlement_date'] == day_str]
        for _, dev in day_events.iterrows():
            amt = float(dev['amount'])
            amt = convert_currency(amt, dev['currency'], home_curr, day_str, rates_df)
            if dev['direction'] == 'debit':
                net_change -= amt
            elif dev['direction'] == 'credit':
                if dev['category'] == 'salary' or dev['status'] == 'settled':
                    net_change += amt
                    
        if msg.get('confirmed_invoice'):
            cinv = msg['confirmed_invoice']
            if cinv['date'] == day_str:
                iamt = convert_currency(cinv['amount'], cinv['currency'], home_curr, day_str, rates_df)
                net_change += iamt

        has_explicit_salary = (day_events['category'] == 'salary').any()
        if not emp_ended and not has_explicit_salary:
            is_payday = False
            if day_date.day == base_sal_day:
                is_payday = True
            elif base_sal_day > 28 and day_date.day == calendar.monthrange(day_date.year, day_date.month)[1]:
                is_payday = True
                
            if is_payday and base_sal_amt is not None:
                sal_payout = base_sal_amt
                if not first_salary_done and msg.get('salary_arrears'):
                    sal_payout += msg['salary_arrears']
                    first_salary_done = True
                net_change += sal_payout
                
        for fs in fixed_streams:
            due_day = fs['due_day']
            is_due = False
            if day_date.day == due_day:
                is_due = True
            elif due_day > 28 and day_date.day == calendar.monthrange(day_date.year, day_date.month)[1]:
                is_due = True
                
            if is_due:
                matched_exp = day_events[(day_events['category'] == fs['category']) & 
                                         (day_events['direction'] == 'debit')]
                if matched_exp.empty:
                    net_change -= fs['amount']
                    
        for vs in var_streams:
            days_since = (day_date - vs['last_date']).days
            if days_since > 0 and days_since % vs['interval_days'] == 0:
                matched_exp = day_events[(day_events['category'] == vs['category']) & 
                                         (day_events['direction'] == 'debit')]
                if matched_exp.empty:
                    net_change -= vs['amount']
                    
        curr_bal += net_change
        daily_balances[day_date] = curr_bal
        
    # Calculate net monthly outflow and net monthly savings
    total_monthly_fixed = sum(fs['amount'] for fs in fixed_streams)
    total_monthly_var = sum((30.0 / vs['interval_days']) * vs['amount'] for vs in var_streams)
    total_monthly_expenses = total_monthly_fixed + total_monthly_var
    net_monthly_savings = max(0.0, (base_sal_amt or 0.0) - total_monthly_expenses) if not emp_ended else 0.0

    return {
        'daily_balances': daily_balances,
        'min_bal': min_bal,
        'init_bal': init_bal,
        'base_sal_amt': base_sal_amt,
        'base_sal_day': base_sal_day,
        'net_monthly_savings': net_monthly_savings,
        'fixed_streams': fixed_streams,
        'var_streams': var_streams,
        'emp_ended': emp_ended
    }
