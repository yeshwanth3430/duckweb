import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_straddle_payoff(spot_price, strike_price, call_premium, put_premium):
    """Calculate straddle strategy payoff"""
    payoff = []
    for price in spot_price:
        call_payoff = max(price - strike_price, 0) - call_premium
        put_payoff = max(strike_price - price, 0) - put_premium
        total_payoff = call_payoff + put_payoff
        payoff.append(total_payoff)
    return payoff

def analyze_straddle_strategy(spot_price, strike_price, call_premium, put_premium, 
                            days_to_expiry, risk_free_rate=0.05):
    """Analyze straddle strategy with various metrics"""
    # Calculate payoff
    payoff = calculate_straddle_payoff(spot_price, strike_price, call_premium, put_premium)
    
    # Calculate breakeven points
    total_premium = call_premium + put_premium
    upper_breakeven = strike_price + total_premium
    lower_breakeven = strike_price - total_premium
    
    # Calculate maximum profit and loss
    max_profit = float('inf')  # Unlimited upside
    max_loss = -total_premium  # Limited to premium paid
    
    # Calculate probability of profit
    profitable_payoffs = sum(1 for p in payoff if p > 0)
    prob_profit = profitable_payoffs / len(payoff) if payoff else 0
    
    # Calculate expected return
    expected_return = np.mean(payoff)
    
    # Calculate risk metrics
    std_dev = np.std(payoff)
    sharpe_ratio = (expected_return / std_dev) * np.sqrt(252/days_to_expiry) if std_dev != 0 else 0
    
    return {
        'payoff': payoff,
        'upper_breakeven': upper_breakeven,
        'lower_breakeven': lower_breakeven,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'prob_profit': prob_profit,
        'expected_return': expected_return,
        'sharpe_ratio': sharpe_ratio
    }

def get_straddle_recommendations(spot_price, volatility, days_to_expiry, 
                               risk_free_rate=0.05, target_return=0.1):
    """Generate straddle strategy recommendations"""
    # Calculate ATM strike
    atm_strike = round(spot_price / 100) * 100
    
    # Calculate option premiums using Black-Scholes (simplified)
    time_to_expiry = days_to_expiry / 365
    d1 = (np.log(spot_price/atm_strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Calculate call and put premiums
    call_premium = spot_price * 0.4 * volatility * np.sqrt(time_to_expiry)
    put_premium = spot_price * 0.4 * volatility * np.sqrt(time_to_expiry)
    
    # Generate price range for analysis
    price_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 100)
    
    # Analyze strategy
    analysis = analyze_straddle_strategy(
        price_range, atm_strike, call_premium, put_premium,
        days_to_expiry, risk_free_rate
    )
    
    # Generate recommendations
    recommendations = {
        'strike_price': atm_strike,
        'call_premium': call_premium,
        'put_premium': put_premium,
        'total_premium': call_premium + put_premium,
        'breakeven_points': {
            'upper': analysis['upper_breakeven'],
            'lower': analysis['lower_breakeven']
        },
        'risk_metrics': {
            'max_loss': analysis['max_loss'],
            'prob_profit': analysis['prob_profit'],
            'expected_return': analysis['expected_return'],
            'sharpe_ratio': analysis['sharpe_ratio']
        }
    }
    
    return recommendations
