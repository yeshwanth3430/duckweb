# 📈 Building a SuperTrend Backtesting Dashboard: Step-by-Step

## Introduction

Backtesting is a vital part of any trading strategy development. In this project, we created an interactive dashboard to backtest the **SuperTrend** indicator (and others like EMA/SMA) on Indian market indices. This post explains the logic, code, and user experience so you can understand and extend the system for your own research.

---

## 1. SuperTrend Signal Generation

We use the SuperTrend indicator to generate buy/sell signals:
- **Bullish Signal:** When price crosses above the SuperTrend line.
- **Bearish Signal:** When price crosses below the SuperTrend line.

The code calculates SuperTrend and adds its columns (`supertrend`, `direction`, etc.) to the data for each backtest run.

---

## 2. Backtest Logic

For each R:R (Risk:Reward) value, we simulate trades as follows:

**Trade Entry:**  
- Enter a trade at the open of the next bar after a Bullish/Bearish signal.
- Direction: "long" for Bullish, "short" for Bearish.

**Trade Management:**  
- **Signal Flip:** If the signal flips, exit at the open of the flip bar, record the trade, and immediately open a new trade in the opposite direction.
- **Stop Loss:**  
  - For long: If the low <= SuperTrend at entry, exit at SuperTrend value.
  - For short: If the high >= SuperTrend at entry, exit at SuperTrend value.
- **R:R Target:**  
  - For long: If high >= target, exit at target.
  - For short: If low <= target, exit at target.

**End of Data:**  
- If still in a trade, close at the last close price.

**Trade Recording:**  
- Each trade records entry/exit time, direction, prices, and points.

---

## 3. Results and Visualization

- **Backtest Results Table:**  
  Shows win rate, expectancy, max drawdown, cumulative PnL, and number of trades for each R:R.

- **Trades History Table:**  
  Lets the user select an R:R value and see all trades for that setting, including entry/exit times, direction, prices, and profit/loss.

- **Interactive UI:**  
  The dashboard is built with Streamlit, so users can select indicators, timeframes, date ranges, and R:R values—all with instant feedback.

---

## Example: What the User Sees

**Backtest Results Table:**  
(Insert screenshot here)

**Trades History Table:**  
(Insert screenshot here)

---

## Why This Approach Works

- **Transparency:** Every trade is shown, so you can verify the logic and results.
- **Flexibility:** You can test different R:R values, indicators, and timeframes.
- **Reproducibility:** The logic is clear and can be extended to other strategies.

---

## How to Extend

- Add slippage/commission for more realistic results.
- Add more indicators or custom entry/exit rules.
- Visualize trades on the price chart for even more insight.

---

## Conclusion

With this dashboard, you can quickly and visually backtest SuperTrend (and other) strategies, see detailed trade logs, and iterate on your ideas.  
**Happy backtesting and trading!** 