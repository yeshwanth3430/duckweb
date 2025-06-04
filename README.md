# Indian Market Analysis Dashboard

A comprehensive dashboard for analyzing Indian market indices (Nifty, Bank Nifty, and Sensex) with advanced technical analysis tools.

## Features

- **Spot Market Analysis**
  - Gap Analysis
  - Volume Analysis
  - Trend Analysis
  - Support/Resistance Analysis
  - Technical Indicators (Moving Averages, MACD, SuperTrend)

- **Options Analysis** (Coming Soon)
  - Straddle Strategy Analysis
  - Options Chain Analysis
  - Greeks Analysis

## Project Structure

```
/
├── main.py (Streamlit entry point)
├── spot.py (Spot market analysis module)
├── analysis_tools.py (Analysis tools including Volume Analysis, Support/Resistance)
├── technical_indicator/ (Technical indicators folder)
│   ├── __init__.py
│   ├── moving_average.py
│   ├── super_trend.py
│   └── macd.py
├── visualization/ (Visualization components)
│   ├── __init__.py
│   └── charts.py
├── options/ (Options analysis module)
│   ├── __init__.py
│   └── straddle.py
├── db_operations.py (Database operations)
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/indian-market-analysis.git
cd indian-market-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Select an index and date range to begin analysis

## Technical Indicators

- **Moving Averages**
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Multiple timeframes (20, 50, 200 periods)

- **MACD**
  - MACD Line
  - Signal Line
  - Histogram

- **SuperTrend**
  - Customizable period and multiplier
  - Trend direction signals

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 