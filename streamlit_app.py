import numpy as np
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import scipy.optimize as spop
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

st.set_page_config(page_title='Matched Pairs Simulator', page_icon=':chart_increasing:')

# Prepare session state for trades output.
if "trade_content" not in st.session_state:
    st.session_state["trade_content"] = ""

st.markdown("# :chart_with_upwards_trend: Matched Pairs Backtesting Simulator")
st.write(
    "Browse stock data from [Yahoo Finance](https://au.finance.yahoo.com/). Run a simulation on two stocks you believe to be related "
    "(e.g., Coca-Cola & Pepsi), and see how pairs trading would have worked in your selected timeframe!"
)

# User inputs
stock_1 = st.text_input("First stock ticker:", "PEP")
stock_2 = st.text_input("Second stock ticker:", "KO")
window = st.text_input("Trading window (How many days worth of trade history do you want to base your trade on?)", "255")
stocks = [stock_1, stock_2]

fee = float(st.slider("How much is your brokerage fee (%)?: ", 0.0, 1.0, 0.1)) / 100
t_threshold = st.slider("Choose a Dickey-Fuller test statistic threshold (the more negative the more unusual):", -4.0, -0.5, -2.5)
# Capture today's date once so it doesn't update on every rerun
today = datetime.today()
year_range = st.slider(
    "During what timeframe do you want to trade?",
    min_value=datetime(2020, 1, 1),
    max_value=today,
    value=(datetime(2020, 1, 1), today),
    format="DD/MM/YY"
)
(start, end) = year_range

if len(stocks) != 2:
    st.warning("Select two stocks")

window = int(window)
data = pd.DataFrame()
returns = pd.DataFrame()

# Download stock data and compute returns.
for stock in stocks:
    prices = yf.download(stock, start, end)
    data[stock] = prices["Close"]
    # Compute daily returns; last return set to 0.
    returns[stock] = np.append(
        data[stock][1:].reset_index(drop=True) / data[stock][:-1].reset_index(drop=True) - 1, 0
    )

gross_returns = np.array([])
net_returns = np.array([])
simulation_dates = []  # store simulation dates
t_s = np.array([])
stock1 = stocks[0]
stock2 = stocks[1]
old_signal = 0

run_sim = st.button("Run Simulation")

# Create placeholders for outputs.
trade_window_placeholder = st.empty()     # Right-side trade window.
stock_plot_placeholder = st.empty()         # Plot of normalized stock prices.
plot_placeholder = st.empty()               # Plot of cumulative returns.

def update_trade_window():
    trade_window_placeholder.markdown(f"""
<div id="trade-window" style="
    position: fixed;
    top: 60px;
    bottom: 10px;
    right: 0;
    width: 300px;
    overflow-y: auto;
    background-color: #262730;
    color: #f9f9f9;
    border-left: 1px solid #515267;
    padding: 10px;
    font-family: sans-serif;
">
  <h4 style="margin-top: 0;">Trades</h4>
  {st.session_state["trade_content"]}
</div>
<script>
  var tw = document.getElementById('trade-window');
  if (tw) {{
      tw.scrollTop = 0;
  }}
</script>
""", unsafe_allow_html=True)

def print_trade_day(day, position, cumulative, net, gross):
    block = f"""
<p style="margin:0; font-size:14px;">Day: {day}</p>
<p style="margin:0; font-size:14px;">Position: {position}</p>
<p style="margin:0; font-size:14px;">Cumulative returns: {cumulative}%</p>
<p style="margin:0; font-size:14px;">Today's net returns: {net}%</p>
<p style="margin:0; font-size:14px;">Today's gross returns: {gross}%</p>
<p style="margin:0; font-size:14px;">--------</p>
"""
    st.session_state["trade_content"] = block + st.session_state["trade_content"]
    update_trade_window()

# Create the figures only once.
stock_fig, stock_ax = plt.subplots()
returns_fig, returns_ax = plt.subplots()

if run_sim:
    st.session_state["trade_content"] = ""
    # Reset arrays for plotting.
    gross_returns = np.array([])
    net_returns = np.array([])
    simulation_dates = []
    
    # Simulation loop:
    for t in range(window, len(data)):
        current_date = data.index[t]
        simulation_dates.append(current_date)
        
        # Compute unit root value.
        def unit_root(b):
            a = np.average(data[stock2][t-window:t] - b * data[stock1][t-window:t])
            fair_value = a + b * data[stock1][t-window:t]
            diff = np.array(fair_value - data[stock2][t-window:t])
            diff_diff = diff[1:] - diff[:-1]
            reg = sm.OLS(diff_diff, diff[:-1])
            res = reg.fit()
            return res.params[0] / res.bse[0]
            
        res1 = spop.minimize(unit_root, data[stock1][t] / data[stock2][t], method='Nelder-Mead')
        t_opt = res1.fun
        b_opt = float(res1.x)
        a_opt = np.average(data[stock2][t-window:t] - b_opt * data[stock1][t-window:t])
        fair_value = a_opt + b_opt * data[stock1][t]
        
        if t == window:
            old_signal = 0
            signal = 0
            gross_return = 0
            net_return = 0
            net_returns = np.append(net_returns, net_return)
            gross_returns = np.append(gross_returns, gross_return)
            print_trade_day(str(current_date),
                            "Initializing simulation, no trade on first day.",
                            "0.00", "0.00", "0.00")
        elif t_opt > t_threshold:
            signal = 0
            gross_return = 0
            net_return = 0
            net_returns = np.append(net_returns, net_return)
            gross_returns = np.append(gross_returns, gross_return)
            cumulative_display = str(round(np.prod(1+net_returns)*100-100, 2)) if len(net_returns) > 0 else "0.00"
            print_trade_day(str(current_date),
                            "No trading, no significant signal.",
                            cumulative_display,
                            "0.00", "0.00")
        else:
            signal = np.sign(fair_value - data[stock2][t])
            gross_return = signal * returns[stock2][t] - signal * returns[stock1][t]
