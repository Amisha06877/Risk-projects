import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
PORTFOLIO_VALUE = 1_000_000
START_DATE = "2019-01-01"
END_DATE   = "2024-01-01"
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

assert abs(sum(WEIGHTS) - 1.0) < 1e-9, "Weights must sum to 1"

# --- Download adjusted close prices ---
print("Downloading price data...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
raw.dropna(inplace=True)
print(f"Downloaded {len(raw)} trading days for {len(TICKERS)} stocks\n")

# --- Daily percentage returns ---
returns = raw.pct_change().dropna()

# --- Weighted portfolio return each day ---
portfolio_returns = (returns * WEIGHTS).sum(axis=1)

print(f"Portfolio return stats:")
print(f"  Mean daily return : {portfolio_returns.mean():.4%}")
print(f"  Std dev (daily)   : {portfolio_returns.std():.4%}")
print(f"  Skewness          : {portfolio_returns.skew():.4f}")
print(f"  Kurtosis (excess) : {portfolio_returns.kurtosis():.4f}\n")

# --- Historical VaR ---
def historical_var(returns, confidence, portfolio_value):
    loss_quantile = 1 - confidence
    var_return = returns.quantile(loss_quantile)
    var_dollar  = abs(var_return) * portfolio_value
    return var_dollar, var_return

# --- Historical CVaR ---
def historical_cvar(returns, confidence, portfolio_value):
    loss_quantile = 1 - confidence
    var_return    = returns.quantile(loss_quantile)
    tail_returns  = returns[returns <= var_return]
    cvar_return   = tail_returns.mean()
    cvar_dollar   = abs(cvar_return) * portfolio_value
    return cvar_dollar, cvar_return

# --- Parametric VaR ---
def parametric_var(returns, confidence, portfolio_value):
    mu    = returns.mean()
    sigma = returns.std()
    z     = stats.norm.ppf(1 - confidence)
    var_return  = mu + z * sigma
    var_dollar  = abs(var_return) * portfolio_value
    return var_dollar, var_return

# --- Print Historical Results ---
print("=" * 55)
print("HISTORICAL SIMULATION RESULTS")
print("=" * 55)

hist_results = {}
for cl in CONFIDENCE_LEVELS:
    var_d, var_r   = historical_var(portfolio_returns, cl, PORTFOLIO_VALUE)
    cvar_d, cvar_r = historical_cvar(portfolio_returns, cl, PORTFOLIO_VALUE)
    hist_results[cl] = {"var": var_d, "cvar": cvar_d}
    print(f"\nConfidence Level: {cl:.0%}")
    print(f"  VaR  : ${var_d:>10,.0f}  (daily return: {var_r:.4%})")
    print(f"  CVaR : ${cvar_d:>10,.0f}  (avg tail return: {cvar_r:.4%})")
    print(f"  CVaR/VaR ratio: {cvar_d/var_d:.2f}x")

# --- Print Parametric Results ---
print("\n" + "=" * 55)
print("PARAMETRIC (NORMAL DISTRIBUTION) RESULTS")
print("=" * 55)

param_results = {}
for cl in CONFIDENCE_LEVELS:
    var_d, var_r = parametric_var(portfolio_returns, cl, PORTFOLIO_VALUE)
    param_results[cl] = {"var": var_d}
    print(f"\nConfidence Level: {cl:.0%}")
    print(f"  VaR  : ${var_d:>10,.0f}  (daily return: {var_r:.4%})")

# --- Comparison ---
print("\n" + "=" * 55)
print("COMPARISON: HISTORICAL vs PARAMETRIC VaR")
print("=" * 55)
print(f"\n{'Confidence':<12} {'Hist VaR':>12} {'Param VaR':>12} {'Difference':>12}")
print("-" * 55)

for cl in CONFIDENCE_LEVELS:
    h = hist_results[cl]["var"]
    p = param_results[cl]["var"]
    diff = h - p
    print(f"{cl:.0%}{'':8} ${h:>10,.0f}   ${p:>10,.0f}   ${diff:>+10,.0f}")

print("""
KEY INSIGHT:
  Historical VaR > Parametric VaR = real returns have fatter tails
  than the normal distribution assumes.
  CVaR always larger than VaR — captures how bad the tail really gets.
""")

# --- Charts ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(portfolio_returns, bins=80, color="steelblue", alpha=0.7, density=True, label="Daily Returns")

colors = {0.90: "gold", 0.95: "orange", 0.99: "red"}
for cl in CONFIDENCE_LEVELS:
    var_r  = portfolio_returns.quantile(1 - cl)
    cvar_r = portfolio_returns[portfolio_returns <= var_r].mean()
    ax.axvline(var_r,  color=colors[cl], linestyle="--", linewidth=1.5, label=f"VaR {cl:.0%}")
    ax.axvline(cvar_r, color=colors[cl], linestyle=":",  linewidth=1.5, label=f"CVaR {cl:.0%}")

ax.set_title("Portfolio Return Distribution\nwith VaR and CVaR thresholds")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Density")
ax.legend(fontsize=7)

ax2 = axes[1]
x_pos = np.arange(len(CONFIDENCE_LEVELS))
width = 0.35
hist_vals  = [hist_results[cl]["var"] / 1000 for cl in CONFIDENCE_LEVELS]
param_vals = [param_results[cl]["var"] / 1000 for cl in CONFIDENCE_LEVELS]

ax2.bar(x_pos - width/2, hist_vals,  width, label="Historical", color="steelblue", alpha=0.8)
ax2.bar(x_pos + width/2, param_vals, width, label="Parametric", color="salmon",    alpha=0.8)
ax2.set_title("Historical vs Parametric VaR\n($000s, $1M portfolio)")
ax2.set_ylabel("VaR ($000s)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{cl:.0%}" for cl in CONFIDENCE_LEVELS])
ax2.legend()

plt.tight_layout()
plt.savefig("var_analysis.png", dpi=150)
plt.show()
print("Chart saved to var_analysis.png")