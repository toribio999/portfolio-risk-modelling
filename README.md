# Portfolio Risk Modelling: GARCH Volatility, DCC Correlations, VaR & Monte Carlo

## Summary

This project develops a two-stage market-risk modelling workflow in Python, progressing from the analysis of a single stock to a multi-asset equity portfolio. The first part focuses on **NVDA**, analysing return distributions, volatility clustering and asymmetric volatility before estimating GARCH-family models and translating the results into one-day VaR measures. The second part extends the framework to an equally weighted portfolio of **NVDA, MSFT, JNJ and PG**, combining individual GARCH models with DCC-GARCH to capture time-varying correlations. Portfolio VaR and Expected Shortfall are then estimated, backtested and complemented with Monte Carlo simulations of future portfolio losses.

## Key Insights

### Part I — Univariate Risk: NVDA

- **Volatility clustering:** squared returns show strong autocorrelation (**Ljung–Box p ≈ 10⁻⁴²**), confirming substantial time dependence in volatility.
- **Fat-tailed returns:** the empirical distribution is clearly non-normal, with a mixture of three Gaussian components providing a better fit than a single Normal distribution.
- **Asymmetric volatility:** the GJR-GARCH(1,1) model identifies a significant leverage effect (**γ ≈ 0.085, p = 0.029**), indicating that negative shocks have a larger impact on volatility.
- **Heavy-tailed innovations:** Student-t innovations outperform Normal innovations (**AIC ≈ 18,719 vs 19,327**).
- **Tail risk:** the **99.9% one-day VaR is ≈ 10.3% under Student-t innovations vs ≈ 6.6% under Normal innovations**, showing the impact of distributional assumptions on extreme-risk estimates.

### Part II — Multivariate Risk: Equity Portfolio

- **Cross-asset risk:** annualised volatility ranges from approximately **17% for JNJ and PG to 45% for NVDA**, while excess kurtosis ranges from **6.9 to 10.6**.
- **GARCH dynamics:** individual GARCH(1,1)-t models remove the volatility dependence from the four return series, with diagnostic p-values **> 0.6**.
- **High persistence:** volatility persistence ranges from approximately **0.94 for PG to 0.996 for NVDA**.
- **Dynamic correlations:** DCC-GARCH estimates **a ≈ 0.011 and b ≈ 0.984**, giving **a + b ≈ 0.995** and indicating highly persistent correlation dynamics.
- **Risk concentration:** NVDA represents **25% of portfolio capital but approximately 58% of portfolio risk**, highlighting the difference between capital allocation and risk contribution.
- **VaR backtesting:** the DCC-Normal model records **192 breaches vs 201 expected at the 95% confidence level**, with a **Kupiec p-value ≈ 0.50**.
- **Forward-looking risk:** Monte Carlo simulations based on the final conditional covariance matrix generate **VaR, Expected Shortfall and multi-day portfolio loss scenarios**.


## 2. Technical Stack

| Area | Tools |
|---|---|
| Language / environment | Python 3, Jupyter Notebook |
| Data acquisition | `yfinance` |
| Data handling | `pandas`, `numpy` |
| Statistical modelling | `arch`, `statsmodels`, `scipy` |
| Machine learning / distributions | `scikit-learn` (`GaussianMixture`) |
| Optimisation | `scipy.optimize` (constrained QMLE for DCC) |
| Visualisation | `matplotlib`, `seaborn` |

### Structure 

Two Jupyter notebooks that build a market-risk workflow in Python, moving from a **single stock (NVDA)** to a **four-asset equity portfolio (NVDA, MSFT, JNJ, PG)**. Daily data runs from January 2010 to December 2025 (4,024 observations).

| Notebook | Scope | Core question |
|---|---|---|
| `Modelling_1.ipynb` | Univariate | How volatile is NVDA, how does that volatility behave over time, and how large can a one-day loss be? |
| `Modelling2_multivariate.ipynb` | Multivariate | How do correlations and volatilities interact at portfolio level, and how reliable is the resulting VaR? |

### Main Analytical Methods

- Log-returns, rolling volatility and distribution analysis
- ACF, Ljung-Box and ARCH-LM diagnostics
- GARCH / GJR-GARCH with AIC/BIC model selection
- Normal, Student-t, GED and Skewed-t innovation distributions
- **DCC-GARCH implemented from scratch**
- Parametric and simulated **VaR / Expected Shortfall**
- **Kupiec** VaR backtesting
- Monte Carlo simulation with Cholesky-correlated shocks

