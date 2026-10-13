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

## Part I — Univariate Risk: NVDA

### Data Download

Historical daily price data for NVIDIA (NVDA) was retrieved using `yfinance`, covering the period from January 2010 to December 2025. The dataset includes adjusted closing prices, daily highs and lows, opening prices, and trading volume.

The data was used to calculate daily log-returns and analyze the evolution of NVIDIA's market price and volatility over time.

| Date | Close | High | Low | Open | Volume |
|---|---:|---:|---:|---:|---:|
| 2025-12-24 | 188.169678 | 188.468980 | 186.154389 | 187.501244 | 65,528,500 |
| 2025-12-26 | 190.085205 | 192.240166 | 187.561113 | 189.476629 | 139,740,300 |
| 2025-12-29 | 187.780594 | 188.319327 | 185.475989 | 187.271790 | 120,006,100 |
| 2025-12-30 | 187.102188 | 188.548815 | 186.493612 | 187.800566 | 97,687,300 |
| 2025-12-31 | 186.064606 | 190.115125 | 186.054635 | 189.127446 | 120,100,500 |

![NVDA Historical Price](images/price_nvda.png)

### Returns and Log-Returns

Daily simple returns and log-returns were calculated from adjusted closing prices to analyze NVIDIA's daily performance. These measures help characterize the distribution of returns, identify extreme market movements, and assess the statistical properties of the series.

| Statistic | Price | Return | Log Return |
|---|---:|---:|---:|
| Mean | 22.61 | 0.00193 | 0.00151 |
| Std. Dev. | 43.89 | 0.02879 | 0.02861 |
| Minimum | 0.20 | -0.18756 | -0.20771 |
| Median | 3.97 | 0.00178 | 0.00174 |
| Maximum | 206.55 | 0.29807 | 0.26088 |

Daily log returns have a mean of approximately **0.15%** and a standard deviation of **2.86%**, with observed daily losses and gains reaching approximately **-20.8% and +26.1%**, respectively.

![NVDA Returns and Log-Returns](images/returns_nvda.png)

### Kernel Density Estimation (KDE)

Kernel Density Estimation (KDE) was used to visualize the empirical distribution of NVIDIA's daily log-returns. This non-parametric approach helps assess the distribution's shape, including its concentration around zero, asymmetry, and potential heavy tails. The empirical distribution of NVIDIA's daily log-returns was compared with a normal distribution and Gaussian Mixture Models (GMMs) with two and three components. This comparison evaluates whether mixtures of Gaussian distributions can better capture the characteristics of financial returns, including asymmetry, heavy tails, and potential heterogeneity in market conditions.

![NVDA Log-Returns KDE](images/kde_nvda.png)

The log-returns of NVDA are highly concentrated around zero and exhibit a pronounced peak with relatively heavy tails. The single Gaussian distribution does not fully capture the empirical shape of the returns. Mixture models provide a more flexible representation, with the three-Gaussian mixture appearing to fit the kernel density particularly well. This suggests that NVDA returns may exhibit non-normal and potentially heterogeneous behavior that is better captured by a mixture distribution.

### Autocorrelation Analysis

The Autocorrelation Function (ACF) was examined for both daily log-returns and squared log-returns. The former helps identify serial dependence in returns, while the latter helps detect volatility clustering and persistence, common characteristics of financial time series.

![ACF of Log-Returns and Squared Log-Returns](images/acf_nvda.png)

The ACF of the log-returns shows little evidence of significant autocorrelation, suggesting that returns are approximately uncorrelated over time. In contrast, the squared log-returns exhibit significant positive autocorrelation, indicating volatility clustering: periods of high (or low) volatility tend to be followed by periods of similarly high (or low) volatility. This is a common feature of financial returns and provides motivation for using GARCH-type models.

### GJR-GARCH Model Fitting

A GJR-GARCH(1,1) model was fitted to NVIDIA's daily log-returns to capture volatility clustering and asymmetric responses to market shocks. The model allows negative shocks to have a different impact on conditional volatility than positive shocks of the same magnitude.
For the asset, returns are modeled as:

$$
r_t = \mu + \varepsilon_t,
\qquad
\varepsilon_t = \sigma_t z_t,
\qquad
z_t \sim \mathcal{N}(0,1)
$$

The conditional variance follows a GJR-GARCH(1,1) specification:

$$
\sigma_t^2 =
\omega
+
\left(
\alpha +
\gamma \mathbb{1}_{\{\varepsilon_{t-1}<0\}}
\right)
\varepsilon_{t-1}^2
+
\beta \sigma_{t-1}^2
$$

where:

- $\omega > 0$ is the long-run variance intercept.
- $\alpha$ captures the impact of past shocks on current volatility.
- $\gamma$ captures the **asymmetric effect** of negative shocks.
- $\beta$ measures **volatility persistence**.
- $\mathbb{1}_{\{\varepsilon_{t-1}<0\}}$ is an indicator equal to 1 when the previous shock is negative and 0 otherwise.

A negative shock therefore has an impact of $(\alpha+\gamma)$, while a positive shock has an impact of $\alpha$. A statistically significant $\gamma$ indicates that negative returns generate a stronger volatility response than positive returns of the same magnitude.

### NVDA Results

The GJR-GARCH(1,1) model was estimated using 4,024 daily observations under a Normal innovation distribution.

| Parameter | Estimate | p-value |
|---|---:|---:|
| $\mu$ | 0.1725 | < 0.001 |
| $\omega$ | 0.3206 | 0.067 |
| $\alpha$ | 0.0537 | 0.061 |
| $\gamma$ | 0.0852 | 0.029 |
| $\beta$ | 0.8701 | < 0.001 |

The model achieves:

| Log-Likelihood | AIC | BIC |
|---:|---:|---:|
| -9,644.24 | 19,298.5 | 19,330.0 |

### Interpretation

The estimated $\gamma=0.0852$ is statistically significant at the 5% level, providing evidence of an **asymmetric volatility response**. Negative shocks therefore have a larger effect on future volatility than positive shocks of the same magnitude.

Volatility is also highly persistent, with $\beta=0.8701$. Under the usual approximation for GJR-GARCH with symmetric innovations, the persistence can be expressed as:

$$
P \approx \alpha + \beta + \frac{\gamma}{2}
$$

giving:

$$
P \approx 0.0537 + 0.8701 + \frac{0.0852}{2}
\approx 0.9664
$$

This indicates that volatility shocks in NVDA decay slowly over time, consistent with strong volatility clustering.

### Conditional Volatility Analysis

The fitted GJR-GARCH model was used to estimate NVIDIA's time-varying conditional volatility. The resulting series captures periods of increased and decreased market uncertainty, highlighting volatility clustering and the persistence of volatility over time.

![NVDA GJR-GARCH Conditional Volatility](images/vol_garch.png)

### Innovation Distribution Comparison

To account for the heavy tails observed in the standardized residuals, the GJR-GARCH model was estimated under four different innovation distributions.

![NVDA GJR-GARCH Standardized Residuals](images/residuals_distr.png)

| Distribution | Log-Likelihood | AIC | BIC |
|---|---:|---:|---:|
| **Student-t** | **-9,354.30** | **18,718.61** | **18,750.11** |
| Skewed Student-t | -9,353.98 | 18,719.97 | 18,757.77 |
| GED | -9,399.78 | 18,809.56 | 18,841.06 |
| Normal | -9,659.30 | 19,326.60 | 19,351.80 |

The heavy-tailed specifications provide a substantially better fit than the Normal distribution. The **Student-t model is selected as the preferred specification**, achieving the lowest AIC and BIC, while the Skewed Student-t provides a very similar log-likelihood at the cost of an additional parameter.

The results support the presence of **heavy-tailed return innovations**, making the Student-t specification more appropriate for subsequent volatility and risk analysis.

### Value at Risk (VaR)

VaR is estimated at 5%, 1%, and 0.1% confidence levels under four alternative innovation distributions: Normal, Student-t, GED, and Skewed Student-t. This comparison evaluates how distributional assumptions affect estimated potential losses, particularly under extreme market conditions.

| Distribution | Confidence Level | Expected Return | Volatility | VaR | VaR Loss (%) |
|---|---:|---:|---:|---:|---:|
| Normal | 5.0% | 0.1981 | 0.1981 | 2.1855 | 3.40% |
| Normal | 1.0% | 0.1981 | 0.1981 | 2.1855 | 4.89% |
| Normal | 0.1% | 0.1981 | 0.1981 | 2.1855 | 6.56% |
| Student-t | 5.0% | 0.1533 | 0.1533 | 2.1769 | 3.20% |
| Student-t | 1.0% | 0.1533 | 0.1533 | 2.1769 | 5.57% |
| Student-t | 0.1% | 0.1533 | 0.1533 | 2.1769 | **10.28%** |
| GED | 5.0% | 0.1657 | 0.1657 | 2.1590 | 3.38% |
| GED | 1.0% | 0.1657 | 0.1657 | 2.1590 | 5.61% |
| GED | 0.1% | 0.1657 | 0.1657 | 2.1590 | 8.62% |
| Skewed Student-t | 5.0% | 0.1406 | 0.1406 | 2.1771 | 3.24% |
| Skewed Student-t | 1.0% | 0.1406 | 0.1406 | 2.1771 | 5.66% |
| Skewed Student-t | 0.1% | 0.1406 | 0.1406 | 2.1771 | **10.47%** |

The results highlight the importance of the assumed return distribution, particularly at extreme confidence levels. At 5%, the estimated losses are relatively similar across distributions, but the differences become substantial at the **0.1% tail**. The Student-t and Skewed Student-t specifications produce VaR losses above 10%, reflecting their ability to capture the heavy tails observed in NVDA returns. This supports the use of **heavy-tailed distributions for extreme-risk estimation** rather than relying solely on the Normal assumption.


## Part II — Multivariate Portfolio Risk Analysis

This section analyses a four-asset portfolio consisting of NVIDIA (NVDA), Microsoft (MSFT), Johnson & Johnson (JNJ), and Procter & Gamble (PG). The objective is to model time-varying volatility and correlations, estimate portfolio risk, and evaluate the reliability of out-of-sample VaR forecasts.

### Data Download

These assets were selected to construct a multi-asset portfolio combining technology and defensive stocks. The resulting dataset was used to calculate daily returns, analyze cross-asset dependencies, and estimate portfolio risk using multivariate volatility models.

| **Date** | **NVDA** | **MSFT** | **JNJ** | **PG** |
|---|---:|---:|---:|---:|
| 2025-12-24 | 188.169678 | 484.943420 | 204.490891 | 141.343948 |
| 2025-12-26 | 190.085205 | 484.635406 | 204.343246 | 141.588516 |
| 2025-12-29 | 187.780594 | 484.029236 | 204.274368 | 141.422211 |
| 2025-12-30 | 187.102173 | 484.406860 | 203.634644 | 140.913528 |
| 2025-12-31 | 186.064621 | 480.571167 | 203.674026 | 140.189636 |

The dataset contains **4,024 daily observations**, covering January 2010 to December 2025.

### 1. Exploratory Data Analysis

### Descriptive Statistics and Diagnostic Tests

Daily returns were analyzed to summarize the assets' return and risk characteristics. The analysis includes annualized volatility, skewness, excess kurtosis, and statistical tests for normality and volatility clustering.

| **Ticker** | **Daily Mean Return (%)** | **Annualized Volatility (%)** | **Skewness** | **Excess Kurtosis** | **Jarque-Bera p-value** | **ARCH-LM p-value (10 lags)** | **Ljung-Box p-value on r² (10 lags)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA | 0.1513 | 45.4174 | 0.1722 | 6.8737 | < 0.001 | < 0.001 | < 0.001 |
| MSFT | 0.0756 | 25.4804 | -0.1430 | 8.0702 | < 0.001 | < 0.001 | < 0.001 |
| JNJ | 0.0406 | 16.9165 | -0.3312 | 9.4007 | < 0.001 | < 0.001 | < 0.001 |
| PG | 0.0327 | 17.2070 | -0.1085 | 10.5686 | < 0.001 | < 0.001 | < 0.001 |

NVDA exhibits the highest daily mean return and annualized volatility among the four assets. All four return series display substantial excess kurtosis, indicating heavier tails than a normal distribution. The Jarque-Bera test rejects normality, while the ARCH-LM and Ljung-Box tests provide evidence of conditional heteroskedasticity and persistence in squared returns. These findings support the use of GARCH-family models to capture time-varying volatility.

### Correlation Matrix

The Pearson correlation matrix was computed using daily returns to examine the linear relationships between the four assets. NVDA and MSFT exhibit the strongest correlation (0.57), while NVDA has weaker correlations with JNJ (0.17) and PG (0.19). These differences suggest potential diversification benefits when combining technology stocks with defensive assets.

| **Ticker** | **NVDA** | **MSFT** | **JNJ** | **PG** |
|---|---:|---:|---:|---:|
| **NVDA** | 1.000000 | 0.570172 | 0.171230 | 0.185359 |
| **MSFT** | 0.570172 | 1.000000 | 0.352367 | 0.372037 |
| **JNJ** | 0.171230 | 0.352367 | 1.000000 | 0.529142 |
| **PG** | 0.185359 | 0.372037 | 0.529142 | 1.000000 |

The unconditional return correlation matrix shows that NVDA and MSFT have the strongest pairwise correlation (0.570), followed by JNJ and PG (0.529). NVDA has relatively weak correlations with JNJ (0.171) and PG (0.185).

These relationships suggest potential diversification benefits from combining technology stocks with defensive assets. However, correlations can change over time, motivating the use of a Dynamic Conditional Correlation (DCC-GARCH) model.

## 3. DCC-GARCH (Engle, 2002)

With $z_t = \varepsilon_t / \sigma_t$ (standardized residuals from Step 1):

$$
Q_t = (1-a-b)\,\bar Q + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1},
\qquad
R_t = \mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}
$$

$$
H_t = D_t R_t D_t,
\qquad
D_t = \mathrm{diag}(\sigma_{1,t},\dots,\sigma_{N,t})
$$

The parameters $(a,b)$ are estimated by quasi-maximum likelihood for the correlation component, subject to $a,b\geq 0$ and $a+b<1$:

$$
\mathcal{L}_C =
-\frac{1}{2}\sum_t
\left(
\ln|R_t| +
z_t'R_t^{-1}z_t -
z_t'z_t
\right)
$$

### GJR-Garch Individual

The leverage parameter $\gamma_i$ captures the additional effect of negative shocks on volatility.

| Ticker | $\alpha_1$ | $\gamma_1$ | $\beta_1$ | $\nu$ | Persistence |
|---|---:|---:|---:|---:|---:|
| NVDA | 0.0300 | 0.0719 | 0.9305 | 4.5476 | 0.9965 |
| MSFT | 0.0254 | 0.1269 | 0.8819 | 4.6325 | 0.9707 |
| JNJ | 0.0436 | 0.0945 | 0.8616 | 4.6128 | 0.9524 |
| PG | 0.0185 | 0.1054 | 0.8889 | 4.4816 | 0.9601 |

Volatility persistence is high across all assets, particularly for NVDA, whose persistence estimate is 0.9965. This indicates that volatility shocks decay slowly. The positive leverage parameters suggest that negative return shocks increase volatility more than positive shocks of the same magnitude, conditional on the model specification.

The estimated degrees of freedom are around 4.5 for all four assets, supporting the use of Student-t innovations to accommodate heavy-tailed returns.

#### Residual Diagnostics

| Ticker | Ljung-Box(10) on $z^2$ p-value | ARCH-LM(10) p-value |
|---|---:|---:|
| NVDA | 0.9016 | 0.9050 |
| MSFT | 0.7519 | 0.7313 |
| JNJ | 0.9960 | 0.9953 |
| PG | 0.8302 | 0.8410 |

The high p-values indicate no significant evidence of remaining autocorrelation in squared standardized residuals or residual ARCH effects at the 10% significance level. This suggests that the individual GJR-GARCH models capture much of the conditional volatility structure in the data.

### 3. Dynamic Conditional Correlation (DCC-GARCH)

The standardized residuals from the individual volatility models are used to estimate a DCC(1,1) model:

$$
Q_t=(1-a-b)\bar Q
+a z_{t-1}z_{t-1}^{\top}+bQ_{t-1}
$$

The matrix $Q_t$ is normalized to obtain the time-varying conditional correlation matrix $R_t$.

The estimated parameters are:

- **DCC parameter ($a$):** 0.0144
- **DCC parameter ($b$):** 0.9769
- **Persistence ($a+b$):** 0.9913

The high persistence indicates that conditional correlations adjust gradually and that the impact of shocks on cross-asset dependence can persist over time.

#### Unconditional Q Matrix

The estimated unconditional matrix $\bar Q$ provides the long-run dependence reference used in the DCC recursion. It is not itself a correlation matrix, since its diagonal elements are not normalized to one.

| Ticker | NVDA | MSFT | JNJ | PG |
|---|---:|---:|---:|---:|
| NVDA | 1.048 | 0.498 | 0.112 | 0.122 |
| MSFT | 0.498 | 1.018 | 0.256 | 0.270 |
| JNJ | 0.112 | 0.256 | 1.024 | 0.432 |
| PG | 0.122 | 0.270 | 0.432 | 0.994 |

The largest off-diagonal element is between NVDA and MSFT, while NVDA has weaker dependence with JNJ and PG. This is broadly consistent with the unconditional return correlations, although the matrices are not directly equivalent because they are based on different quantities.

### 4. Portfolio Volatility and Risk Contributions

Assuming equal portfolio weights of 25% per asset, the conditional covariance matrix is obtained from:

$$
H_t=D_tR_tD_t
$$

Portfolio conditional volatility is then calculated as:

$$
\sigma_{p,t}=\sqrt{w^\top H_tw}
$$

At the final sample date, **31 December 2025**, the estimated portfolio daily volatility is **0.786%**, equivalent to approximately **12.48% annualized volatility**, assuming 252 trading days per year.

| Ticker | Weight | Conditional Daily Volatility (%) | Risk Contribution | Relative Risk Contribution |
|---|---:|---:|---:|---:|
| NVDA | 25% | 2.2572 | 0.4381 | 55.72% |
| MSFT | 25% | 1.1616 | 0.1965 | 25.00% |
| JNJ | 25% | 0.9680 | 0.0721 | 9.17% |
| PG | 25% | 1.0980 | 0.0795 | 10.11% |

Despite representing only one quarter of the portfolio, NVDA contributes approximately **55.7% of total portfolio risk**. MSFT contributes another 25%, while JNJ and PG together account for approximately 19.3%.

This illustrates that equal capital allocation does not imply equal risk allocation. The portfolio's risk is concentrated in the two technology stocks, particularly NVDA.

### 5. Out-of-Sample Parametric Value at Risk

The portfolio VaR is estimated out of sample using the conditional volatility forecasts generated by the DCC-GARCH framework. Under a conditional normal approximation, VaR measures the loss threshold expected to be exceeded with probability $1-c$ at confidence level $c$.

| Date | VaR 95% (%) | VaR 99% (%) |
|---|---:|---:|
| 2025-12-24 | 1.669 | 3.136 |
| 2025-12-26 | 1.616 | 3.036 |
| 2025-12-29 | 1.563 | 2.937 |
| 2025-12-30 | 1.527 | 2.869 |
| 2025-12-31 | 1.485 | 2.789 |

The VaR estimates decline over these final out-of-sample observations, indicating that the model forecasts lower conditional portfolio risk during this period. The 99% VaR is higher than the 95% VaR, as it corresponds to a more extreme loss threshold.

These values describe model-based risk forecasts rather than maximum possible losses. Their reliability must be assessed by comparing the forecasts against realized portfolio returns.










### Value-at-Risk Backtesting

#### Methodology

Value-at-Risk (VaR) estimates the loss threshold expected to be exceeded with probability \(1-c\), where \(c\) is the confidence level.

$$
\Pr(R_{p,t} < -\text{VaR}_{c,t}) = 1-c
$$

Under the conditional Gaussian assumption:

$$
\text{VaR}_{c,t}
= -\mu_{p,t}-\sigma_{p,t}\Phi^{-1}(1-c)
$$

where \(\mu_{p,t}\) and \(\sigma_{p,t}\) are the conditional portfolio return mean and volatility, respectively.

#### Kupiec Unconditional Coverage Test

Let \(T\) denote the number of out-of-sample observations, \(N\) the number of exceptions, \(p=1-c\) the expected exception probability, and \(\hat{p}=N/T\) the observed exception rate.

The likelihood-ratio statistic is

$$
LR_{\text{uc}}
=-2\ln\left[
\frac{(1-p)^{T-N}p^N}
{(1-\hat{p})^{T-N}\hat{p}^{N}}
\right]
$$

Under the null hypothesis of correct unconditional coverage, this statistic asymptotically follows a chi-squared distribution with one degree of freedom.

#### Christoffersen Conditional Coverage Test

The Christoffersen test evaluates both the frequency and the temporal independence of VaR exceptions. Its conditional coverage statistic combines the Kupiec statistic with the independence statistic:

$$
LR_{\text{cc}}=LR_{\text{uc}}+LR_{\text{ind}}
$$

Under the null hypothesis, \(LR_{\text{cc}}\) asymptotically follows a chi-squared distribution with two degrees of freedom.

For both tests, a p-value below 0.05 leads to rejection of the corresponding null hypothesis.

#### Results

| Model | Confidence Level | Exceptions | Expected | Observed Rate | Kupiec p-value | Christoffersen Independence p-value | Conditional Coverage p-value |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 95% | 134 | 125.7 | 5.33% | 0.4522 | 0.2743 | 0.4147 |
| Normal | 99% | 40 | 25.1 | 1.59% | 0.0061 | 0.6687 | 0.0212 |
| Normal | 99.9% | 12 | 2.5 | 0.48% | 0.0000 | 0.7344 | 0.0001 |
| Student-t | 95% | 157 | 125.7 | 6.25% | 0.0057 | 0.9311 | 0.0220 |
| Student-t | 99% | 30 | 25.1 | 1.19% | 0.3445 | 0.3945 | 0.4452 |
| Student-t | 99.9% | 3 | 2.5 | 0.12% | 0.7661 | 0.9325 | 0.9533 |
| FHS | 95% | 133 | 125.7 | 5.29% | 0.5079 | 0.2551 | 0.4203 |
| FHS | 99% | 30 | 25.1 | 1.19% | 0.3445 | 0.3945 | 0.4452 |
| FHS | 99.9% | 4 | 2.5 | 0.16% | 0.3883 | 0.9101 | 0.6849 |

*Note: P-values displayed as 0.0000 are rounded and should not be interpreted as exactly zero.*

#### Interpretation

**Normal distribution.** The model provides acceptable coverage at the 95% confidence level. However, it significantly underestimates risk at the 99% and 99.9% levels, where observed exceptions exceed the expected frequency. This indicates that the Gaussian assumption does not adequately capture extreme portfolio losses.

**Student-t distribution.** The model fails the coverage tests at the 95% level but performs adequately at the 99% and 99.9% levels. The heavier-tailed distribution therefore provides a more suitable parametric representation of extreme losses, although its performance varies across confidence levels.

**Filtered Historical Simulation (FHS).** FHS passes the coverage and conditional coverage tests at all three confidence levels. By combining volatility filtering with the empirical distribution of standardized residuals, it avoids imposing a specific parametric distribution on the innovations.

The independence tests do not reject their null hypotheses for any model, providing insufficient evidence that exceptions cluster over time. However, failure to reject a hypothesis does not prove that a model is correctly specified.

#### Conclusion

FHS exhibits the most consistent out-of-sample VaR performance across the confidence levels considered. Student-t provides a competitive parametric alternative at higher confidence levels, while the Normal model struggles to capture extreme losses.

These results should be interpreted cautiously, particularly at the 99.9% level, where the expected number of exceptions is small and statistical power is limited. The identical results for Student-t and FHS at the 99% level should also be verified to ensure that each model uses its own VaR forecasts. Further comparison using Expected Shortfall (ES) would help assess the magnitude of losses beyond the VaR threshold.




























