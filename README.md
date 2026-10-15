# Portfolio Risk Modelling: GJR-GARCH Volatility, DCC Correlations, VaR & Monte Carlo

## Summary

This project builds a market-risk workflow in Python that goes from a single stock to a multi-asset equity portfolio. Part I studies **NVDA** on its own: it documents fat tails, volatility clustering and the leverage effect, selects an innovation distribution for a GJR-GARCH(1,1) model, and shows how that choice changes extreme one-day VaR. Part II extends the analysis to an equally weighted portfolio of **NVDA, MSFT, JNJ and PG**. Each asset gets its own GJR-GARCH(1,1)-t model, a **DCC-GARCH model implemented from scratch** adds time-varying correlations, and the resulting covariance matrix is used to decompose portfolio risk, produce out-of-sample VaR forecasts under three methods (Normal, Student-t, Filtered Historical Simulation), backtest them with Kupiec and Christoffersen tests, and simulate future portfolio losses with Monte Carlo.

The main findings are that equal capital weights hide a very uneven risk allocation (NVDA is 25% of capital but ~54% of risk), and that Normal VaR clearly fails in the tails (40 breaches vs 25 expected at 99%, and 12 vs 2.5 at 99.9%) while Student-t and FHS remain well calibrated.

## Repository Structure

Two Jupyter notebooks build the workflow, moving from a **single stock (NVDA)** to a **four-asset equity portfolio (NVDA, MSFT, JNJ, PG)**. Daily data runs from January 2010 to December 2025 (4,024 observations).

| Notebook | Scope | Core question |
|---|---|---|
| `Modelling_1.ipynb` | Univariate | How volatile is NVDA, how does that volatility behave over time, and how large can a one-day loss be? |
| `Modelling2_multivariate.ipynb` | Multivariate | How do correlations and volatilities interact at portfolio level, and how reliable is the resulting VaR? |

## Key Insights

### Part I: Univariate Risk (NVDA)

- **Fat tails and non-normality:** daily log-returns range from about -20.8% to +26.1%, and a mixture of three Gaussians fits the empirical density better than a single Normal.
- **Volatility clustering:** log-returns show almost no autocorrelation, while squared log-returns are strongly and significantly autocorrelated, which motivates GARCH-type models.
- **Leverage effect:** the GJR-GARCH(1,1) estimates a significant asymmetry parameter (**γ ≈ 0.085, p = 0.029**), so negative shocks raise future volatility more than positive ones. Persistence is high (**≈ 0.966**).
- **Innovation distribution matters:** Student-t innovations give the lowest AIC/BIC (**AIC ≈ 18,719 vs 19,327 for Normal**).
- **Extreme VaR is highly model-dependent:** the 1-day 99.9% VaR is **≈ 10.3% under Student-t vs ≈ 6.6% under Normal**, while at the 95% level all four distributions give similar values (3.2%-3.4%).

### Part II: Multivariate Risk (NVDA, MSFT, JNJ, PG)

- **Heterogeneous assets:** annualised volatility ranges from **≈ 17% (JNJ, PG) to ≈ 45% (NVDA)**, excess kurtosis from **6.9 to 10.6**, and the strongest pairwise correlation is NVDA-MSFT (0.57).
- **Univariate models are well specified:** GJR-GARCH(1,1)-t removes the remaining ARCH effects in all four series (residual diagnostic p-values **> 0.88**). All leverage parameters are positive (0.069-0.085) and persistence lies between **0.955 and 0.966**.
- **Persistent correlations:** DCC-GARCH gives **a ≈ 0.014 and b ≈ 0.977 (a + b ≈ 0.991)**, so conditional correlations adapt slowly and shocks to dependence last a long time.
- **Risk concentration:** on 31 Dec 2025 the portfolio has a daily volatility of **0.806% (≈ 12.8% annualised)**. NVDA holds **25% of capital but contributes ≈ 54% of risk**, and NVDA plus MSFT together contribute ≈ 81%.
- **VaR backtesting (2,514 out-of-sample days):** the Normal model is rejected at 99% (40 breaches vs 25.1 expected, Kupiec p = 0.006) and at 99.9% (12 vs 2.5, p < 0.001). **FHS passes every test at all three confidence levels**, and Student-t passes at 99% and 99.9% but is rejected at 95% (157 breaches vs 125.7 expected).
- **Forward-looking risk:** Monte Carlo simulation using the final conditional covariance state generates one-day and multi-day portfolio loss distributions, from which VaR and Expected Shortfall are computed.

## Technical Stack

| Area | Tools |
|---|---|
| Language / environment | Python 3, Jupyter Notebook |
| Data acquisition | `yfinance` |
| Data handling | `pandas`, `numpy` |
| Statistical modelling | `arch`, `statsmodels`, `scipy` |
| Machine learning / distributions | `scikit-learn` (`GaussianMixture`) |
| Optimisation | `scipy.optimize` (constrained QMLE for DCC) |
| Visualisation | `matplotlib`, `seaborn` |

### Main Analytical Methods

- Log-returns, rolling volatility and distribution analysis (KDE, Gaussian mixtures)
- ACF, Ljung-Box and ARCH-LM diagnostics
- GJR-GARCH with AIC/BIC model selection
- Normal, Student-t, GED and Skewed Student-t innovation distributions
- **DCC-GARCH implemented from scratch** (constrained QMLE)
- Parametric (Normal, Student-t) and Filtered Historical Simulation VaR
- **Kupiec** and **Christoffersen** VaR backtesting
- Monte Carlo simulation with Cholesky-correlated shocks, one-day and multi-day

---

## Part I: Univariate Risk (NVDA)

### Data

Historical daily prices for NVIDIA (NVDA) were downloaded with `yfinance` for January 2010 to December 2025: adjusted close, high, low, open and volume. The data were used to compute daily log-returns and to analyse the evolution of price and volatility.

| Date | Close | High | Low | Open | Volume |
|---|---:|---:|---:|---:|---:|
| 2025-12-24 | 188.169678 | 188.468980 | 186.154389 | 187.501244 | 65,528,500 |
| 2025-12-26 | 190.085205 | 192.240166 | 187.561113 | 189.476629 | 139,740,300 |
| 2025-12-29 | 187.780594 | 188.319327 | 185.475989 | 187.271790 | 120,006,100 |
| 2025-12-30 | 187.102188 | 188.548815 | 186.493612 | 187.800566 | 97,687,300 |
| 2025-12-31 | 186.064606 | 190.115125 | 186.054635 | 189.127446 | 120,100,500 |

![NVDA historical price](images/price_nvda.png)

### Returns and Log-Returns

Simple returns and log-returns were computed from adjusted closing prices to characterise NVIDIA's daily performance and its extreme movements.

| Statistic | Price | Return | Log Return |
|---|---:|---:|---:|
| Mean | 22.61 | 0.00193 | 0.00151 |
| Std. Dev. | 43.89 | 0.02879 | 0.02861 |
| Minimum | 0.20 | -0.18756 | -0.20771 |
| Median | 3.97 | 0.00178 | 0.00174 |
| Maximum | 206.55 | 0.29807 | 0.26088 |

Daily log-returns have a mean of about **0.15%** and a standard deviation of **2.86%**, with extreme daily moves of about **-20.8%** and **+26.1%**.

![NVDA returns and log-returns](images/returns_nvda.png)

### Return Distribution (KDE)

Kernel Density Estimation was used to visualise the empirical distribution of log-returns, which was compared with a Normal distribution and Gaussian Mixture Models (GMMs) with two and three components.

![NVDA log-returns KDE](images/kde_nvda.png)

Log-returns are concentrated around zero, with a pronounced peak and heavy tails. A single Gaussian does not capture this shape, whereas the three-component mixture fits the kernel density particularly well. This suggests heterogeneous, non-normal behaviour that a mixture distribution represents better.

### Autocorrelation Analysis

The ACF was computed for log-returns and for squared log-returns. The former checks serial dependence in returns, the latter detects volatility clustering.

![ACF of log-returns and squared log-returns](images/acf_nvda.png)

Log-returns show little significant autocorrelation, so they are approximately uncorrelated over time. Squared log-returns, in contrast, show significant positive autocorrelation: periods of high (or low) volatility tend to be followed by similar periods. This is a standard feature of financial returns and motivates GARCH-type models.

### GJR-GARCH(1,1) Model

A GJR-GARCH(1,1) model was fitted to the log-returns to capture volatility clustering and asymmetric responses to shocks. Returns are modelled as

$$
r_t = \mu + \varepsilon_t,
\qquad
\varepsilon_t = \sigma_t z_t,
\qquad
z_t \sim \mathcal{N}(0,1)
$$

and the conditional variance follows

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

- $\omega > 0$ is the variance intercept.
- $\alpha$ captures the impact of past shocks on current volatility.
- $\gamma$ captures the **asymmetric effect** of negative shocks.
- $\beta$ measures **volatility persistence**.
- $\mathbb{1}_{\{\varepsilon_{t-1}<0\}}$ equals 1 when the previous shock is negative and 0 otherwise.

A negative shock has an impact of $(\alpha+\gamma)$ and a positive shock an impact of $\alpha$, so a significantly positive $\gamma$ means negative returns raise volatility more than positive returns of the same size.

#### Results

The model was estimated on 4,024 daily observations with Normal innovations.

| Parameter | Estimate | p-value |
|---|---:|---:|
| $\mu$ | 0.1725 | < 0.001 |
| $\omega$ | 0.3206 | 0.067 |
| $\alpha$ | 0.0537 | 0.061 |
| $\gamma$ | 0.0852 | 0.029 |
| $\beta$ | 0.8701 | < 0.001 |

| Log-Likelihood | AIC | BIC |
|---:|---:|---:|
| -9,644.24 | 19,298.5 | 19,330.0 |

#### Interpretation

The estimated $\gamma = 0.0852$ is significant at the 5% level, which supports an **asymmetric volatility response**. Parameters $\alpha$ and $\omega$ are only significant at the 10% level. Under symmetric innovations, persistence can be approximated as

$$
P \approx \alpha + \beta + \frac{\gamma}{2}
\approx 0.0537 + 0.8701 + \frac{0.0852}{2}
\approx 0.9664
$$

so volatility shocks in NVDA decay slowly, in line with strong volatility clustering.

### Conditional Volatility

The fitted model produces a time-varying conditional volatility series that tracks periods of high and low market uncertainty.

![NVDA GJR-GARCH conditional volatility](images/vol_garch.png)

### Innovation Distribution Comparison

To account for the heavy tails in the standardised residuals, the model was estimated under four innovation distributions.

![NVDA standardised residuals by innovation distribution](images/residuals_distr.png)

| Distribution | Log-Likelihood | AIC | BIC |
|---|---:|---:|---:|
| **Student-t** | **-9,354.30** | **18,718.61** | **18,750.11** |
| Skewed Student-t | -9,353.98 | 18,719.97 | 18,757.77 |
| GED | -9,399.78 | 18,809.56 | 18,841.06 |
| Normal | -9,659.30 | 19,326.60 | 19,351.80 |

<!-- TODO: the Normal row here (AIC 19,326.60) does not match the baseline Normal fit above (AIC 19,298.5). Check that both use the same mean specification, or add a note explaining the difference. -->

The heavy-tailed specifications fit substantially better than the Normal. **Student-t is selected** as the preferred model because it has the lowest AIC and BIC, while the Skewed Student-t reaches almost the same log-likelihood at the cost of an extra parameter. This supports heavy-tailed innovations for the risk analysis that follows.

### Value at Risk (VaR)

One-day VaR was computed at the 95%, 99% and 99.9% confidence levels (5%, 1% and 0.1% tail probabilities) under each innovation distribution, to measure how the distributional assumption affects potential losses.

<!-- TODO: the original table had "Expected Return" and "Volatility" columns with identical values and a "VaR" column that did not change with the confidence level. Check the notebook output; those columns were removed here. -->

| Distribution | VaR 95% | VaR 99% | VaR 99.9% |
|---|---:|---:|---:|
| Normal | 3.40% | 4.89% | 6.56% |
| Student-t | 3.20% | 5.57% | **10.28%** |
| GED | 3.38% | 5.61% | 8.62% |
| Skewed Student-t | 3.24% | 5.66% | **10.47%** |

At 95%, losses are similar across distributions (3.2%-3.4%), but the differences grow sharply in the tail. At 99.9%, Student-t and Skewed Student-t give VaR above 10%, against 6.6% under Normal. Extreme-risk estimates should therefore rely on **heavy-tailed distributions** rather than on the Normal assumption.

---

## Part II: Multivariate Portfolio Risk

This part analyses an equally weighted (25% each) portfolio of NVIDIA (NVDA), Microsoft (MSFT), Johnson & Johnson (JNJ) and Procter & Gamble (PG). The goal is to model time-varying volatilities and correlations, estimate portfolio risk and evaluate the reliability of out-of-sample VaR forecasts.

### Data

The assets combine technology and defensive stocks. The dataset contains **4,024 daily observations** from January 2010 to December 2025, used to compute returns, study cross-asset dependence and estimate portfolio risk.

| **Date** | **NVDA** | **MSFT** | **JNJ** | **PG** |
|---|---:|---:|---:|---:|
| 2025-12-24 | 188.169678 | 484.943420 | 204.490891 | 141.343948 |
| 2025-12-26 | 190.085205 | 484.635406 | 204.343246 | 141.588516 |
| 2025-12-29 | 187.780594 | 484.029236 | 204.274368 | 141.422211 |
| 2025-12-30 | 187.102173 | 484.406860 | 203.634644 | 140.913528 |
| 2025-12-31 | 186.064621 | 480.571167 | 203.674026 | 140.189636 |

![Prices of the four portfolio assets](images/mult_vol.png)

### Exploratory Data Analysis

#### Descriptive Statistics and Diagnostic Tests

| **Ticker** | **Daily Mean Return (%)** | **Annualised Volatility (%)** | **Skewness** | **Excess Kurtosis** | **Jarque-Bera p-value** | **ARCH-LM p-value (10 lags)** | **Ljung-Box p-value on r² (10 lags)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA | 0.1513 | 45.4174 | 0.1722 | 6.8737 | < 0.001 | < 0.001 | < 0.001 |
| MSFT | 0.0756 | 25.4804 | -0.1430 | 8.0702 | < 0.001 | < 0.001 | < 0.001 |
| JNJ | 0.0406 | 16.9165 | -0.3312 | 9.4007 | < 0.001 | < 0.001 | < 0.001 |
| PG | 0.0327 | 17.2070 | -0.1085 | 10.5686 | < 0.001 | < 0.001 | < 0.001 |

NVDA has the highest mean return and volatility. All four series show substantial excess kurtosis, the Jarque-Bera test rejects normality, and the ARCH-LM and Ljung-Box tests show conditional heteroskedasticity and persistence in squared returns. These results support GARCH-family models.

#### Correlation Matrix

| **Ticker** | **NVDA** | **MSFT** | **JNJ** | **PG** |
|---|---:|---:|---:|---:|
| **NVDA** | 1.000000 | 0.570172 | 0.171230 | 0.185359 |
| **MSFT** | 0.570172 | 1.000000 | 0.352367 | 0.372037 |
| **JNJ** | 0.171230 | 0.352367 | 1.000000 | 0.529142 |
| **PG** | 0.185359 | 0.372037 | 0.529142 | 1.000000 |

NVDA and MSFT have the strongest unconditional correlation (0.57), followed by JNJ and PG (0.53). NVDA is only weakly correlated with JNJ (0.17) and PG (0.19), which suggests diversification benefits from mixing technology and defensive stocks. Since correlations change over time, a Dynamic Conditional Correlation model is used.

### Step 1: Univariate GJR-GARCH(1,1)-t Models

Each asset is first modelled with its own GJR-GARCH(1,1) with Student-t innovations. The leverage parameter $\gamma_i$ captures the extra effect of negative shocks on volatility.

| **Asset** | **mu** | **omega** | **alpha[1]** | **gamma[1]** | **beta[1]** | **Persistence** |
|----------|-------:|----------:|-------------:|-------------:|-----------:|----------------:|
| NVDA     | 0.1726 | 0.3201    | 0.0537       | 0.0850       | 0.8703     | 0.9664          |
| MSFT     | 0.0732 | 0.0863    | 0.0145       | 0.0820       | 0.9092     | 0.9647          |
| JNJ      | 0.0372 | 0.0479    | 0.0332       | 0.0690       | 0.8879     | 0.9556          |
| PG       | 0.0252 | 0.0478    | 0.0250       | 0.0824       | 0.8886     | 0.9548          |

Volatility persistence is high for all assets (0.955-0.966), so volatility shocks decay slowly. The leverage parameters are positive for every asset, meaning negative shocks raise volatility more than positive shocks of the same size. The estimated degrees of freedom are around 4.5 for all four assets, which supports Student-t innovations.

#### Residual Diagnostics

| **Asset** | **Ljung-Box(10) z² p-value** | **ARCH-LM(10) p-value** |
|----------|----------------------------:|------------------------:|
| NVDA     | 0.9201                      | 0.9188                  |
| MSFT     | 0.9481                      | 0.9514                  |
| JNJ      | 0.9998                      | 0.9998                  |
| PG       | 0.8884                      | 0.8975                  |

The high p-values show no evidence of remaining autocorrelation in squared standardised residuals or of residual ARCH effects, so the univariate models capture most of the conditional volatility structure.

### Step 2: DCC-GARCH (Engle, 2002)

The standardised residuals $z_t = \varepsilon_t / \sigma_t$ from Step 1 are used to estimate a DCC(1,1) model:

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

The estimated parameters are:

- **DCC parameter $a$:** 0.0143
- **DCC parameter $b$:** 0.9766
- **Persistence $a+b$:** 0.9909

The high persistence means conditional correlations adjust gradually and the effect of a shock on cross-asset dependence lasts a long time.

#### Unconditional Matrix $\bar Q$

$\bar Q$ is the long-run reference matrix of the DCC recursion. It is the unconditional covariance of the standardised residuals, so its diagonal is close to, but not exactly, one.

| **Asset** | **NVDA** | **MSFT** | **JNJ** | **PG** |
|----------|---------:|---------:|--------:|-------:|
| NVDA     | 0.999    | 0.496    | 0.110   | 0.123  |
| MSFT     | 0.496    | 1.000    | 0.260   | 0.278  |
| JNJ      | 0.110    | 0.260    | 1.000   | 0.435  |
| PG       | 0.123    | 0.278    | 0.435   | 1.000  |

The largest off-diagonal element is NVDA-MSFT, while NVDA has weaker dependence with JNJ and PG. This is broadly consistent with the raw return correlations, although the two matrices are not directly comparable because they are built from different quantities (standardised residuals instead of raw returns).

#### Dynamic Conditional Correlations

![Dynamic conditional correlations from DCC-GARCH](images/dcc_corr.png)

### Portfolio Volatility and Risk Contributions

With equal weights of 25% per asset, the conditional covariance matrix is $H_t = D_t R_t D_t$ and portfolio conditional volatility is

$$
\sigma_{p,t}=\sqrt{w^\top H_tw}
$$

On the last sample date, **31 December 2025**, the portfolio daily volatility is **0.806%**, or about **12.79% annualised** (252 trading days).

| **Asset** | **Weight** | **Conditional Daily Volatility (%)** | **Risk Contribution (daily volatility points)** | **Relative Risk Contribution** |
|----------|----------:|------------------------------------:|----------------------------------------------:|-------------------------------:|
| NVDA     | 0.25      | 2.2420                              | 0.4337                                        | 0.5383                         |
| MSFT     | 0.25      | 1.2714                              | 0.2230                                        | 0.2767                         |
| JNJ      | 0.25      | 0.9882                              | 0.0734                                        | 0.0911                         |
| PG       | 0.25      | 1.0724                              | 0.0756                                        | 0.0939                         |

NVDA is only a quarter of the portfolio but contributes about **53.8% of total risk**. MSFT adds 27.7%, and JNJ and PG together account for 18.5%. Equal capital allocation therefore does not mean equal risk allocation: portfolio risk is concentrated in the two technology stocks, above all NVDA.

### Out-of-Sample VaR Forecasts

The table shows DCC-GARCH one-day VaR forecasts (in % of portfolio value) for the last five trading days of 2025 under three approaches: Normal (baseline), Student-t, and Filtered Historical Simulation (FHS), at 95%, 99% and 99.9%. The model is re-estimated using only observations available before each forecasting block, which avoids look-ahead bias.

| **Date** | **VaR 95%** | **VaR 95% (t)** | **VaR 95% (FHS)** | **VaR 99%** | **VaR 99% (t)** | **VaR 99% (FHS)** | **VaR 99.9%** | **VaR 99.9% (t)** | **VaR 99.9% (FHS)** |
|----------|------------:|----------------:|------------------:|------------:|----------------:|------------------:|--------------:|------------------:|--------------------:|
| 2025-12-24 | 1.398 | 1.320 | 1.391 | 2.009 | 2.204 | 2.203 | 2.693 | 3.766 | 3.612 |
| 2025-12-26 | 1.354 | 1.279 | 1.348 | 1.947 | 2.137 | 2.136 | 2.611 | 3.651 | 3.502 |
| 2025-12-29 | 1.311 | 1.239 | 1.305 | 1.887 | 2.071 | 2.070 | 2.532 | 3.541 | 3.397 |
| 2025-12-30 | 1.290 | 1.219 | 1.285 | 1.857 | 2.039 | 2.038 | 2.492 | 3.486 | 3.344 |
| 2025-12-31 | 1.256 | 1.187 | 1.251 | 1.809 | 1.986 | 1.985 | 2.428 | 3.398 | 3.259 |

VaR forecasts fall steadily over the period, which means the model sees portfolio risk declining. As expected, VaR increases with the confidence level. At 95% the Normal and FHS forecasts are almost identical and the Student-t forecast is slightly lower, while at 99% and 99.9% the Student-t and FHS forecasts are clearly higher than the Normal one, reflecting their greater sensitivity to tail risk. These are model-based risk forecasts, not maximum possible losses, and five days are not enough to judge accuracy, so the next section compares them with realised returns over the full out-of-sample period.

### Out-of-Sample VaR Backtesting

The three approaches were backtested on **2,514 out-of-sample observations**. The expected number of breaches is the nominal breach rate times 2,514.

| **Model** | **Confidence Level** | **Expected Breach Rate** | **Observed Breach Rate** | **Number of Breaches** | **Expected Breaches** |
|-----------|---------------------:|-------------------------:|-------------------------:|-----------------------:|----------------------:|
| Normal    | 95%                  | 0.050                    | 0.0533                   | 134                    | 125.7                 |
| Normal    | 99%                  | 0.010                    | 0.0159                   | 40                     | 25.1                  |
| Normal    | 99.9%                | 0.001                    | 0.0048                   | 12                     | 2.5                   |
| Student-t | 95%                  | 0.050                    | 0.0625                   | 157                    | 125.7                 |
| Student-t | 99%                  | 0.010                    | 0.0119                   | 30                     | 25.1                  |
| Student-t | 99.9%                | 0.001                    | 0.0012                   | 3                      | 2.5                   |
| FHS       | 95%                  | 0.050                    | 0.0529                   | 133                    | 125.7                 |
| FHS       | 99%                  | 0.010                    | 0.0119                   | 30                     | 25.1                  |
| FHS       | 99.9%                | 0.001                    | 0.0016                   | 4                      | 2.5                   |

All three approaches record more breaches than expected at 95%, meaning VaR is slightly too low, and FHS is the closest to the target (133 vs 125.7). At 99% the Normal model is the worst (40 breaches vs 25.1), while Student-t and FHS both record 30. The gap is largest at 99.9%: the Normal model has 12 breaches against 2.5 expected, a clear underestimation of extreme tail risk, whereas Student-t (3) and FHS (4) are close to the expected value.

![Out-of-sample VaR backtest](images/backtest.png)

#### Statistical Tests

**Kupiec unconditional coverage test.** Let $T$ be the number of out-of-sample observations, $N$ the number of exceptions, $p=1-c$ the expected exception probability and $\hat{p}=N/T$ the observed exception rate. The likelihood-ratio statistic is

$$
LR_{\text{uc}}
=-2\ln\left[
\frac{(1-p)^{T-N}p^N}
{(1-\hat{p})^{T-N}\hat{p}^{N}}
\right]
$$

Under the null of correct unconditional coverage it follows asymptotically a $\chi^2$ distribution with one degree of freedom.

**Christoffersen conditional coverage test.** It checks both the frequency and the temporal independence of exceptions, combining the Kupiec statistic with an independence statistic:

$$
LR_{\text{cc}}=LR_{\text{uc}}+LR_{\text{ind}}
$$

Under the null, $LR_{\text{cc}}$ follows asymptotically a $\chi^2$ distribution with two degrees of freedom. For both tests, a p-value below 0.05 rejects the null.

#### Results

| **Model** | **Confidence Level** | **Exceptions** | **Expected** | **Observed Rate** | **Kupiec p (coverage)** | **Christoffersen p (independence)** | **Christoffersen p (cond. coverage)** |
|-----------|---------------------:|---------------:|-------------:|------------------:|------------------------:|------------------------------------:|-------------------------------------:|
| Normal    | 95%                  | 134            | 125.7        | 0.0533            | 0.4522                  | 0.2743                              | 0.4147                               |
| Normal    | 99%                  | 40             | 25.1         | 0.0159            | 0.0061                  | 0.6687                              | 0.0212                               |
| Normal    | 99.9%                | 12             | 2.5          | 0.0048            | 0.0000                  | 0.7344                              | 0.0001                               |
| Student-t | 95%                  | 157            | 125.7        | 0.0625            | 0.0057                  | 0.9311                              | 0.0220                               |
| Student-t | 99%                  | 30             | 25.1         | 0.0119            | 0.3445                  | 0.3945                              | 0.4452                               |
| Student-t | 99.9%                | 3              | 2.5          | 0.0012            | 0.7661                  | 0.9325                              | 0.9533                               |
| FHS       | 95%                  | 133            | 125.7        | 0.0529            | 0.5079                  | 0.2551                              | 0.4203                               |
| FHS       | 99%                  | 30             | 25.1         | 0.0119            | 0.3445                  | 0.3945                              | 0.4452                               |
| FHS       | 99.9%                | 4              | 2.5          | 0.0016            | 0.3883                  | 0.9101                              | 0.6849                               |

At 95%, the Normal and FHS models pass all three tests, while Student-t is rejected by the Kupiec and conditional coverage tests because it has too many exceptions. At 99%, the Normal model fails coverage and conditional coverage, while Student-t and FHS pass all tests. At 99.9%, the Normal model is strongly rejected, while Student-t and FHS pass. In no case is the independence test rejected, so the problems of the Normal model are about how often breaches occur, not about breaches clustering in time.

#### Conclusion

FHS gives the most consistent out-of-sample VaR across the three confidence levels. Student-t is a competitive parametric alternative at 99% and 99.9% but is too aggressive at 95%, and the Normal model fails to capture extreme losses. These conclusions should be read with caution at 99.9%, where only about 2.5 exceptions are expected and the tests have limited power. The identical exception counts and nearly identical forecasts of Student-t and FHS at 99% should also be verified to make sure each method uses its own VaR series. A comparison based on Expected Shortfall would help assess the size of losses beyond VaR.

### Monte Carlo Simulation

#### One-Day Portfolio Returns

Monte Carlo simulation estimates the distribution of next-day portfolio returns by generating 50,000 (`N_SIM`) correlated scenarios from the fitted GJR-GARCH and DCC models. It incorporates time-varying volatilities, asymmetric responses to negative shocks and dynamic correlations. Standardised shocks come from either a multivariate Normal or a Student-t distribution.

The simulation has four steps:

1. **Forecast conditional volatility.** The one-step-ahead variance of each asset comes from the fitted GJR-GARCH(1,1):

   $$\sigma_{i,T+1}^{2}=\omega_i+\alpha_i\varepsilon_{i,T}^{2}+\gamma_i\varepsilon_{i,T}^{2}I(\varepsilon_{i,T}<0)+\beta_i\sigma_{i,T}^{2}$$

2. **Forecast conditional correlations.** The DCC model gives the next-day correlation matrix $R_{T+1}$, which is combined with the forecast volatilities into the covariance matrix:

   $$H_{T+1}=D_{T+1}R_{T+1}D_{T+1}$$

3. **Generate correlated scenarios.** A Cholesky decomposition of $R_{T+1}$ introduces dependence between simulated asset returns. Shocks follow a Normal or a Student-t distribution; in the Student-t case a common mixing factor preserves the dependence structure and the shocks are scaled to unit variance.

4. **Compute portfolio returns.** Simulated log-returns are converted into simple returns before applying the weights (returns in %):

   $$R_{p,T+1}^{(s)}=\sum_{i=1}^{N}w_i\left[\exp\left(\frac{r_{i,T+1}^{(s)}}{100}\right)-1\right]\times100$$

The simulated distribution approximates the one-day-ahead portfolio return distribution and is used to estimate expected return, volatility, downside quantiles, VaR and Expected Shortfall (ES).

![Simulated one-day portfolio return distributions](images/mont_hist.png)

Comparing Normal and Student-t simulations shows the effect of the distributional assumption on extreme losses: Student-t allows heavier tails and therefore more severe downside scenarios, although the size of the difference depends on the estimated degrees of freedom and on the covariance structure.

<!-- TODO: add the numerical results (VaR/ES for Normal and Student-t, one-day) from the notebook. -->

**Limitation.** The simulation uses full-sample parameters and the final observed volatility and correlation states, so it is a conditional one-day scenario analysis, not an out-of-sample backtest. A fair assessment of predictive performance requires parameters and states estimated only with data available before each forecast date.

#### Multi-Day Portfolio Simulation

The multi-day simulation generates portfolio value paths over an investment horizon. Unlike a simulation with a constant covariance matrix, volatilities and correlations are updated at every step using the fitted GJR-GARCH and DCC models, so the paths reflect volatility persistence, leverage effects and changing dependence.

1. **Initialise.** Each path starts from the latest observed residuals, conditional variances, standardised residuals and DCC state, with the same initial portfolio value.

2. **Update volatilities.** At each simulated day the GJR-GARCH recursion is applied to every asset:

   $$\sigma_{i,t}^{2}=\omega_i+\alpha_i\varepsilon_{i,t-1}^{2}+\gamma_i\varepsilon_{i,t-1}^{2}I(\varepsilon_{i,t-1}<0)+\beta_i\sigma_{i,t-1}^{2}$$

3. **Update correlations and simulate returns.** The DCC recursion updates the correlation-generating matrix:

   $$Q_t=(1-a-b)\bar{Q}+az_{t-1}z_{t-1}^{\top}+bQ_{t-1}$$

   $Q_t$ is normalised to $R_t$, and correlated standardised shocks (Normal or Student-t) are generated with a Cholesky decomposition.

4. **Update portfolio value.** Log-returns are converted into simple returns and combined with the weights. Assuming daily rebalancing:

   $$V_t=V_{t-1}\left(1+\sum_{i=1}^{N}w_iR_{i,t}\right)$$

Risk is measured on the distribution of terminal profit and loss,

$$PnL^{(s)}=V_H^{(s)}-V_0$$

For a lower-tail probability $\alpha$, the horizon VaR and Expected Shortfall are

$$VaR_{1-\alpha}=-Q_{\alpha}(PnL)$$

$$ES_{1-\alpha}=-\mathbb{E}\left[PnL\mid PnL\leq Q_{\alpha}(PnL)\right]$$

where $Q_{\alpha}$ is the empirical $\alpha$-quantile of terminal PnL. Both measures are reported in monetary terms, and VaR also as a percentage of the initial portfolio value. The mean simulated path summarises the average evolution of the paths, but it is not a forecast or necessarily the most likely outcome.

![Simulated portfolio value paths](images/mont_paths.png)

<!-- TODO: add the numerical results (horizon, VaR and ES, Normal vs Student-t) from the notebook. -->

**Limitations.** Results depend on the fitted parameters, the shock distribution, the rebalancing convention and the horizon. Monte Carlo estimates carry simulation error, especially for extreme quantiles such as 99.9% VaR; more paths improve numerical precision but do not remove model risk. As in the one-day case, the simulation uses full-sample estimates and the final observed state, so it is a conditional scenario analysis and not an out-of-sample validation.




