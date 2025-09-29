# Clinical Event Time-Series Prediction

This project implements a multi-method pipeline for forecasting clinical time series (with focus on Heart Rate) using the MIMIC-III–derived sepsis dataset. It was completed as part of a Neural Networks and Deep Learning course homework. We explore classical and deep learning methods, design both deterministic and probabilistic predictors, and analyze robustness under noise.

## 📌 Project Description

The aim of this project is to forecast near-future clinical measurements for ICU patients using:

- Classical statistical modeling:
  - SARIMAX: with an exogenous regressor (O2Sat) and orders guided by ADF, ACF, and PACF analyses.
- Deep learning models:
  - Dense (state-space inspired, Markov-style), LSTM, GRU.
  - Bidirectional variants: Bi-LSTM, Bi-GRU.
- Probabilistic modeling (maximum likelihood):
  - LSTM with Gaussian output (predicting mean μ and variance σ² per step) trained via Gaussian negative log-likelihood.

Evaluation is based on MSE, MAE, R², and Cosine Distance. A persistence baseline (last value carried forward) is also reported.

## ⚙️ Tasks Overview

### 1) Data Preparation and Statistical Analysis
- Dataset: pre-split CSVs for Train, Validation, and Test; “Hour” encodes time progression; “Patient_ID” identifies patient-specific sequences.
- Correlation analysis:
  - Computed the correlation matrix on Validation.
  - Selected the 20 features with lowest mean absolute correlation to reduce multicollinearity and favor more independent signals.
  - Rationale for “lowest correlation”: mitigates redundancy and helps models (especially state-space/event formulations) focus on complementary information.
- Grouping and series formation:
  - Grouped rows by Patient_ID; each patient forms one multivariate time series.
- Stationarity check (HR):
  - Differenced HR once and ran ADF; p-value < 0.05 after first difference ⇒ d ≈ 1.
  - Used ACF and PACF of differenced series to guide AR/MA order selection.
- SARIMAX (with exogenous O2Sat):
  - Fit and evaluated with both in-sample and held-out testing configurations.
  - Reported R² on a cleaned validation range and on a strict test split.

### 2) Methodology Overview
- Time-series forecasting families (high level):
  - Linear ARIMA/SARIMAX: y_t = Σ φ_i y_{t−i} + Σ θ_j ε_{t−j} + β^T x_t + ε_t
  - State-space Markov event (feed-forward): y_t ≈ f([y_{t−1}, x_t, …]) with dense layers approximating transition/observation mappings.
  - Recurrent models (LSTM/GRU): h_t = RNN(h_{t−1}, x_t); ŷ_t = g(h_t)
  - Bidirectional RNNs: concatenate forward/backward context during training to learn richer representations.
  - Probabilistic (MLE): predict μ_t and σ²_t; train with NLL to model uncertainty.
- State-Space Markov Event prediction (Dense):
  - Architecture: flattened time window → Dense hidden → linear head for next-step HR.
  - Activations: ReLU in hidden, Linear in output.
  - Normalization layers: can be used cautiously (no leakage, sequence-aware); optional.
  - Hyperparameters used: hidden width 64, learning rate 5e-3, batch size 32.
- LSTM-based event prediction:
  - Forward pass: x_{t−W+1:t} → LSTM → h_t → Dense head → ŷ_{t+1}
  - Loss: MSE for deterministic regression; for probabilistic variant, Gaussian NLL on (μ, σ²).
- Bidirectional RNNs:
  - Structure: forward and backward encoders; concatenated states improve context capture.
  - Advantage vs. unidirectional: richer sequence representations and better short-horizon forecasts.
  - Heads: all models use a final fully-connected layer to produce predictions.

### 3) Windowing, Features, and Normalization
- Windowing:
  - Input window (W_in): 12 time steps (guided by ACF/PACF behavior).
  - Forecast horizon (W_out): 1 step ahead.
- Features used for deep models:
  - Hour (scaled) and engineered HR features: lags (1, 3, 6, 12) and a 3-hour rolling mean ⇒ 6 features per step.
  - Only the 20 lowest-correlation features subset was used for the broader analysis (per assignment’s instruction).
- Normalization:
  - Min-Max scaling fit on Train; applied to Validation and Test to avoid leakage.
  - Hour scaled separately per split (per-timeframe scaling).
  - Rationale vs. mean-variance scalers: Min-Max is robust to distributional shifts and outliers across splits in clinical data; reduces risk of leakage from global statistics.
- Baselines:
  - Persistence (ŷ_t = y_{t−1})

### 4) Models Trained & Evaluation Protocol
- Models:
  - Dense (state-space inspired), LSTM, GRU, Bi-LSTM, Bi-GRU
  - SARIMAX with exogenous O2Sat
  - LSTM–Gaussian (μ, σ²), trained via Gaussian NLL (MLE approach)
- Training setup:
  - Epochs up to 20, early stopping on validation loss.
  - Batch size: 32 (also aligned with assignment options).
  - Learning rates: 5e-3 for deterministic deep models; 1e-4 for Gaussian NLL model.
- Metrics:
  - MSE, MAE, R², Cosine Distance
  - Explained Variance Score (EVS): 1 − Var(y − ŷ) / Var(y)
    - EVS is informative in time series as it measures the proportion of variance captured by the model; robust to scale and complements R².

### 5) Results Analysis
- Summary metrics (Test):
  - Persistence baseline: R² = 0.671
  - SARIMAX:
    - Validation (cleaned range): R² ≈ 0.683
    - Test (strict split): R² ≈ −0.433
  - Dense: MSE 0.00147, MAE 0.0282, R² 0.7790, CosDist ~ 9.50e-09
  - LSTM: MSE 0.00129, MAE 0.0244, R² 0.8060, CosDist ~ 1.25e-08
  - GRU: MSE 0.00130, MAE 0.0248, R² 0.8048, CosDist ~ 6.91e-09
  - Bi-LSTM: MSE 0.00127, MAE 0.0259, R² 0.8083, CosDist ~ 8.21e-09
  - Bi-GRU: MSE 0.00124, MAE 0.0253, R² 0.8140, CosDist ~ 7.34e-09
  - LSTM–Gaussian (MLE): MSE 0.0016, MAE 0.0292, R² 0.7545
- Observations:
  - All deep models surpassed the persistence baseline and SARIMAX on the strict test split.
  - Bidirectional models slightly improved over unidirectional counterparts; Bi-GRU achieved the best R².
  - The probabilistic LSTM–Gaussian underperformed deterministic RNNs on point R², but provides calibrated uncertainty (σ²) which is valuable clinically.
  - SARIMAX showed decent fit in-sample/validation but degraded on the strict test, indicating sensitivity to nonstationarity and covariate shifts.

### 6) Discussion & Insights
- Why lowest-correlation features?
  - Reduces multicollinearity, stabilizes parameter estimation, and complements event/state-space formulations by leveraging more independent signals.
- Why Min-Max scaling and separate Hour scaling?
  - Clinical data often exhibit nonstationary shifts and outliers. Min-Max trained on Train reduces leakage risk and keeps the temporal scale interpretable. Hour is monotonic and benefits from independent scaling.
- LSTM vs. GRU vs. Bidirectional:
  - LSTM and GRU performed similarly; GRU converged slightly faster in practice.
  - Bidirectional encoders captured richer short-term context during training, modestly improving R².
- Probabilistic (MLE) modeling and robustness:
  - Under Laplace noise injections, the Gaussian-NLL models produced larger σ² around abrupt changes, expressing uncertainty explicitly.
  - Visual inspection suggests the Bi-LSTM Gaussian yields smoother means and appropriate variance inflation near noisy segments—indicative of better robustness to heavy-tailed noise than purely deterministic losses.
- Trade-offs:
  - Deterministic RNNs deliver the best point forecasts.
  - Probabilistic models add clinically meaningful uncertainty quantification.
  - Classical models (SARIMAX) can be competitive with careful tuning but struggled under split distribution shifts.

## 📊 Example Results
- Best model (by R² on Test): Bi-GRU (R² ≈ 0.814)
- Strong contenders: Bi-LSTM and LSTM (~0.808–0.806)
- Dense baseline is solid but lags recurrent models (R² ≈ 0.779)
- Persistence baseline (R² ≈ 0.671) is outperformed by all deep models
- SARIMAX degraded on the strict test split (negative R²), highlighting challenges with regime shifts

## 🧠 Key Takeaways
- Deep sequential models (especially bidirectional GRU/LSTM) provide substantial gains for clinical HR forecasting over both statistical baselines and persistence.
- Pre-analysis with ADF/ACF/PACF is valuable to set differencing and window lengths, even when using neural models.
- Selecting low-correlation features and careful normalization help mitigate leakage and multicollinearity in clinical datasets.
- Probabilistic forecasting (μ, σ²) is crucial in healthcare: slightly lower point accuracy but richer, uncertainty-aware predictions suitable for risk-sensitive decision-making.
