# Clinical Event Time-Series Prediction (MIMIC-III Sepsis subset)

This project builds a multi-method pipeline to forecast clinical time series (focus on Heart Rate) for ICU patients, combining classical statistics, deep learning, and probabilistic modeling. It was completed as part of a Neural Networks and Deep Learning course assignment.

## 📌 Project Description

The goal is to predict near-future clinical measurements using a combination of:
- SARIMAX: classical time-series model with an exogenous regressor (O2Sat), guided by ADF/ACF/PACF.
- Deep models: Dense (state-space inspired), LSTM, GRU.
- Bidirectional RNNs: Bi-LSTM and Bi-GRU to capture richer context.
- Probabilistic modeling (MLE): LSTM with Gaussian outputs predicting mean (μ) and variance (σ²) via Gaussian NLL.

Evaluation includes MSE, MAE, R², and Cosine Distance, alongside a persistence baseline. We also analyze robustness under Laplace noise and quantify uncertainty with σ².

## ⚙️ Tasks Overview

### 1) Data Preparation and Statistical Analysis
- Loaded pre-split Train/Validation/Test CSVs; Hour encodes temporal progression, Patient_ID groups sequences.
- Correlation analysis (Validation):
  - Computed correlation matrix; identified the 20 features with lowest mean absolute correlation.
  - Rationale: lower multicollinearity → more complementary signals for forecasting.
- Grouped by Patient_ID so each patient forms a multivariate time series.
- Stationarity (HR):
  - Differenced once; ADF p-value < 0.05 → d ≈ 1.
  - ACF/PACF used to guide AR/MA orders.
- SARIMAX with O2Sat as exogenous regressor:
  - Fit/evaluated on validation and strict test splits; reported R².

### 2) Methodology Overview
- Families of models (one-liners):
  - ARIMA/SARIMAX: y_t = Σ φ_i y_{t−i} + Σ θ_j ε_{t−j} + βᵀx_t + ε_t
  - State-space Markov (Dense): y_t ≈ f([y_{t−1}, x_t, …]) with a feed-forward network approximating transition/observation mappings.
  - RNNs (LSTM/GRU): h_t = RNN(h_{t−1}, x_t); ŷ_t = g(h_t)
  - Bidirectional RNNs: concatenate forward/backward encodings for richer context.
  - Probabilistic (MLE): predict μ_t and σ²_t; optimize Gaussian negative log-likelihood (NLL).
- State-Space Markov Event prediction (Dense):
  - Design: flattened window → Dense(hidden) → linear head.
  - Activations: ReLU (hidden), Linear (output). Normalization is permissible if leakage-free.
  - This implementation used hidden width 64, LR 5e-3, batch size 32.
- LSTM-based event prediction:
  - Forward pass: window x_{t−W+1:t} → LSTM → h_t → Dense → ŷ_{t+1}
  - Loss: MSE (deterministic). For probabilistic, optimize Gaussian NLL over (μ, σ²).
- Bidirectional LSTM/GRU:
  - Structure: forward/backward encoders; concatenated states fed to a Dense head.
  - Advantage over unidirectional: better local context utilization for short-horizon prediction.
  - All models include a final fully-connected head.

### 3) Windowing, Features, and Normalization
- Windowing:
  - Input window (W_in) = 12 steps; forecast horizon (W_out) = 1 step, guided by ACF/PACF.
- Features per time step (deep models):
  - Hour (separately scaled), HR lags (1, 3, 6, 12), and 3-hour rolling mean → 6 features.
  - For broader analysis, the 20 lowest-correlation features were considered per instructions.
- Normalization:
  - Min-Max fitted on Train; applied to Validation/Test to avoid leakage.
  - Hour scaled separately per split.
  - Why not mean-variance? Min-Max is less sensitive to split-wise distribution shifts/outliers common in clinical data.
- Baseline:
  - Persistence (ŷ_t = y_{t−1}); reported for reference.

### 4) Model Training & Evaluation
- Models trained: Dense, LSTM, GRU, Bi-LSTM, Bi-GRU; SARIMAX (exog O2Sat); LSTM–Gaussian (μ, σ² via NLL).
- Training: up to 20 epochs, early stopping on validation loss, batch size 32, LR 5e-3 (deterministic) and 1e-4 (Gaussian-NLL).
- Metrics: MSE, MAE, R², Cosine Distance.
- Explained Variance Score (EVS): 1 − Var(y − ŷ) / Var(y); useful to assess how much variation the model explains in time series.

### 5) Results Analysis

- Persistence baseline: R² ≈ 0.671
- SARIMAX:
  - Validation (cleaned range): R² ≈ 0.683
  - Test (strict split): R² ≈ −0.433
- Deep models (Test):

| Model                 | MSE      | MAE      | R²      | Cosine Distance |
|-----------------------|----------|----------|---------|------------------|
| Dense                 | 0.001468 | 0.028169 | 0.778972 | ~9.50e−09        |
| LSTM                  | 0.001289 | 0.024377 | 0.806027 | ~1.25e−08        |
| GRU                   | 0.001297 | 0.024783 | 0.804753 | ~6.91e−09        |
| Bi-LSTM               | 0.001273 | 0.025886 | 0.808312 | ~8.21e−09        |
| Bi-GRU                | 0.001236 | 0.025325 | 0.813971 | ~7.34e−09        |
| LSTM–Gaussian (μ, σ²) | 0.001600 | 0.029200 | 0.754500 | —                |

- Observations:
  - All deep models outperformed the persistence baseline and SARIMAX on the strict test split.
  - Bidirectional variants slightly improved R²; Bi-GRU performed best.
  - LSTM–Gaussian had lower point R² but provides uncertainty (σ²), valuable clinically.
  - SARIMAX fit validation reasonably but degraded under test distribution shift.

### 6) Discussion & Insights
- Lowest-correlation feature selection reduced redundancy and stabilized learning.
- Min-Max scaling (with separate Hour scaling) minimized leakage and handled nonstationary shifts/outliers better than mean-variance scaling.
- LSTM vs. GRU vs. Bidirectional:
  - LSTM and GRU were close; GRU often trained a bit faster.
  - Bidirectional encoders captured richer local context and nudged performance upward.
- Probabilistic MLE:
  - Under injected Laplace noise, Gaussian-NLL models increased σ² around abrupt changes, expressing uncertainty where needed and showing better robustness characteristics than purely deterministic losses.

## 📊 Example Results
- Best test R²: Bi-GRU (~0.814), followed by Bi-LSTM (~0.808) and LSTM (~0.806).
- Dense provided a strong baseline but trailed RNNs (~0.779).
- Persistence baseline (~0.671) was surpassed by all deep models.
- SARIMAX struggled on test (negative R²), highlighting sensitivity to regime shifts.

## 🧠 Key Takeaways
- Deep sequential models (especially Bi-GRU) deliver substantial gains for clinical HR forecasting over statistical baselines.
- Pre-analysis with ADF/ACF/PACF remains useful to set differencing and window lengths, even with neural networks.
- Selecting low-correlation features and careful normalization are crucial in clinical time-series pipelines.
- Uncertainty-aware forecasting (μ, σ²) is clinically important: slightly lower point accuracy but better risk-aware decisions.
