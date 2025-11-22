# FinancialDenoising Project Context

## Project Overview

**Purpose**: VP-SDE diffusion-based denoising framework for financial time series using Mamba state-space models.

**Goal**: Denoise 94 noisy financial features from S&P 500 stocks to improve downstream ML model performance for forward returns prediction.

**Current Status**: ✅ **Paper-compliant implementation complete** (2025-11-22)
- Training: Pure MSE loss (NO guidance)
- Inference: Iterative denoising with TV + Fourier guidance (Algorithm 2)
- Next: Retrain models and validate performance

---

## Architecture Overview

### 1. VP-SDE (Variance Preserving SDE)

**Location**: `models/diffusion_mamba/vp_sde.py`

**Theory**:
```python
# Forward diffusion (add noise)
x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε

# Reverse diffusion (denoise)
x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z
```

**Parameters**:
- Linear β schedule: β_min=0.0001, β_max=0.02
- Total timesteps: T=1000
- Device-agnostic (CPU/CUDA)

**Key Methods**:
- `sample(x0, t)`: Forward process (add noise)
- `denoise_step(x_t, t, predicted_noise)`: Reverse process (one denoising step)
- `marginal_prob(x0, t)`: Get mean and std at timestep t
- `alphas_cumprod`: Cumulative product of alphas (α_bar)

### 2. Mamba Denoiser

**Location**: `models/diffusion_mamba/denoiser.py`

**Architecture**:
```
Input: [batch, 60, n_features]
  ↓
Time Embedding (sinusoidal, dim=64)
  ↓
4 × BiMamba Blocks (d_model=128)
  ↓
Output: Predicted Noise ε [batch, 60, n_features]
```

**Key Design Choices**:
- **Bidirectional Mamba**: Captures both past and future context within window
- **Time Conditioning**: Sinusoidal timestep embedding added to each layer
- **Noise Prediction**: Model outputs ε_θ(x_t, t), NOT clean signal x_0

**Model Variants**:
- `CausalMambaDenoiser`: Causal-only processing (for online deployment)
- `MultivariateMambaDenoiser`: Bidirectional (for research/offline)

### 3. Feature Clustering

**Location**: `models/clustering/`, `training/cluster_features.py`

**Rationale**: Different financial features have different noise characteristics.

**Approach**:
- K-means clustering (K=7) on 94 features
- Cluster by statistical properties: volatility, autocorrelation, trend strength
- Train **separate denoiser per cluster** (7 models total)

**Cluster Types**:
- **Mean-reverting**: High autocorrelation, stable mean
- **Trending**: Strong directional movement
- **Random walk**: Low autocorrelation, high noise
- **Volatile**: High variance, rapid changes

---

## Training Pipeline (Paper-Compliant)

### Training Objective

**File**: `training/train_denoiser.py`

**Loss Function** (Paper methodology):
```python
# NO guidance in training (guidance_weight = 0.0)
loss = MSE(predicted_noise, actual_noise)
```

**Previous (INCORRECT)**:
```python
# Old approach violated paper methodology
loss = MSE(predicted_noise, actual_noise) + 0.1 * (TV_loss + Fourier_loss)
```

**Process**:
1. Load cluster configuration and extract features
2. Create overlapping windows (stride=1, window=60)
3. **Instance normalization**: Each window normalized to mean=0, std=1
4. Sample random timestep t ~ Uniform(0, 1000)
5. Add noise: `x_t, noise = sde.sample(x_0, t)`
6. Predict noise: `predicted_noise = model(x_t, t)`
7. Compute loss: `loss = MSE(predicted_noise, noise)`
8. Track IC (Information Coefficient) during validation

**Output**: `trained_models/cluster_X_best.pt`
- `model_state_dict`: Trained model weights
- `normalization_mean/std`: Training statistics (for instance norm)
- `n_features`, `window_size`, `d_model`, `n_layers`: Architecture config
- `normalization_type`: "instance" (new) or "global" (old models)

### Training Commands

```bash
# Single cluster
uv run python training/train_denoiser.py \
    --cluster_id 0 \
    --data_path ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --guidance_weight 0.0 \
    --device cuda

# All clusters (Kaggle)
uv run python kaggle_train_all_clusters.py
```

---

## Inference Pipeline (Paper Algorithm 2)

### Iterative Denoising with Guidance

**File**: `inference/denoise_causal.py`

**Implementation**: Paper Algorithm 2 with TV + Fourier guidance

```python
def denoise_single_window(model, window_raw, sde, device, num_steps=50):
    """
    Paper-compliant iterative denoising:

    1. Instance normalize: window_norm = (window - mean) / std
    2. Start from observation (treated as noisy x_t)
    3. Iterative refinement (num_steps iterations):
         for t in [500, 490, 480, ..., 10, 0]:
             # Predictor: Reverse SDE step
             predicted_noise = model(x_t, t)
             x_t = sde.denoise_step(x_t, t, predicted_noise)

             # Guidance: TV + Fourier
             x_t = x_t - η_TV * ∇L_TV(x_t)
             x_t = x_t - η_F * ∇L_F(x_t, x_ref)
    4. Extract last row (causal!)
    5. Denormalize: output = output * std + mean
    """
```

**Key Modules**:

1. **`models/diffusion_mamba/iterative_denoiser.py`**:
   - `iterative_denoise()`: Full Algorithm 2 implementation
   - `langevin_corrector()`: Optional MCMC refinement (not used by default)
   - `denoise_single_window_iterative()`: Wrapper for single window

2. **`models/diffusion_mamba/guidance.py`**:
   - `compute_tv_gradient()`: Total Variation gradient (promotes smoothness)
   - `compute_fourier_gradient()`: Fourier loss gradient (suppresses high frequencies)
   - `apply_guidance()`: Combined TV + Fourier guidance application

**Hyperparameters**:
- `num_steps`: Number of denoising iterations (default: 50)
- `noise_level`: Initial timestep T' (default: 500, middle of 1000)
- `eta_tv`: TV guidance strength (default: 0.01)
- `eta_fourier`: Fourier guidance strength (default: 0.01)
- `fourier_threshold`: Frequency cutoff (default: 0.1, keep 10% low frequencies)

### Causality Guarantee

**Causal Processing** (Production-ready):
```python
# For each row t:
window = data[t-59:t+1]  # Past 60 days including current
denoised_row_t = denoise_single_window(model, window, sde, device)
# Row t NEVER sees row t+1 → No future leakage
```

**Files**:
- ✅ `inference/denoise_causal.py`: Stride=1, causal (PRODUCTION)
- ⚠️ `inference/denoise_dataset.py`: Stride=60, non-causal (EVALUATION ONLY)

### Inference Commands

```bash
# Causal denoising (production)
uv run python inference/denoise_causal.py \
    --input_csv ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised_causal.csv \
    --num_steps 50 \
    --device cuda

# Non-causal denoising (evaluation)
uv run python inference/denoise_dataset.py \
    --input_csv ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised_noncausal.csv \
    --device cuda
```

---

## Guidance Theory (Paper Equations 16-17)

### Total Variation (TV) Guidance

**Purpose**: Promote temporal smoothness (reduce high-frequency noise)

**Equation 16**:
```
L_TV(x) = Σ_{i=1}^{L-1} |x[i+1] - x[i]|
```

**Gradient**:
```python
∇L_TV[i] = sign(x[i] - x[i-1]) - sign(x[i+1] - x[i])
```

**Application**: `x ← x - η_TV * ∇L_TV(x)`

**Best for**: Mean-reverting features, oscillating patterns

### Fourier Guidance

**Purpose**: Suppress high-frequency components (preserve trends)

**Equation 17**:
```
L_F(x_t, x_ref) = ||FFT(x_t) - LowPass(FFT(x_ref))||²
```

**Gradient**:
```python
∇L_F = 2 * IFFT(FFT(x_t) - LowPass(FFT(x_ref)))
# Only penalize high frequencies (above threshold)
```

**Application**: `x ← x - η_F * ∇L_F(x, x_ref)`

**Best for**: Trending features, smooth macro patterns

---

## File Structure

```
FinancialDenoising/
├── models/
│   ├── diffusion_mamba/
│   │   ├── __init__.py              # Module exports
│   │   ├── vp_sde.py                # VP-SDE implementation
│   │   ├── denoiser.py              # Mamba denoiser models
│   │   ├── mamba_block.py           # Bidirectional Mamba blocks
│   │   ├── losses.py                # Loss functions (MSE, TV, Fourier)
│   │   ├── guidance.py              # ✨ NEW: TV/Fourier gradients
│   │   └── iterative_denoiser.py    # ✨ NEW: Algorithm 2 implementation
│   └── clustering/
│       ├── cluster_manager.py       # K-means clustering
│       └── feature_analyzer.py      # Feature statistics
│
├── training/
│   ├── train_denoiser.py            # Main training script
│   ├── cluster_features.py          # Clustering pipeline
│   └── kaggle_train_all_clusters.py # Train all 7 clusters
│
├── inference/
│   ├── denoise_causal.py            # ✅ Causal denoising (PRODUCTION)
│   ├── denoise_dataset.py           # ⚠️ Non-causal (EVALUATION ONLY)
│   └── visualize_denoising.py       # Visualization tools
│
├── clustering_results/
│   ├── cluster_assignments.json     # Feature→Cluster mapping
│   ├── cluster_labels.csv           # Cluster metadata
│   └── feature_analysis.csv         # Feature statistics
│
├── trained_models/
│   ├── cluster_0_best.pt            # 7 cluster-specific models
│   ├── cluster_1_best.pt            # Each contains model + norm stats
│   └── ...
│
├── Common/
│   └── evaluation/
│       └── validate_trading_signals.py  # Performance validation
│
├── CLAUDE.md                        # This file
└── README.md
```

---

## Data Sources

**Training Data**: `../TRMwithQuant/TinyRecursiveModels/CSVs/`

**Files**:
- `train_only.csv`: 7192 rows (80% split)
- `val_only.csv`: 1798 rows (20% split)

**Schema**:
- **Features**: F1-F94 (94 financial features)
- **Target**: `forward_returns` (next-day stock returns)
- **Metadata**: `date_id`, `risk_free_rate`, `market_forward_excess_returns`

**Preprocessing**:
- NaN/Inf handling: Replace with 0.0
- Instance normalization: Per-window mean=0, std=1
- No global normalization (scale-invariant)

---

## Current Issues & Next Steps

### ✅ RESOLVED Issues

1. **✅ Future Leakage**: Fixed with `denoise_causal.py` (stride=1, causal windowing)
2. **✅ Guidance in Training**: Removed (guidance_weight=0.0, paper-compliant)
3. **✅ Iterative Denoising**: Implemented (Algorithm 2 with TV + Fourier guidance)
4. **✅ Training-Inference Mismatch**: Aligned (training: pure MSE, inference: guidance)

### ⚠️ PENDING Validation

**Status**: Implementation complete, but need to retrain and validate

**Previous Results** (old single-step approach):
```
Tree models (XGBoost/LightGBM):  ✅ +2-4% Sharpe improvement
Linear models (Ridge/LinearReg): ❌ -40% IC, negative performance
```

**Hypothesis**: Iterative denoising may improve linear model performance

### 🔬 Experiments to Run

**Priority 1: Retrain with New Approach**
```bash
# Retrain all clusters with guidance_weight=0.0
uv run python kaggle_train_all_clusters.py
```

**Priority 2: Validate Iterative Denoising**
```bash
# Denoise with new iterative method
uv run python inference/denoise_causal.py \
    --num_steps 50 \
    --device cuda

# Validate performance
uv run python Common/evaluation/validate_trading_signals.py \
    --train_denoised train_denoised_causal.csv \
    --val_denoised val_denoised_causal.csv
```

**Priority 3: Hyperparameter Tuning**
- `noise_level`: [100, 250, 500, 750, 900]
- `num_steps`: [10, 25, 50, 100]
- `eta_tv`: [0.001, 0.01, 0.05]
- `eta_fourier`: [0.001, 0.01, 0.05]

**Priority 4: Feature-Level Analysis**
```python
# Visualize denoising effect
uv run python inference/visualize_denoising.py \
    --original train_only.csv \
    --denoised train_denoised_causal.csv \
    --features F1,F10,F50  # Example features
```

---

## Validation Metrics

### Primary (Competition Metric)
- **Adjusted Sharpe Ratio**: `Sharpe / (vol_penalty * return_penalty)`
  - `vol_penalty = max(1, realized_vol / target_vol)`
  - `return_penalty = 1 - max(0, (cum_return - target_return) / target_return)`

### Secondary
- **IC (Information Coefficient)**: Correlation(prediction, forward_returns)
- **Sharpe Ratio**: `mean(returns) / std(returns) * √252`
- **Cumulative Returns**: Total strategy returns over validation period
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: % of days with positive returns

### Model Types Tested
- **Linear**: Ridge, LinearRegression, Lasso
- **Tree**: XGBoost, LightGBM, CatBoost
- **Neural**: MLP, Transformer (future)

**Goal**: Denoised data should improve ALL metrics vs original data

---

## Key Design Principles

### 1. Instance Normalization (Scale Invariance)
```python
# Each window normalized independently
window_norm = (window - window.mean()) / (window.std() + 1e-6)
```

**Rationale**: Financial data has varying scales across time and features. Instance norm provides scale invariance.

### 2. Causal Processing (No Future Leakage)
```python
# Row t uses ONLY rows [t-59:t]
for t in range(59, len(data)):
    window = data[t-59:t+1]
    denoised[t] = denoise_window(window)[-1]  # Last row only
```

**Rationale**: Production deployment requires strict causality (no peeking into future).

### 3. Cluster-Specific Models (Feature Heterogeneity)
```python
# 7 separate models for 7 feature clusters
cluster_0: mean-reverting features (e.g., volatility)
cluster_1: trending features (e.g., momentum)
...
```

**Rationale**: Different features have different noise characteristics → specialized denoisers.

### 4. Noise Prediction (Not Direct Denoising)
```python
# Model learns to predict noise, not clean signal
predicted_noise = model(x_t, t)
x_0 = (x_t - √(1-α_bar) * predicted_noise) / √α_bar
```

**Rationale**: Noise prediction is easier to learn and more stable (DDPM insight).

---

## Implementation Timeline

**2025-11-13**: Initial implementation with single-step denoising
- Trained models with guidance_weight=0.1 (INCORRECT)
- Used single-step denoising at t=500
- Validation showed mixed results (trees good, linear bad)

**2025-11-22**: Paper-compliant refactor
- ✅ Removed guidance from training (guidance_weight=0.0)
- ✅ Implemented iterative denoising (Algorithm 2)
- ✅ Added TV and Fourier guidance during inference
- ✅ Created new modules: `guidance.py`, `iterative_denoiser.py`
- ⏳ Need to retrain and validate

---

## Reference Papers

**Primary Reference**:
- Paper: `2409.02138v1 (1).pdf` (in project root)
- Title: VP-SDE-based Diffusion Denoising for Financial Time Series
- Key Algorithms: Algorithm 2 (Iterative Denoising with Guidance)

**Foundational Papers**:
- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **Mamba**: State Space Models for Sequence Modeling (Gu & Dao, 2023)
- **VP-SDE**: Score-Based Generative Modeling through SDEs (Song et al., 2021)

---

## Common Commands Reference

```bash
# 1. Feature Clustering (one-time)
uv run python training/cluster_features.py \
    --data_path ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --n_clusters 7

# 2. Train all clusters
uv run python kaggle_train_all_clusters.py

# 3. Train single cluster
uv run python training/train_denoiser.py \
    --cluster_id 0 \
    --epochs 100 \
    --device cuda

# 4. Causal denoising (production)
uv run python inference/denoise_causal.py \
    --input_csv train_only.csv \
    --output_csv train_denoised.csv \
    --num_steps 50

# 5. Validate performance
uv run python Common/evaluation/validate_trading_signals.py \
    --train_denoised train_denoised.csv \
    --val_denoised val_denoised.csv

# 6. Visualize results
uv run python inference/visualize_denoising.py \
    --original train_only.csv \
    --denoised train_denoised.csv
```

---

## FAQ

**Q: Why instance normalization instead of global?**
A: Financial data has non-stationary distributions. Instance norm provides scale invariance and adapts to local statistics.

**Q: Why predict noise instead of clean signal?**
A: DDPM research shows noise prediction is easier to learn and more stable than direct denoising.

**Q: Why separate models per cluster?**
A: Different features have different characteristics (mean-reverting vs trending). Specialized models work better.

**Q: Why iterative denoising instead of single-step?**
A: Paper Algorithm 2 uses iterative refinement with guidance. Single-step was a simplification that may lose quality.

**Q: Why guidance only in inference, not training?**
A: Paper methodology: train with pure MSE to learn data distribution, apply domain constraints (smoothness, spectral) during sampling.

**Q: Is causal denoising slower than non-causal?**
A: Yes (stride=1 vs stride=60), but causal is required for production deployment (no future leakage).

---

## Troubleshooting

### Training Issues

**Problem**: OOM (Out of Memory) errors
```bash
# Reduce batch size
--batch_size 128  # Default is 256
```

**Problem**: Slow training
```bash
# Use smaller model
--d_model 64 --n_layers 2  # Default: 128, 4
```

**Problem**: NaN losses
```bash
# Check data for NaN/Inf
# Reduce learning rate
--learning_rate 5e-5  # Default: 1e-4
```

### Inference Issues

**Problem**: Slow inference
```bash
# Reduce num_steps
--num_steps 25  # Default: 50

# Disable guidance (faster but lower quality)
# Modify denoise_causal.py: eta_tv=0.0, eta_fourier=0.0
```

**Problem**: Poor denoising quality
```bash
# Increase num_steps
--num_steps 100

# Tune guidance strengths
# Modify denoise_causal.py: eta_tv=0.05, eta_fourier=0.05
```

---

## Next Session TODO

1. **Retrain all 7 clusters** with guidance_weight=0.0
2. **Run validation** on new iterative denoising approach
3. **Compare metrics**: Old (single-step) vs New (iterative)
4. **Hyperparameter tuning** if results are promising
5. **Feature-level analysis** to understand what denoising does
6. **Write results** to project report/paper

**Expected Outcome**: Iterative denoising should improve both tree AND linear model performance (unlike previous mixed results).
