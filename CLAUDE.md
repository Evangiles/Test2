# FinancialDenoising Project Context

## Project Overview

**Purpose**: Diffusion-based denoising framework for financial time series data using Mamba state-space models.

**Goal**: Clean noisy financial features (94 features from S&P 500 stocks) to improve downstream ML model performance for forward returns prediction.

**Status**: ⚠️ Implementation complete but has **critical issues** that need investigation.

---

## Model Architecture

### 1. VP-SDE (Variance Preserving SDE)

**Location**: `models/diffusion_mamba/vp_sde.py`

**Theory**:
- Forward process: `x_t = √(α_bar) * x_0 + √(1-α_bar) * ε`
- Reverse process: Recover `x_0` from noisy `x_t`
- Linear β schedule: β_min=0.0001, β_max=0.02, T=1000 steps

**Key Functions**:
```python
sde.sample(x0, t)  # Forward: Add noise
sde.marginal_prob(x0, t)  # Get mean/std at timestep t
sde.alphas_cumprod  # α_bar values for each timestep
```

### 2. Mamba Denoiser

**Location**: `models/diffusion_mamba/denoiser.py`

**Architecture**:
- Input: `[batch, 60, n_features]` (60-day windows)
- Bidirectional Mamba blocks (4 layers)
- Time embedding: Sinusoidal encoding of diffusion timestep
- Output: **Predicted noise** ε (same shape as input)

**Training Objective**:
```python
x_t, noise = sde.sample(x0, t)  # Add synthetic noise
predicted_noise = model(x_t, t)
loss = MSE(predicted_noise, noise) + guidance_loss
```

**CRITICAL**: Model is trained to predict **noise**, not clean signal!

### 3. Feature Clustering

**Location**: `models/clustering/`, `training/cluster_features.py`

**Approach**:
- K-means clustering (K=7) on 94 features
- Cluster by characteristics: mean-reverting, trending, random walk
- Train **separate denoiser per cluster** (7 models total)

**Guidance Losses** (`models/diffusion_mamba/losses.py`):
- **TV Loss**: Penalize high-frequency changes (for mean-reverting)
- **Fourier Loss**: Suppress high frequencies (for trending)
- **Cluster-specific weights**: Adjust guidance based on cluster type

---

## Current Implementation

### Training Pipeline

**File**: `training/train_denoiser.py`

**Process**:
1. Load cluster configuration (`clustering_results/cluster_assignments.json`)
2. Extract features for specific cluster
3. Create overlapping windows (stride=1, window=60)
4. Normalize using training statistics (mean/std saved to checkpoint)
5. Train model to predict noise:
   ```python
   x_t, noise = sde.sample(x0, t)
   predicted_noise = model(x_t, t)
   predicted_x0 = (x_t - sqrt(1-alpha_bar) * predicted_noise) / sqrt(alpha_bar)
   loss = MSE(predicted_noise, noise) + guidance_weight * guidance_loss(predicted_x0)
   ```

**Output**: `trained_models/cluster_X_best.pt` (contains model + normalization stats)

### Inference Pipeline

**Files**:
- `inference/denoise_causal.py` (Production, causal)
- `inference/denoise_dataset.py` (Evaluation, non-causal)

**Current Implementation** (`denoise_causal.py`):
```python
def denoise_single_window(model, window_norm, sde, device):
    """
    CURRENT APPROACH (Single-step denoising):
    1. Take normalized window [60, F] as x_t
    2. Set t = 500 (middle timestep)
    3. predicted_noise = model(x_t, t)
    4. x0 = (x_t - sqrt(1-alpha_bar) * predicted_noise) / sqrt(alpha_bar)
    5. Return last row only (causal)
    """
    window_tensor = torch.FloatTensor(window_norm).unsqueeze(0).to(device)
    t = torch.full((1,), sde.num_timesteps // 2, device=device, dtype=torch.long)

    # Model predicts NOISE
    predicted_noise = model(window_tensor, t)

    # Recover clean signal
    alpha_bar = sde.alphas_cumprod[t]
    while alpha_bar.dim() < window_tensor.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)

    denoised_window = (window_tensor - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / (torch.sqrt(alpha_bar) + 1e-8)
    denoised_window = torch.clamp(denoised_window, -10, 10)

    # Return ONLY last row (causal)
    return denoised_window[0, -1, :].cpu().numpy()
```

---

## ⚠️ CRITICAL ISSUES

### Issue 1: Future Leakage (RESOLVED in causal version)

**Problem**: stride=60 version uses future information

**Example**:
```
denoise_dataset.py (stride=60):
Window [0:60] → denoises rows 0-59 (ALL rows use info from each other)
Window [60:120] → denoises rows 60-119

Row 0 denoising uses information from rows 1-59 → Future leakage!
```

**Solution**: `denoise_causal.py` (stride=1)
```
Row t denoising:
- Window [t-59:t] (only past 60 days)
- Model outputs [60, F] but use ONLY last row
- Row t never sees row t+1

✅ NO future leakage, suitable for production
```

### Issue 2: Single-step vs Iterative Denoising (UNRESOLVED)

**Current Approach** (Single-step):
```python
t = 500  # Fixed middle timestep
predicted_noise = model(x_t, t)  # One-shot prediction
x0 = (x_t - sqrt(1-alpha_bar) * predicted_noise) / sqrt(alpha_bar)
```

**Theory-Correct Approach** (Iterative, from DDPM paper):
```python
# Start from noisy observation
x_t = add_noise(x0, t=T')  # Add noise up to T' steps

# Iteratively denoise T' → 0
for i in range(T', 0, -1):
    t = i
    predicted_noise = model(x_t, t)
    x_t = sde.denoise_step(x_t, t, predicted_noise)  # One step back
    # Optional: Apply guidance (TV/Fourier)

x0_final = x_t  # Final denoised result
```

**Concern Raised by Critic**:
> "현재 코드는 깨끗한 데이터(clean observation)를 노이즈 섞인 데이터(x_t)로 착각하고 모델에 넣고 있다.
> Algorithm 2 (iterative denoising)를 전혀 구현하지 않았다."

**Counter-Argument**:
- Real financial data IS inherently noisy (not clean)
- Treating observation as x_t is valid
- Single-step denoising is common in practical applications (e.g., DDIM)
- **BUT**: Iterative might give better results

### Issue 3: Validation Results (CONTRADICTORY)

**Observation**:
```
Denoised data results:
✅ Tree models (XGBoost/LightGBM/CatBoost): +2-4% Sharpe improvement
❌ Linear models (Ridge/LinearReg): -40% IC, negative performance

Question: Why does denoising help tree models but hurt linear models?
```

**Possible Explanations**:
1. **Denoising works but changes data distribution** (tree models robust, linear models sensitive)
2. **t=500 is wrong hyperparameter** (too much/little noise assumed)
3. **Single-step insufficient** (need iterative refinement)
4. **Model undertrained or buggy**
5. **Normalization mismatch** (though we use training stats)

### Issue 4: Training vs Inference Consistency

**Training**:
```python
x0 = clean_data (synthetic clean from overlapping windows)
x_t, noise = sde.sample(x0, t=random(0, 1000))  # Add synthetic noise
predicted_noise = model(x_t, t)
```

**Inference (Current)**:
```python
x_t = observed_data (real market data, inherently noisy)
t = 500  # Fixed
predicted_noise = model(x_t, t)
x0 = recover_from_noise_prediction(x_t, predicted_noise, t)
```

**Question**: Is treating real observations as x_t at t=500 theoretically sound?

---

## File Structure

```
FinancialDenoising/
├── models/
│   ├── diffusion_mamba/
│   │   ├── __init__.py
│   │   ├── vp_sde.py           # VP-SDE implementation
│   │   ├── denoiser.py         # Mamba denoiser architecture
│   │   ├── mamba_block.py      # Bidirectional Mamba blocks
│   │   └── losses.py           # TV/Fourier guidance losses
│   └── clustering/
│       ├── cluster_manager.py  # K-means clustering
│       └── feature_analyzer.py # Feature statistics
│
├── training/
│   ├── train_denoiser.py       # Train cluster-specific denoisers
│   └── cluster_features.py     # K-means clustering pipeline
│
├── inference/
│   ├── denoise_causal.py       # ✅ Production (stride=1, causal)
│   ├── denoise_dataset.py      # ⚠️ Evaluation only (stride=60, non-causal)
│   └── visualize_denoising.py  # Visualization tools
│
├── clustering_results/
│   ├── cluster_assignments.json # Feature→Cluster mapping
│   ├── cluster_labels.csv       # Cluster metadata
│   └── feature_analysis.csv     # Feature statistics
│
├── trained_models/
│   ├── cluster_0_best.pt        # 7 cluster-specific models
│   ├── cluster_1_best.pt        # Each contains:
│   └── ...                      # - model_state_dict
│                                # - normalization_mean/std
│                                # - n_features, window_size, etc.
│
├── Common/
│   └── evaluation/
│       └── validate_trading_signals.py  # Trading performance validation
│
└── README.md
```

---

## Data Sources

**Training Data Location**: `../TRMwithQuant/TinyRecursiveModels/CSVs/`

Files:
- `train_only.csv`: 7192 rows (80% split)
- `val_only.csv`: 1798 rows (20% split)

**Features**:
- 94 financial features (F1-F94)
- Metadata: date_id, forward_returns, risk_free_rate, market_forward_excess_returns

**Target**: `forward_returns` (next-day returns for S&P 500 stocks)

---

## How to Run

### 1. Feature Clustering (One-time)

```bash
python training/cluster_features.py \
    --data_path ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --output_dir clustering_results \
    --n_clusters 7
```

### 2. Train Denoisers (GPU Required)

```bash
# Train all 7 clusters
for i in {0..6}; do
    python training/train_denoiser.py \
        --cluster_id $i \
        --data_path ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
        --epochs 100 \
        --device cuda
done
```

### 3. Causal Denoising (Production)

```bash
python inference/denoise_causal.py \
    --input_csv ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --output_csv train_denoised_causal.csv \
    --device cuda
```

### 4. Validation

```bash
python Common/evaluation/validate_trading_signals.py \
    --train_original ../TRMwithQuant/TinyRecursiveModels/CSVs/train_only.csv \
    --train_denoised train_denoised_causal.csv \
    --val_original ../TRMwithQuant/TinyRecursiveModels/CSVs/val_only.csv \
    --val_denoised val_denoised_causal.csv
```

---

## Key Questions to Resolve

### 1. Is Single-Step Denoising Valid?

**Current**: One-shot prediction at t=500

**Alternative**: Implement iterative denoising (Algorithm 2 from DDPM)

**Need to test**: Which gives better validation results?

### 2. What is the Optimal t Value?

**Current**: t=500 (middle of T=1000)

**Hypothesis**: Different t values assume different noise levels
- Small t (e.g., 100): Assumes less noise in observation
- Large t (e.g., 900): Assumes more noise in observation

**Experiment**: Grid search over t ∈ {100, 250, 500, 750, 900}

### 3. Should We Use Iterative Refinement?

**Option A** (Current): Single-step
```python
x0 = one_shot_denoise(x_t, t=500)
```

**Option B**: Multi-step (like DDPM)
```python
x_t = add_noise(observation, t=T')
for i in range(T', 0, -1):
    x_t = denoise_step(x_t, i)
x0 = x_t
```

**Option C**: Few-step (like DDIM)
```python
timesteps = [500, 400, 300, 200, 100, 0]
x_t = observation
for t in timesteps:
    x_t = denoise_step(x_t, t)
x0 = x_t
```

### 4. Why Do Linear Models Fail?

**Observation**: IC becomes negative after denoising for linear models

**Hypotheses**:
1. Denoising changes feature distribution (tree models robust, linear models not)
2. Denoising removes signal along with noise
3. Bug in implementation (though causal logic seems correct)
4. Normalization issues

**Next Steps**:
- Compare feature distributions before/after
- Check if denoising preserves correlation with target
- Visualize what denoising actually does to individual features

---

## Next Steps (Priority Order)

### High Priority

1. **Implement iterative denoising** as Option C (few-step DDIM)
   - Add `denoise_step()` method to VPSDE
   - Modify `denoise_single_window()` to support multi-step
   - Compare results with current single-step

2. **Hyperparameter search for t**
   - Grid search: t ∈ {100, 250, 500, 750, 900}
   - Metric: IC and Adjusted Sharpe on validation

3. **Feature-level analysis**
   - Plot original vs denoised for each feature
   - Check correlation with target before/after
   - Verify denoising doesn't destroy signal

### Medium Priority

4. **Implement n_seeds ensemble** (as mentioned in critic's Algorithm 2)
   - Average multiple denoising runs
   - May improve stability

5. **Add guidance during inference**
   - Apply TV/Fourier losses during iterative denoising
   - Currently guidance only used in training

6. **Test different timestep schedules**
   - Linear: [500, 400, 300, 200, 100]
   - Quadratic: [500, 450, 350, 200, 50]
   - Exponential: [500, 400, 250, 100, 10]

### Low Priority

7. **Retrain with different settings**
   - Different guidance weights
   - Different number of clusters
   - Longer training (more epochs)

---

## Validation Metrics

**Primary** (Competition metric):
- **Adjusted Sharpe Ratio**: `Sharpe / (vol_penalty * return_penalty)`

**Secondary**:
- **IC (Information Coefficient)**: Correlation between prediction and actual returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Cumulative Returns**: Total strategy returns
- **Max Drawdown**: Worst peak-to-trough decline

**Goal**: Denoised data should **improve** all metrics vs original data

---

## Important Notes

1. **Causality is CORRECT**: `denoise_causal.py` has no future leakage
2. **Training statistics**: Always use train_mean/train_std from checkpoint (no validation leakage)
3. **Model predicts NOISE**: Don't confuse with predicting clean signal
4. **VP-SDE formula**: `x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)` is the correct recovery formula
5. **Cluster models**: Each cluster has different characteristics (mean-reverting vs trending) → different guidance

---

## Recent Changes

**2025-11-13**:
- Fixed critical bug: Model output is noise, not clean signal
- Implemented VP-SDE recovery formula in inference
- Separated into independent project from TRMwithQuant

**Issues from Previous Session**:
- Validation results show contradictory performance (tree models good, linear models bad)
- Debate about whether single-step denoising is valid
- Critic suggested implementing full Algorithm 2 (iterative denoising)

---

## References

**VP-SDE Theory**: `models/diffusion_mamba/vp_sde.py` docstrings

**DDPM Paper**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)

**Mamba Architecture**: State-space models for sequence modeling

**Related Project**: `../TRMwithQuant/` (uses denoised features for TRM training)
