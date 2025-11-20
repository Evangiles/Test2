# Research Request: Optimal Loss Threshold for VP-SDE Denoising Model

## Project Context

I'm training a **VP-SDE (Variance Preserving Stochastic Differential Equation)** based denoising model for financial time series data using Mamba state-space architecture. I need theoretical and empirical research on appropriate **training loss thresholds** for early stopping.

---

## Model Architecture

### 1. VP-SDE Forward Process

**Theoretical Foundation**: DDPM (Denoising Diffusion Probabilistic Models)

**Forward SDE**:
```
dx = -0.5 * β(t) * x dt + √β(t) dw
```

**Discrete Implementation**:
```python
# Linear β schedule
β_min = 0.0001
β_max = 0.02
T = 1000 timesteps

# α values
α_t = 1 - β_t
α_bar_t = ∏(α_s) for s=0 to t

# Forward diffusion (add noise)
x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε
where ε ~ N(0, I)
```

### 2. Denoiser Architecture

**Model**: CausalMamba (Bidirectional Mamba blocks)

**Input**:
- Shape: `[batch_size, window_size=60, n_features]`
- Normalization: **Instance normalization** (per-window mean=0, std=1)
- Time embedding: Sinusoidal encoding of diffusion timestep t

**Output**:
- **Predicted noise** ε_θ(x_t, t) ∈ R^(60 × n_features)
- NOT the clean signal x_0 directly

**Architecture Details**:
```python
class CausalMambaDenoiser(nn.Module):
    def __init__(self, n_features, window_size=60, d_model=128, n_layers=4):
        # Time embedding: sinusoidal position encoding
        # Input projection: [B, 60, F] → [B, 60, 128]
        # Bidirectional Mamba blocks × 4 layers
        # Output projection: [B, 60, 128] → [B, 60, F]
```

### 3. Training Objective

**Loss Function**:
```python
# 1. Sample random timestep
t ~ Uniform(0, T-1)

# 2. Forward diffusion (add synthetic noise)
ε ~ N(0, I)
x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε

# 3. Model predicts noise
ε_pred = model(x_t, t)

# 4. Recover predicted x_0 (for guidance loss only)
x_0_pred = (x_t - √(1 - α_bar_t) * ε_pred) / √(α_bar_t)

# 5. Total loss
loss = MSE(ε_pred, ε) + λ * guidance_loss(x_0_pred)
```

**Guidance Losses** (cluster-specific):
- **Total Variation (TV)**: `∑|x_0_pred[t] - x_0_pred[t-1]|` for mean-reverting features
- **Fourier Loss**: Suppress high frequencies for trending features
- **Guidance weight**: λ = 0.1

---

## Data Characteristics

### Input Data Statistics

**After Instance Normalization** (per-window):
- Mean: 0.0 (by design)
- Std: 1.0 (by design)
- Range: Typically [-3, +3] (99.7% of normalized values)
- Clamping: Output clamped to [-10, +10] to prevent extreme values

**Training Setup**:
- Number of clusters: 5 (K-means on 85 financial features)
- Window size: 60 days (sliding window, stride=1)
- Batch size: 32
- Total windows: ~9,000 (from train_only.csv)

### Noise Distribution

**Synthetic Noise** (ε):
- Distribution: N(0, I) (standard normal)
- Per-element variance: 1.0
- Expected MSE between two random N(0,I) samples: **2.0**

---

## Baseline Calculation

```python
# Random baseline: Predict random noise instead of true noise
true_noise = torch.randn(batch, 60, n_features)  # N(0,1)
random_pred = torch.randn(batch, 60, n_features)  # N(0,1)

mse_random = E[(random_pred - true_noise)²]
            = E[(N(0,1) - N(0,1))²]
            ≈ 2.0
```

---

## Research Questions

### 1. Theoretical Optimal Loss Threshold

**Question**: What is the **theoretically justified MSE threshold** for stopping VP-SDE training?

**Sub-questions**:
- Is there a lower bound below which further training provides diminishing returns?
- How does the optimal threshold relate to:
  - Number of diffusion timesteps T (1000)?
  - Instance normalization (mean=0, std=1)?
  - Data dimensionality (window_size × n_features = 60 × F)?
  - β schedule (linear from 0.0001 to 0.02)?

### 2. DDPM Literature Review

**Question**: What loss values are reported in **DDPM/VP-SDE papers** for similar setups?

**Context**:
- Original DDPM paper (Ho et al., 2020)
- Score-based SDE papers (Song et al., 2021)
- Financial time series applications (if any)

**Specific interests**:
- Final MSE values reported
- Convergence criteria used
- Relationship between training loss and denoising quality

### 3. Instance Normalization Impact

**Question**: How does **instance normalization** affect the loss scale?

**Context**:
- Input data: mean=0, std=1 (per window)
- Noise: N(0, 1)
- Does this make loss scale different from typical DDPM implementations?

### 4. Guidance Loss Interaction

**Question**: How does the **guidance term** affect the optimal threshold?

**Current setup**:
```python
total_loss = mse_loss + 0.1 * guidance_loss
```

**Concerns**:
- Should the threshold be based on `mse_loss` alone or `total_loss`?
- Does guidance weight λ=0.1 bias the loss scale?
- How to balance noise prediction accuracy vs guidance constraints?

### 5. Validation vs Training Loss

**Question**: What is the expected relationship between training MSE and downstream performance?

**Validation metrics**:
- Information Coefficient (IC): Correlation between denoised features and forward returns
- Adjusted Sharpe Ratio: Risk-adjusted trading performance

**Observed phenomenon**:
- Denoising helps tree models (+2-4% Sharpe) but hurts linear models (-40% IC)
- Could this be related to training loss threshold?

---

## Inference Method (Single-Step)

```python
def denoise_single_window(window, model, sde):
    # 1. Normalize window (instance norm)
    window_norm = (window - window.mean()) / (window.std() + 1e-6)

    # 2. Treat as noisy observation at t=500
    t = 500  # Middle timestep

    # 3. Predict noise
    ε_pred = model(window_norm, t)

    # 4. Recover clean signal
    x_0 = (window_norm - √(1 - α_bar_500) * ε_pred) / √(α_bar_500)

    # 5. Denormalize
    x_0_denorm = x_0 * window.std() + window.mean()

    return x_0_denorm[-1, :]  # Return last row only (causal)
```

**Note**: Uses **single-step denoising** (not iterative DDPM Algorithm 2)

---

## Training Convergence Observations

**From preliminary training**:
- Epoch 1: Loss ≈ 1.5-1.8
- Epoch 5: Loss ≈ 0.8-1.2
- Epoch 10: Loss ≈ 0.4-0.6
- Epoch 20: Loss ≈ 0.2-0.3
- Epoch 50: Loss ≈ 0.05-0.1
- Epoch 100: Loss ≈ 0.02-0.03

---

## Expected Output

Please provide:

1. **Theoretical Analysis**:
   - Formula-based analysis for optimal MSE threshold
   - Relationship to VP-SDE parameters (T, β schedule, normalization)
   - Mathematical bounds or theoretical limits

2. **Literature Survey**:
   - Loss values from DDPM/Score-SDE papers
   - Reported convergence criteria and early stopping strategies
   - Domain-specific applications (financial data, time series)

3. **Loss Threshold Analysis**:
   - Multiple candidate threshold values with theoretical justification
   - Trade-offs for each threshold level
   - Expected number of epochs to reach each threshold

4. **Validation Strategy**:
   - How to validate if a chosen threshold is appropriate
   - Metrics to monitor beyond MSE loss
   - Relationship between training loss and downstream task performance

---

## References

**Papers to consider**:
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
- Song et al. (2021): "Score-Based Generative Modeling through SDEs"
- Any financial time series denoising papers using diffusion models

**Implementation reference**:
- This project uses VP-SDE from Song et al. (2021) framework
- Instance normalization for scale-invariance (not typical in DDPM papers)

---

## Constraints

- Training budget: 30 GPU hours/week (Kaggle)
- 5 clusters × N epochs per cluster
- Trade-off: Training time vs denoising quality

Thank you for your research assistance!
