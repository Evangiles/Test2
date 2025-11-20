"""
Guidance Functions for Financial Time Series Denoising

Implements TV (Total Variation) and Fourier guidance as per paper Algorithm 2.
These are applied during INFERENCE only, not training.

Paper Reference:
- Equation 16: L_TV(x) = Σ|x[i+1] - x[i]|
- Equation 17: L_F(x_t, x) = ||FFT(x_t) - Filter(FFT(x), f)||²

Usage:
    # During iterative denoising (Algorithm 2, lines 10-11):
    tv_grad = compute_tv_gradient(x_i)
    fourier_grad = compute_fourier_gradient(x_i, x0_ref)
    x_i = x_i - eta_tv * tv_grad - eta_fourier * fourier_grad
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_tv_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of Total Variation loss.

    TV Loss (Equation 16):
        L_TV(x) = Σ_{i=1}^{L-1} |x[i+1] - x[i]|

    Gradient:
        ∇L_TV = [x[1]-x[0], x[2]-2*x[1]+x[0], ..., -x[L-1]+x[L-2]]
        (simplified for L1 norm)

    Args:
        x: Time series [batch, seq_len, n_features]

    Returns:
        Gradient [batch, seq_len, n_features]
    """
    batch_size, seq_len, n_features = x.shape

    # Compute finite differences
    diff = x[:, 1:, :] - x[:, :-1, :]  # [B, L-1, F]

    # Sign of differences (for L1 gradient)
    sign_diff = torch.sign(diff)  # [B, L-1, F]

    # Construct gradient
    grad = torch.zeros_like(x)  # [B, L, F]

    # Interior points: ∇[i] = sign(x[i]-x[i-1]) - sign(x[i+1]-x[i])
    grad[:, 0, :] = -sign_diff[:, 0, :]  # First point
    grad[:, 1:-1, :] = sign_diff[:, :-1, :] - sign_diff[:, 1:, :]  # Middle points
    grad[:, -1, :] = sign_diff[:, -1, :]  # Last point

    return grad


def compute_fourier_gradient(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    threshold: float = 0.1,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute gradient of Fourier loss.

    Fourier Loss (Equation 17):
        L_F(x_t, x) = ||FFT(x_t) - Filter(FFT(x), f)||²_2

    Where Filter keeps low frequencies (below threshold f) and zeros high frequencies.

    Gradient:
        ∇L_F = 2 * IFFT(FFT(x_t) - Filter(FFT(x_ref), f))
        (high-frequency components only)

    Args:
        x: Current denoising sample [batch, seq_len, n_features]
        x_ref: Reference signal (e.g., initial x0 estimate) [batch, seq_len, n_features]
        threshold: Frequency cutoff (0.1 = keep bottom 10% frequencies)
        device: Device for computation

    Returns:
        Gradient [batch, seq_len, n_features]
    """
    batch_size, seq_len, n_features = x.shape

    # Move to correct device
    x = x.to(device)
    x_ref = x_ref.to(device)

    # Reshape for batched FFT: [B, F, L]
    x_freq = x.transpose(1, 2)  # [B, F, L]
    x_ref_freq = x_ref.transpose(1, 2)  # [B, F, L]

    # Apply FFT along time dimension
    X_fft = torch.fft.rfft(x_freq, dim=-1)  # [B, F, L//2+1]
    X_ref_fft = torch.fft.rfft(x_ref_freq, dim=-1)  # [B, F, L//2+1]

    # Create low-pass filter
    freq_len = X_fft.shape[-1]
    cutoff_idx = int(freq_len * threshold)

    # Filter reference signal (keep low frequencies only)
    X_ref_filtered = X_ref_fft.clone()
    X_ref_filtered[:, :, cutoff_idx:] = 0  # Zero out high frequencies

    # Compute difference (x - filtered_ref)
    diff = X_fft - X_ref_filtered  # [B, F, L//2+1]

    # Only penalize high frequencies (above cutoff)
    diff[:, :, :cutoff_idx] = 0  # Keep low frequencies unchanged

    # Inverse FFT to get gradient
    grad_freq = torch.fft.irfft(diff, n=seq_len, dim=-1)  # [B, F, L]

    # Transpose back: [B, L, F]
    grad = grad_freq.transpose(1, 2)

    # Scale by 2 (from L2 loss derivative)
    grad = 2.0 * grad

    return grad


def apply_guidance(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    eta_tv: float = 0.01,
    eta_fourier: float = 0.01,
    fourier_threshold: float = 0.1,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Apply both TV and Fourier guidance to current sample.

    This is Algorithm 2, lines 10-11:
        x_i ← x_i - η_TV ∇L_TV(x_i)
        x_i ← x_i - η_F ∇L_F(x_i, x)

    Args:
        x: Current sample [batch, seq_len, n_features]
        x_ref: Reference signal for Fourier guidance
        eta_tv: TV guidance step size
        eta_fourier: Fourier guidance step size
        fourier_threshold: Frequency cutoff for Fourier loss
        device: Device

    Returns:
        Guided sample [batch, seq_len, n_features]
    """
    # Compute gradients
    tv_grad = compute_tv_gradient(x)
    fourier_grad = compute_fourier_gradient(x, x_ref, fourier_threshold, device)

    # Apply guidance
    x_guided = x - eta_tv * tv_grad - eta_fourier * fourier_grad

    return x_guided


if __name__ == "__main__":
    print("Testing guidance functions...")

    # Test data
    batch_size = 4
    seq_len = 60
    n_features = 20

    x = torch.randn(batch_size, seq_len, n_features)
    x_ref = torch.randn(batch_size, seq_len, n_features)

    # Test TV gradient
    tv_grad = compute_tv_gradient(x)
    print(f"TV gradient shape: {tv_grad.shape}")
    print(f"TV gradient mean: {tv_grad.mean().item():.6f}")
    print(f"TV gradient std: {tv_grad.std().item():.6f}")

    # Test Fourier gradient
    fourier_grad = compute_fourier_gradient(x, x_ref, threshold=0.1)
    print(f"\nFourier gradient shape: {fourier_grad.shape}")
    print(f"Fourier gradient mean: {fourier_grad.mean().item():.6f}")
    print(f"Fourier gradient std: {fourier_grad.std().item():.6f}")

    # Test guidance application
    x_guided = apply_guidance(x, x_ref, eta_tv=0.01, eta_fourier=0.01)
    print(f"\nGuided sample shape: {x_guided.shape}")
    print(f"Change magnitude: {(x_guided - x).abs().mean().item():.6f}")

    print("\n✓ All tests passed!")
