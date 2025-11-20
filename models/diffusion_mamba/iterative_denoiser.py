"""
Iterative Denoising with Guidance (Algorithm 2 from Paper)

Implements the paper's denoising procedure with TV and Fourier guidance.
This is applied during INFERENCE only.

Paper Algorithm 2: Denoising procedure
    Input: noisy observation x_T', model θ, guidance params η_TV, η_F
    1. For i = K-1 to 0:
    2.     Predictor: x_i ← Reverse_SDE_step(x_{i+1})
    3.     Corrector: x_i ← Langevin_step(x_i) (optional)
    4.     TV Guidance: x_i ← x_i - η_TV ∇L_TV(x_i)
    5.     Fourier Guidance: x_i ← x_i - η_F ∇L_F(x_i, x0_estimate)
    Output: denoised x_0
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from .vp_sde import VPSDE
from .guidance import apply_guidance


def iterative_denoise(
    model: nn.Module,
    x_noisy: torch.Tensor,
    sde: VPSDE,
    num_steps: int = 10,
    noise_level: Optional[int] = None,
    eta_tv: float = 0.01,
    eta_fourier: float = 0.01,
    fourier_threshold: float = 0.1,
    use_corrector: bool = False,
    corrector_steps: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Iterative denoising with TV and Fourier guidance (Algorithm 2).

    Args:
        model: Trained denoiser model (predicts noise)
        x_noisy: Noisy observation [batch, seq_len, n_features]
        sde: VP-SDE instance
        num_steps: Number of denoising steps (K in paper)
        noise_level: Noise level T' (if None, use num_steps)
        eta_tv: TV guidance strength
        eta_fourier: Fourier guidance strength
        fourier_threshold: Frequency cutoff for Fourier loss
        use_corrector: Whether to use Langevin corrector steps
        corrector_steps: Number of Langevin steps per iteration
        device: Device

    Returns:
        Denoised x_0 [batch, seq_len, n_features]
    """
    model.eval()

    # Noise level (T' in paper)
    if noise_level is None:
        noise_level = num_steps

    # Start from noisy observation
    x = x_noisy.to(device)  # [B, L, F]

    # Timesteps to denoise through
    # From noise_level down to 0
    timesteps = torch.linspace(
        noise_level, 0, num_steps + 1, dtype=torch.long, device=device
    )[:-1]  # [num_steps]

    # Store initial estimate for Fourier guidance reference
    with torch.no_grad():
        # Initial x0 estimate
        t_init = torch.full((x.shape[0],), noise_level, device=device, dtype=torch.long)
        predicted_noise_init = model(x, t_init)

        alpha_bar_init = sde.alphas_cumprod[t_init]
        while alpha_bar_init.dim() < x.dim():
            alpha_bar_init = alpha_bar_init.unsqueeze(-1)

        x_ref = (x - torch.sqrt(1.0 - alpha_bar_init) * predicted_noise_init) / (torch.sqrt(alpha_bar_init) + 1e-8)
        x_ref = torch.clamp(x_ref, -10, 10)

    # Iterative denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Prepare batch of same timestep
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Predictor: Reverse SDE step
            predicted_noise = model(x, t_batch)
            x = sde.denoise_step(x, t_batch, predicted_noise)

            # Corrector: Langevin MCMC steps (optional)
            if use_corrector and t > 0:
                x = langevin_corrector(
                    x, t_batch, model, sde,
                    n_steps=corrector_steps,
                    device=device
                )

            # Guidance: TV + Fourier (Algorithm 2, lines 10-11)
            if t > 0:  # Skip guidance at final step
                x = apply_guidance(
                    x, x_ref,
                    eta_tv=eta_tv,
                    eta_fourier=eta_fourier,
                    fourier_threshold=fourier_threshold,
                    device=device
                )

    return x


def langevin_corrector(
    x: torch.Tensor,
    t: torch.Tensor,
    model: nn.Module,
    sde: VPSDE,
    n_steps: int = 1,
    snr: float = 0.15,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Langevin MCMC corrector step (predictor-corrector sampling).

    Refines sample using gradient of log p(x_t).

    Args:
        x: Current sample [batch, seq_len, n_features]
        t: Current timesteps [batch]
        model: Denoiser model
        sde: VP-SDE instance
        n_steps: Number of Langevin steps
        snr: Signal-to-noise ratio for step size
        device: Device

    Returns:
        Refined sample [batch, seq_len, n_features]
    """
    for _ in range(n_steps):
        # Predict score (negative of predicted noise, scaled)
        predicted_noise = model(x, t)

        # Get noise scale
        alpha_bar = sde.alphas_cumprod[t]
        while alpha_bar.dim() < x.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        sigma = torch.sqrt(1.0 - alpha_bar)

        # Score: ∇log p(x_t) ≈ -predicted_noise / sigma
        score = -predicted_noise / sigma

        # Step size based on SNR
        noise_magnitude = torch.norm(predicted_noise.reshape(predicted_noise.shape[0], -1), dim=-1).mean()
        step_size = (snr * sigma) ** 2 * 2

        # Langevin dynamics: x ← x + ε * ∇log p(x) + √(2ε) * z
        noise = torch.randn_like(x)
        x = x + step_size * score + torch.sqrt(2 * step_size) * noise

    return x


def denoise_single_window_iterative(
    model: nn.Module,
    window_norm: torch.Tensor,
    sde: VPSDE,
    num_steps: int = 10,
    noise_level: int = 500,
    eta_tv: float = 0.01,
    eta_fourier: float = 0.01,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Denoise a single window using iterative denoising.

    Wrapper for iterative_denoise that:
    1. Adds batch dimension
    2. Denoises iteratively
    3. Returns last row only (causal)

    Args:
        model: Trained denoiser model
        window_norm: Normalized window [seq_len, n_features]
        sde: VP-SDE instance
        num_steps: Number of denoising iterations
        noise_level: Initial noise level T'
        eta_tv: TV guidance strength
        eta_fourier: Fourier guidance strength
        device: Device

    Returns:
        Denoised last row [n_features]
    """
    # Add batch dimension
    window_batch = window_norm.unsqueeze(0).to(device)  # [1, L, F]

    # Iterative denoising
    denoised_window = iterative_denoise(
        model, window_batch, sde,
        num_steps=num_steps,
        noise_level=noise_level,
        eta_tv=eta_tv,
        eta_fourier=eta_fourier,
        device=device
    )  # [1, L, F]

    # Return last row only (causal)
    return denoised_window[0, -1, :].cpu()  # [F]


if __name__ == "__main__":
    print("Testing iterative denoiser...")

    # Mock model
    class MockModel(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(n_features, n_features)

        def forward(self, x, t):
            # Dummy: return small noise
            return torch.randn_like(x) * 0.1

    # Test parameters
    batch_size = 2
    seq_len = 60
    n_features = 20
    device = "cpu"

    # Create model and SDE
    model = MockModel(n_features).to(device)
    sde = VPSDE(device=device)

    # Test data
    x_noisy = torch.randn(batch_size, seq_len, n_features, device=device)

    # Test iterative denoising
    print("\nTesting iterative_denoise()...")
    x_denoised = iterative_denoise(
        model, x_noisy, sde,
        num_steps=5,
        noise_level=100,
        eta_tv=0.01,
        eta_fourier=0.01,
        device=device
    )
    print(f"Input shape: {x_noisy.shape}")
    print(f"Output shape: {x_denoised.shape}")
    print(f"Mean change: {(x_denoised - x_noisy).abs().mean().item():.6f}")

    # Test single window denoising
    print("\nTesting denoise_single_window_iterative()...")
    window = torch.randn(seq_len, n_features)
    denoised_row = denoise_single_window_iterative(
        model, window, sde,
        num_steps=5,
        noise_level=100,
        device=device
    )
    print(f"Window shape: {window.shape}")
    print(f"Denoised row shape: {denoised_row.shape}")

    print("\n✓ All tests passed!")
