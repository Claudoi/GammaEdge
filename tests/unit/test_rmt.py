import numpy as np

from portfolio.optim.robust import clean_covariance_rmt, marcenko_pastur_limits


def test_marcenko_pastur_limits():
    # Theoretical limits for random matrix
    T, N = 1000, 100
    l_min, l_max = marcenko_pastur_limits(T, N, var_eps=1.0)

    # Q = 10 -> [ (1-sqrt(0.1))^2, (1+sqrt(0.1))^2 ]
    #          [ (1-0.316)^2, (1+0.316)^2 ]
    #          [ 0.68^2, 1.316^2 ] ~ [0.46, 1.73]

    assert l_min < l_max
    assert l_min > 0.4
    assert l_max < 2.0


def test_clean_covariance_rmt_noise_reduction():
    # Generate pure noise matrix (random normal returns)
    np.random.seed(42)
    T, N = 2000, 50
    ret_noise = np.random.normal(0, 1, size=(T, N))

    # Sample Covariance
    Sigma_noisy = np.cov(ret_noise, rowvar=False)

    # Clean it using RMT
    # Since it is pure noise (identity true cov), RMT should effectively set it to Identity-like
    # or shrink eigenvalues significantly towards 1.0.
    Sigma_clean = clean_covariance_rmt(Sigma_noisy, T, N)

    # Check if cleaned matrix is closer to Identity than noisy one?
    # Or check Condition Number.
    # Noisy random matrix has spreading eigenvalues. Cleaned one should have clumped ones.

    cond_noisy = np.linalg.cond(Sigma_noisy)
    cond_clean = np.linalg.cond(Sigma_clean)

    # Condition number should IMPROVE (decrease) significantly
    msg = f"Cleaning should improve conditioning: {cond_clean} vs {cond_noisy}"
    assert cond_clean < cond_noisy, msg

    # In pure noise case, MP cleaning replaces all noise (all evals) with mean.
    # So it should be very close to diagonal * mean_var.

    off_diag_noisy = Sigma_noisy - np.diag(np.diag(Sigma_noisy))
    off_diag_clean = Sigma_clean - np.diag(np.diag(Sigma_clean))

    rms_noisy = np.sqrt(np.mean(off_diag_noisy**2))
    rms_clean = np.sqrt(np.mean(off_diag_clean**2))

    # Clean matrix should have near-zero correlations for pure noise input
    msg = f"Clean matrix should have lower off-diagonal noise: {rms_clean} vs {rms_noisy}"
    assert rms_clean < rms_noisy, msg


def test_clean_covariance_rmt_signal_preservation():
    # Generate Signal + Noise
    # 1 Dominant factor (market)
    np.random.seed(42)
    T, N = 1000, 20

    # Market factor
    mkt = np.random.normal(0, 0.02, size=(T, 1))
    # Betas ~ U[0.5, 1.5]
    betas = np.random.uniform(0.5, 1.5, size=(1, N))

    # Idiosyncratic noise
    noise = np.random.normal(0, 0.01, size=(T, N))

    rets = mkt @ betas + noise

    Sigma_sample = np.cov(rets, rowvar=False)

    # Clean
    Sigma_clean = clean_covariance_rmt(Sigma_sample, T, N)

    # The cleaned matrix should preserved the market structure (high correlation)
    # Average correlation of cleaned should be close to sample

    corr_sample = np.corrcoef(rets, rowvar=False)
    corr_clean = Sigma_clean / np.outer(
        np.sqrt(np.diag(Sigma_clean)), np.sqrt(np.diag(Sigma_clean))
    )

    avg_corr_s = np.mean(corr_sample)
    avg_corr_c = np.mean(corr_clean)

    # Should not destroy the signal
    assert abs(avg_corr_c - avg_corr_s) < 0.1
