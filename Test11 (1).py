import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from matplotlib.ticker import FuncFormatter
from scipy.stats import gamma

# Parameters
n_assets = 50
n_simulations = 100000
initial_investment = 20000
V0 = n_assets * initial_investment

# Corrected scale parameter ( Sqrt(0,01^2/3))
#scale_param = 0.0057735
scale_param = np.sqrt((0.01**2)/3)
print(scale_param)

# Degrees of freedom for marginal t-distribution
df_marginal = 3

# Copula parameters
tau = 0.4
rho = np.sin(np.pi * tau / 2)  # Convert Kendall's tau to Pearson's rho, 2*exp/6
theta_clayton = 2 * tau / (1 - tau)  # Clayton copula parameter

# Correlation matrix for Gaussian and t-copula
correlation_matrix = rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)
L = np.linalg.cholesky(correlation_matrix)  # Cholesky decomposition

def simulate_gaussian_copula(n_simulations=100000, seed=None):
    """Simulate portfolio using Gaussian copula."""
    if seed is not None:
        np.random.seed(seed)
    z = np.random.normal(size=(n_simulations, n_assets))  # Independent standard normal samples
    correlated_normals = z @ L.T  # Correlate using Cholesky decomposition
    correlated_t = t.ppf(norm.cdf(correlated_normals), df_marginal) * scale_param  # Transform to marginal t
    return correlated_t


def simulate_t4_copula(n_simulations=100000, seed=None):
    """Simulate portfolio using t4 copula."""
    if seed is not None:
        np.random.seed(seed)
    df_copula_t4 = 4  # Degrees of freedom for t4-copula
    z = np.random.standard_t(df_copula_t4, size=(n_simulations, n_assets))  # Independent t4 samples
    correlated_t4 = z @ L.T  # Correlate using Cholesky decomposition
    correlated_t = t.ppf(t.cdf(correlated_t4, df_copula_t4), df_marginal) * scale_param  # Transform to marginal t
    return correlated_t


def simulate_clayton_copula(n_simulations=100000, seed=None):
    """Simulate portfolio using Clayton copula."""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.uniform(size=(n_simulations, n_assets))  # Uniform samples
    Xc = np.random.gamma(1 / theta_clayton, 1, size=(n_simulations,))  # Gamma random variables
    V = (-np.log(U) / Xc[:, None] + 1) ** (-1 / theta_clayton)  # Clayton copula formula
    correlated_t = t.ppf(V, df_marginal) * scale_param  # Transform to marginal t
    return correlated_t


def calculate_portfolio_statistics(correlated_t, copula_name):
    """Calculate portfolio statistics given the copula type."""

    # Compute portfolio value tomorrow
    V1 = initial_investment * np.sum(np.exp(correlated_t), axis=1)  # Sum across assets

    # Log returns
    log_portfolio_returns = np.log(V1 / V0)

    # Define plot limits with added buffer
    v1_min, v1_max = np.percentile(V1, [0.1, 99.9])
    buffer = 0.05 * (v1_max - v1_min)  # Add 5% buffer
    v1_min -= buffer
    v1_max += buffer
    log_returns_min, log_returns_max = np.percentile(log_portfolio_returns, [0.1, 99.9])
    log_buffer = 0.05 * (log_returns_max - log_returns_min)
    log_returns_min -= log_buffer
    log_returns_max += log_buffer

    # Plot distributions
    plt.figure(figsize=(16, 8))

    # Portfolio value V1
    plt.subplot(1, 2, 1)
    plt.hist(V1, bins=50, range=(v1_min, v1_max), density=True, alpha=0.75, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.title(f"Portfolio Value (V1) Distribution with {copula_name} Copula", fontsize=14)
    plt.xlabel("Portfolio Value V1 (in $10^6$)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.2f}"))  # Add more decimals for precision

    # Portfolio log-returns
    plt.subplot(1, 2, 2)
    plt.hist(log_portfolio_returns, bins=50, range=(log_returns_min, log_returns_max), density=True, alpha=0.75, color='darkgreen', edgecolor='black', linewidth=0.5)
    plt.title(f"Portfolio Log-Return Distribution with {copula_name} Copula", fontsize=14)
    plt.xlabel("Log-Return (log(V1/V0))", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    summary_stats = {
        "V1 Mean": np.mean(V1),
        "V1 Std Dev": np.std(V1),
        "Log-Return Mean": np.mean(log_portfolio_returns),
        "Log-Return Std Dev": np.std(log_portfolio_returns),
        "Log-Return Skewness": np.mean((log_portfolio_returns - np.mean(log_portfolio_returns))**3) / np.std(log_portfolio_returns)**3,
        "Log-Return Kurtosis": np.mean((log_portfolio_returns - np.mean(log_portfolio_returns))**4) / np.std(log_portfolio_returns)**4,
    }

    # Print summary statistics
    print(f"Summary Statistics with {copula_name} Copula:")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")


# Simulate and analyze portfolios for each copula
copulas = {
    "Gaussian": simulate_gaussian_copula,
    "t4": simulate_t4_copula,
    "Clayton": simulate_clayton_copula,
}

for copula_name, simulate_function in copulas.items():
    correlated_t = simulate_function(n_simulations=100000, seed=42)
    calculate_portfolio_statistics(correlated_t, copula_name)