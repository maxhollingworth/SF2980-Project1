import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Parameters
n_simulations = 10000
nSimList=[1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,80000,100000]
std_dev = 0.01  # Standard deviation for log-returns
dof_marginals = 3  # Degrees of freedom for the t-distribution (t_3)
initial_investment = 1_000_000  # All money invested in a single asset


# Determine scale parameter for t_3 marginals
raw_std = np.sqrt(dof_marginals / (dof_marginals - 2))  # Raw standard deviation of t_3
scale_param = std_dev / raw_std

var_values = []
es_values = []
for n_simulations in nSimList:

    # Step 1: Generate independent t_3-distributed log-returns
    log_returns = np.random.standard_t(dof_marginals, size=n_simulations) * scale_param

    # Step 2: Simulate portfolio value at time t=1
    V1 = initial_investment * np.exp(log_returns)  # Future portfolio values

    # Compute log-returns for the portfolio
    portfolio_log_returns = np.log(V1 / initial_investment)

    # Compute VaR (1st percentile of returns)
    portfolio_returns = V1 - initial_investment
    var_01 = np.percentile(portfolio_returns, 1)
    var_values.append(var_01)

    # Filter returns to calculate Expected Shortfall (ES)
    losses_below_var = portfolio_returns[portfolio_returns <= var_01]
    es_01 = np.mean(losses_below_var)
    es_values.append(es_01)

    # Print results
    print(f"VaR 0.01: {var_01}")
    print(f"ES 0.01: {es_01}")



# Plot stability of VaR and ES
plt.figure(figsize=(10, 6))
plt.plot(nSimList, var_values, marker='o', label='VaR 0.01')
plt.plot(nSimList, es_values, marker='o', label='ES 0.01', linestyle='--')
plt.title('Stability of VaR and ES with Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()




