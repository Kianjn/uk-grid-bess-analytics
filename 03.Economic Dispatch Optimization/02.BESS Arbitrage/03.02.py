# %% [markdown]
# # Part 3: Mini-Project 2 - Basic BESS Arbitrage

# %%
import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. Define Problem Data

# %%
# Time horizon
T = 24
hours = list(range(T))
dt = 1 # hour time step

# Price Vector (£/MWh) - Example: Fluctuating prices
np.random.seed(42) # for reproducibility
prices = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, T)) + np.random.normal(0, 5, T)
prices = np.maximum(5, prices) # Ensure prices are positive

# Battery Parameters
soc_max = 10      # MWh (Max energy stored)
soc_min = 1       # MWh (Min energy stored)
p_max_chg = 5     # MW (Max charge rate)
p_max_dis = 5     # MW (Max discharge rate)
eff_chg = 0.90    # [-] (Charge efficiency)
eff_dis = 0.90    # [-] (Discharge efficiency)
soc_initial = 5   # MWh (Starting energy)
soc_final = 5     # MWh (Target ending energy)

# %% [markdown]
# ## 2. Create LP Problem

# %%
# Create the LP problem object (maximize profit = minimize -profit)
prob_bess = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)

# %% [markdown]
# ## 3. Define Decision Variables

# %%
# Charge Power (MW) for each hour
p_charge = pulp.LpVariable.dicts(
    "P_Charge",
    hours,
    lowBound=0,
    upBound=p_max_chg,
    cat='Continuous'
)

# Discharge Power (MW) for each hour
p_discharge = pulp.LpVariable.dicts(
    "P_Discharge",
    hours,
    lowBound=0,
    upBound=p_max_dis,
    cat='Continuous'
)

# State of Charge (MWh) for each hour (including hour T for final state)
soc = pulp.LpVariable.dicts(
    "SoC",
    list(range(T + 1)), # Index 0 to T (inclusive)
    lowBound=soc_min,
    upBound=soc_max,
    cat='Continuous'
)

print("Decision Variables (examples):")
print(p_charge[0], p_discharge[0], soc[0])


# %% [markdown]
# ## 4. Define Objective Function

# %%
# Maximize Profit: Sum over hours [ Price * (Power_Discharged * eff_dis - Power_Charged / eff_chg) * dt ]
# Note: PuLP requires efficiency applied correctly depending on variable definition.
# If P_Charge is power into the grid connection point, cost is Price * P_Charge * dt
# If P_Discharge is power out of grid connection point, revenue is Price * P_Discharge * dt
# The energy change IN the battery is P_Charge * eff_chg and P_Discharge / eff_dis.
# Let's define profit from the grid perspective: Revenue (Discharge) - Cost (Charge)

profit = pulp.lpSum( prices[t] * (p_discharge[t] - p_charge[t]) * dt for t in hours )
prob_bess += profit, "Total_Profit"

print("\nObjective Function:")
print(prob_bess.objective)


# %% [markdown]
# ## 5. Define Constraints

# %%
print("\nAdding Constraints...")

# Constraint 1: Initial State of Charge
prob_bess += soc[0] == soc_initial, "Initial_SoC"

# Constraint 2: Final State of Charge
prob_bess += soc[T] >= soc_final, "Final_SoC"
# Or use == if exact final SoC is needed:
# prob_bess += soc[T] == soc_final, "Final_SoC"

# Constraint 3: SoC dynamics (Energy Balance)
# SoC[t+1] = SoC[t] + (Charge_Power_effective * dt) - (Discharge_Power_effective * dt)
# Charge_Power_effective = p_charge[t] * eff_chg (energy reaching battery)
# Discharge_Power_effective = p_discharge[t] / eff_dis (energy drawn from battery)

for t in hours:
    prob_bess += soc[t+1] == soc[t] + (p_charge[t] * eff_chg * dt) - (p_discharge[t] * (1.0 / eff_dis) * dt), f"SoC_Update_{t}"

# Note: Power limits and SoC limits are already defined in the LpVariable definitions.

# Optional Constraint: Prevent simultaneous charge and discharge (often unnecessary with LP objective, but can be added)
# Using binary variables and Big-M (makes it a MILP):
# use_charge = pulp.LpVariable.dicts("Use_Charge", hours, cat='Binary')
# use_discharge = pulp.LpVariable.dicts("Use_Discharge", hours, cat='Binary')
# M = p_max_chg # A sufficiently large number
# for t in hours:
#   prob_bess += p_charge[t] <= M * use_charge[t]
#   prob_bess += p_discharge[t] <= M * use_discharge[t]
#   prob_bess += use_charge[t] + use_discharge[t] <= 1 # Can only do one or the other
# We will omit this for simplicity for now, the LP should naturally avoid it if eff < 1

print("Constraints added.")

# %% [markdown]
# ## 6. Solve the Problem

# %%
print("\nSolving...")
prob_bess.solve()

# %% [markdown]
# ## 7. Display Results

# %%
print("\n--- Results ---")
print(f"Status: {pulp.LpStatus[prob_bess.status]}")

if pulp.LpStatus[prob_bess.status] == 'Optimal':
    optimal_profit = pulp.value(prob_bess.objective)
    print(f"Optimal Total Profit: £{optimal_profit:.2f}")

    # Extract results into a DataFrame for plotting
    results_data = {'Hour': hours}
    results_data['Price (£/MWh)'] = prices
    results_data['Charge Power (MW)'] = [p_charge[t].varValue for t in hours]
    results_data['Discharge Power (MW)'] = [p_discharge[t].varValue for t in hours]
    # Add SoC at the START of the hour
    results_data['SoC (MWh)'] = [soc[t].varValue for t in hours]

    results_df = pd.DataFrame(results_data)

    # --- Visualize ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot Prices
    ax[0].plot(results_df['Hour'], results_df['Price (£/MWh)'], label='Hourly Price', color='black', marker='o', linestyle='-')
    ax[0].set_ylabel('Price (£/MWh)')
    ax[0].set_title('BESS Optimal Dispatch Strategy')
    ax[0].grid(True)
    ax[0].legend()

    # Plot Charge/Discharge Power
    ax[1].bar(results_df['Hour'], results_df['Charge Power (MW)'], label='Charge Power', color='blue', alpha=0.7, width=0.4, align='edge')
    # Negate discharge power for visual clarity on the same plot
    ax[1].bar(results_df['Hour'], -results_df['Discharge Power (MW)'], label='Discharge Power', color='red', alpha=0.7, width=-0.4, align='edge')
    ax[1].set_ylabel('Power (MW)')
    ax[1].axhline(0, color='black', linewidth=0.5)
    ax[1].legend()
    ax[1].grid(True)

    # Plot State of Charge
    # Include the final SoC at hour T for a complete picture
    soc_plot = [soc[t].varValue for t in range(T + 1)]
    hours_plot = list(range(T + 1))
    ax[2].plot(hours_plot, soc_plot, label='State of Charge (SoC)', color='green', marker='.')
    ax[2].set_ylabel('SoC (MWh)')
    ax[2].set_xlabel('Hour')
    ax[2].axhline(soc_min, color='grey', linestyle='--', label='Min SoC')
    ax[2].axhline(soc_max, color='grey', linestyle='--', label='Max SoC')
    ax[2].set_ylim(0, soc_max * 1.1) # Adjust ylim for visibility
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary of actions
    print("\nDispatch Summary:")
    print(results_df[['Hour', 'Price (£/MWh)', 'Charge Power (MW)', 'Discharge Power (MW)', 'SoC (MWh)']].round(2))

else:
    print("Could not find an optimal solution.")