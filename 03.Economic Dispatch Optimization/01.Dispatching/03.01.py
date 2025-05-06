# %% [markdown]
# # Part 3: Mini-Project 1 - Simple Economic Dispatch

# %%
import pulp
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## 1. Define Problem Data

# %%
generators = {
    'G1': {'min_power': 10, 'max_power': 50, 'cost': 20},
    'G2': {'min_power': 20, 'max_power': 100, 'cost': 30},
}
total_demand = 110  # MW

# %% [markdown]
# ## 2. Create LP Problem

# %%
# Create the LP problem object (minimize cost)
prob_ed = pulp.LpProblem("Economic_Dispatch", pulp.LpMinimize)

# %% [markdown]
# ## 3. Define Decision Variables

# %%
# Power output for each generator (continuous variables)
# Format: LpVariable(name, lowBound, upBound, cat='Continuous')
gen_power = pulp.LpVariable.dicts(
    "GenPower",
    generators.keys(),
    lowBound=0, # Initial lower bound is 0
    cat='Continuous'
)

# Apply specific min/max bounds from data
for gen_name in generators:
    gen_power[gen_name].lowBound = generators[gen_name]['min_power']
    gen_power[gen_name].upBound = generators[gen_name]['max_power']

print("Decision Variables:")
print(gen_power)

# %% [markdown]
# ## 4. Define Objective Function

# %%
# Minimize total generation cost: sum(cost_i * power_i)
prob_ed += pulp.lpSum(generators[gen_name]['cost'] * gen_power[gen_name] for gen_name in generators), "Total_Generation_Cost"

print("\nObjective Function:")
print(prob_ed.objective)

# %% [markdown]
# ## 5. Define Constraints

# %%
# Constraint 1: Power Balance (Total Generation = Total Demand)
prob_ed += pulp.lpSum(gen_power[gen_name] for gen_name in generators) == total_demand, "Power_Balance"

# Min/Max power constraints are set in the variable definitions (lowBound/upBound)

print("\nConstraints:")
# PuLP doesn't easily print all constraints like this, but we know the power balance is added.
# We can check the problem object structure:
# print(prob_ed) # Uncomment to see full LP formulation if desired

# %% [markdown]
# ## 6. Solve the Problem

# %%
print("\nSolving...")
# Use the default CBC solver bundled with PuLP
prob_ed.solve()

# %% [markdown]
# ## 7. Display Results

# %%
print("\n--- Results ---")
print(f"Status: {pulp.LpStatus[prob_ed.status]}")

if pulp.LpStatus[prob_ed.status] == 'Optimal':
    print(f"Optimal Total Cost: £{pulp.value(prob_ed.objective):.2f}")
    print("\nOptimal Generation Dispatch (MW):")
    results = {}
    for gen_name in generators:
        power = gen_power[gen_name].varValue
        results[gen_name] = {'Power (MW)': power, 'Cost (£/MWh)': generators[gen_name]['cost']}
        print(f"  {gen_name}: {power:.2f} MW")

    # --- Visualize ---
    results_df = pd.DataFrame.from_dict(results, orient='index')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of dispatch
    results_df['Power (MW)'].plot(kind='bar', ax=ax[0], title='Optimal Generation Dispatch')
    ax[0].set_ylabel('Power (MW)')
    ax[0].set_xlabel('Generator')
    ax[0].grid(axis='y')

    # Cost curve visualization (conceptual)
    costs = sorted(generators.items(), key=lambda item: item[1]['cost'])
    cumulative_mw = 0
    x_mw = [0]
    y_cost = [0] # Start at 0 cost for 0 MW
    for name, data in costs:
         dispatch = gen_power[name].varValue
         # Add segment for this generator's dispatch at its cost
         if dispatch > 0.01: # Only plot if dispatched significantly
              x_mw.append(cumulative_mw)
              y_cost.append(data['cost']) # Step up cost just before adding MW
              cumulative_mw += dispatch
              x_mw.append(cumulative_mw)
              y_cost.append(data['cost']) # Maintain cost level

    ax[1].step(x_mw, y_cost, where='post', label='Dispatch Stack')
    ax[1].set_title('Conceptual Merit Order / Cost Stack')
    ax[1].set_xlabel('Cumulative Power (MW)')
    ax[1].set_ylabel('Marginal Cost (£/MWh)')
    ax[1].axvline(total_demand, color='r', linestyle='--', label=f'Demand = {total_demand} MW')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

else:
    print("Could not find an optimal solution.")