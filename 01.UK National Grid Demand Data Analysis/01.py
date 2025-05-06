# %% [markdown]
# # Part 1: Python & Data Wrangling Mini-Project - UK Electricity Demand Analysis
#
# **Objective:** Load, clean, explore, and visualize historic UK electricity demand data from National Grid ESO.
#
# **Libraries:** pandas, numpy, matplotlib, seaborn

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size

# %% [markdown]
# ## 1. Data Loading
#
# Load the dataset. Ensure the file path is correct. We expect datetime information and demand values.

# %%
# --- Parameters ---
data_file_path = 'demanddata_2024.csv'

# --- Load Data ---
try:
    # National Grid data often uses specific datetime formats and column names.
    df_raw = pd.read_csv(data_file_path)
    print(f"Successfully loaded data from: {data_file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {data_file_path}")
    print("Please update the 'data_file_path' variable.")
    # Exit or raise error if file not found, or handle appropriately
    df_raw = pd.DataFrame() # Create empty df to avoid downstream errors if you want to continue structure checks

if not df_raw.empty:
    print("Raw Data Info:")
    df_raw.info()
    print("\nRaw Data Head:")
    print(df_raw.head())
    print("\nRaw Data Tail:")
    print(df_raw.tail())


# %% [markdown]
# ## 2. Data Cleaning and Preprocessing
#
# - Combine date and settlement period into a proper datetime index.
# - Select relevant columns.
# - Handle missing values.
# - Check data types.

# %%
if not df_raw.empty:
    df = df_raw.copy() # Work on a copy

    # --- Datetime Creation ---
    # National Grid data often has Date and Settlement Period (1-48/50 for half-hours)
    # We need to convert this into a proper timestamp.
    # Settlement period 1 is 00:00-00:30, Period 48 is 23:30-00:00
    # Note: This conversion logic might need slight adjustment based on exact data format/clock changes

    try:
        # Ensure SETTLEMENT_DATE is parsed as datetime
        df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'], dayfirst=True) # Adjust dayfirst=True/False based on CSV format

        # Calculate timedelta based on settlement period
        # Period 1 starts at 00:00, Period 2 at 00:30, ..., Period 48 at 23:30
        # timedelta = (period - 1) * 30 minutes
        df['timedelta'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')

        # Create the timestamp (represents the START of the half-hour period)
        df['timestamp'] = df['SETTLEMENT_DATE'] + df['timedelta']

        # --- Set Index ---
        df = df.set_index('timestamp')

        # --- Select and Rename Columns ---
        # Keep only the relevant demand column.
        demand_col = 'ND' # National Demand
        if demand_col not in df.columns:
             # Try common alternatives if 'ND' isn't found
             potential_cols = ['NATIONAL_DEMAND', 'TND', 'ENGLAND_WALES_DEMAND']
             for col in potential_cols:
                 if col in df.columns:
                     demand_col = col
                     print(f"Using demand column: {demand_col}")
                     break
             else: # If no alternative found
                 raise KeyError(f"Demand column '{demand_col}' (and alternatives) not found in DataFrame. Available columns: {df.columns.tolist()}")


        df = df[[demand_col]].copy() # Select only the demand column
        df.rename(columns={demand_col: 'Demand_MW'}, inplace=True) # Rename for clarity

        # --- Sort Index ---
        # Data should already be roughly sorted, but good practice to ensure.
        df = df.sort_index()

        # --- Handle Missing Values ---
        print("\nMissing values before handling:")
        print(df.isnull().sum())

        # Strategy: Forward fill (common for time series) or Interpolate
        df['Demand_MW'] = df['Demand_MW'].fillna(method='ffill')
        # Alternative: df['Demand_MW'] = df['Demand_MW'].interpolate(method='time')

        print("\nMissing values after handling:")
        print(df.isnull().sum())

        # --- Check Data Types ---
        print("\nData types after cleaning:")
        print(df.dtypes)

        print("\nCleaned Data Head:")
        print(df.head())

    except KeyError as e:
        print(f"\nError during cleaning: {e}")
        print("Please check the column names ('SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND' or alternatives) in your CSV file and update the code.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during cleaning: {e}")

else:
    print("\nSkipping cleaning and EDA as data loading failed.")


# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA) & Visualization

# %%
if not df.empty and 'Demand_MW' in df.columns: # Proceed only if cleaning was successful

    # --- Basic Statistics ---
    print("\nDescriptive Statistics for Demand (MW):")
    print(df['Demand_MW'].describe())

    # --- Time Series Plot (Overall Trend) ---
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Demand_MW'])
    plt.title('UK National Demand Over Time (Half-Hourly)')
    plt.xlabel('Timestamp')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.show()

    # --- Resampling to Daily Average Demand ---
    df_daily = df['Demand_MW'].resample('D').mean() # Calculate mean daily demand

    plt.figure(figsize=(14, 7))
    plt.plot(df_daily.index, df_daily)
    plt.title('Average Daily UK National Demand')
    plt.xlabel('Date')
    plt.ylabel('Average Demand (MW)')
    plt.grid(True)
    plt.show()

    # --- Distribution of Demand ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Demand_MW'], kde=True, bins=50)
    plt.title('Distribution of Half-Hourly Demand')
    plt.xlabel('Demand (MW)')
    plt.ylabel('Frequency')
    plt.show()

    # --- Rolling Average (e.g., 7-day rolling mean of daily data) ---
    df_daily_rolling = df_daily.rolling(window=7).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(df_daily.index, df_daily, label='Daily Average Demand', alpha=0.6)
    plt.plot(df_daily_rolling.index, df_daily_rolling, label='7-Day Rolling Average', color='red')
    plt.title('Daily Demand with 7-Day Rolling Average')
    plt.xlabel('Date')
    plt.ylabel('Average Demand (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Seasonality Check (e.g., Box plot by Month) ---
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly.index.strftime('%Y-%m') # Use YYYY-MM for sorting
    df_monthly['Month_Name'] = df_monthly.index.month_name() # For display
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_monthly['Month_Name'] = pd.Categorical(df_monthly['Month_Name'], categories=month_order, ordered=True)


    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Month_Name', y='Demand_MW', data=df_monthly)
    plt.title('Distribution of Demand by Month')
    plt.xlabel('Month')
    plt.ylabel('Demand (MW)')
    plt.xticks(rotation=45)
    plt.show()

    # --- Seasonality Check (e.g., Box plot by Day of Week) ---
    df_dow = df.copy()
    df_dow['DayOfWeek'] = df_dow.index.day_name()
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_dow['DayOfWeek'] = pd.Categorical(df_dow['DayOfWeek'], categories=day_order, ordered=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='DayOfWeek', y='Demand_MW', data=df_dow)
    plt.title('Distribution of Demand by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Demand (MW)')
    plt.show()


# %% [markdown]
# ## 4. Mini-Project Conclusion & Potential Next Steps
#
# - We have successfully loaded, cleaned, and performed basic EDA on the UK National Grid demand data.
# - Key findings include [mention 1-2 key observations, e.g., clear seasonal pattern with higher demand in winter, distinct weekly pattern with lower demand on weekends, typical range of demand values].
# - **Potential Next Steps:**
#     - Deeper dive into anomalies or specific events.
#     - Feature engineering (e.g., adding lag features, time-based features like hour of day).
#     - Correlation with other factors (e.g., temperature, generation mix - requires merging other datasets).
#     - Time series forecasting.

# %%
print("\nPart 1 Mini-Project script finished.")