# %% [markdown]
# # Part 2: ML Pipeline & Scikit-Learn Mini-Project - Demand Prediction (Self-Contained)
#
# **Objective:** Load raw demand data, clean it, then build an ML pipeline to predict electricity demand using Linear Regression 
# and Decision Trees, incorporating feature engineering, scaling, training, evaluation, and basic hyperparameter tuning.
#
# **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size

# %% [markdown]
# ## 1. Data Loading (Raw Data)
#
# Load the raw dataset (e.g., for 2024). Ensure the file path is correct.

# %%
# --- Parameters ---
data_file_path = 'demanddata_2024.csv'

# --- Load Data ---
try:
    df_raw = pd.read_csv(data_file_path)
    print(f"Successfully loaded data from: {data_file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {data_file_path}")
    print("Please update the 'data_file_path' variable.")
    df_raw = pd.DataFrame() # Creates an empty df to allow script structure check

if not df_raw.empty:
    print("Raw Data Info:")
    df_raw.info()
    print("\nRaw Data Head:")
    print(df_raw.head())


# %% [markdown]
# ## 2. Data Cleaning and Preprocessing
#
# - Combine date and settlement period into a proper datetime index.
# - Select relevant columns.
# - Handle missing values.
# - Check data types.
# - This section replicates the cleaning from Part 1 miniproject. (01.Python)

# %%
df = pd.DataFrame() # Initialize empty DataFrame for cleaned data
if not df_raw.empty:
    df = df_raw.copy() # Work on a copy

    # --- Datetime Creation ---
    try:
        # Ensures SETTLEMENT_DATE is parsed as datetime
        df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'], dayfirst=True)

        # Calculates timedelta based on settlement period
        df['timedelta'] = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')

        # Creates the timestamp (represents the START of the half-hour period)
        df['timestamp'] = df['SETTLEMENT_DATE'] + df['timedelta']

        # --- Set Index ---
        df = df.set_index('timestamp')

        # --- Select and Rename Columns ---
        # Identify the demand column ('ND')
        demand_col = None
        potential_cols = ['ND', 'NATIONAL_DEMAND', 'TND', 'ENGLAND_WALES_DEMAND']
        for col in potential_cols:
            if col in df.columns:
                demand_col = col
                print(f"Identified demand column: {demand_col}")
                break
        if demand_col is None:
             raise KeyError(f"Could not find a recognized demand column. Available columns: {df.columns.tolist()}")

        df = df[[demand_col]].copy() # Select only the demand column
        df.rename(columns={demand_col: 'Demand_MW'}, inplace=True) # Rename for clarity

        # --- Sort Index ---
        df = df.sort_index()

        # --- Handle Duplicates (good practice for time series) ---
        df = df[~df.index.duplicated(keep='first')] # Keep first occurrence if any duplicate timestamps exist

        # --- Handle Missing Values ---
        print("\nMissing values before handling:")
        print(df.isnull().sum())
        # Strategy: Forward fill is often suitable for demand time series
        df['Demand_MW'] = df['Demand_MW'].fillna(method='ffill')
        # Optional: Backward fill remaining NaNs at the very beginning if ffill didn't catch them
        df['Demand_MW'] = df['Demand_MW'].fillna(method='bfill')
        print("\nMissing values after handling:")
        print(df.isnull().sum())

        # --- Check Data Types ---
        print("\nData types after cleaning:")
        print(df.dtypes)
        if not pd.api.types.is_numeric_dtype(df['Demand_MW']):
             print("Warning: Demand_MW column is not numeric. Attempting conversion.")
             df['Demand_MW'] = pd.to_numeric(df['Demand_MW'], errors='coerce')
             # Re-check and handle NaNs potentially introduced by coerce
             if df['Demand_MW'].isnull().any():
                  print("NaNs found after numeric conversion. Re-applying fill.")
                  df['Demand_MW'] = df['Demand_MW'].fillna(method='ffill').fillna(method='bfill')


        print("\nCleaned Data Head:")
        print(df.head())
        print(f"\nCleaned data shape: {df.shape}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")


    except KeyError as e:
        print(f"\nError during cleaning: Missing expected column - {e}")
        print("Please check the column names ('SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', demand column) in your CSV file.")
        df = pd.DataFrame() # Ensures df is empty on error
    except Exception as e:
         print(f"\nAn unexpected error occurred during cleaning: {e}")
         df = pd.DataFrame() # Ensures df is empty on error

else:
    print("\nSkipping cleaning as data loading failed.")


# %% [markdown]
# ## 3. Feature Engineering
#
# Create features (X) and target (y) from the cleaned data.
# - Target (y): Demand_MW at time 't'.
# - Features (X): Lagged demand, time-based features.

# %%
if not df.empty and 'Demand_MW' in df.columns:
    # Target variable is the current demand
    # Create target before modifying Demand_MW with shifts
    df['target'] = df['Demand_MW']

    # --- Create Lag Features ---
    lags = [1, 2, 3, 24, 48, 48*7] # t-30m, t-1h, t-1.5h, t-12h, t-1d, t-1w
    print("\nCreating lag features...")
    for lag in lags:
        df[f'Demand_lag_{lag}'] = df['Demand_MW'].shift(lag)
    print("Lag features created.")

    # --- Create Time Features ---
    print("Creating time features...")
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear # Added dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int) # Added weekofyear
    # df['quarter'] = df.index.quarter   # Optional
    print("Time features created.")

    # --- Handle NaNs created by shifts ---
    # Lag features introduce NaNs at the beginning. We must drop these rows.
    print(f"\nData shape before dropping NaNs from lags: {df.shape}")
    df.dropna(inplace=True)
    print(f"Data shape after dropping NaNs: {df.shape}")

    # --- Define Features (X) and Target (y) ---
    features = [col for col in df.columns if col not in ['Demand_MW', 'target']]
    X = df[features]
    y = df['target']

    # --- Final Check ---
    if X.empty or y.empty:
        print("Error: Feature matrix X or target vector y is empty after processing.")
    else:
        print("\nFeatures (X) Head:")
        print(X.head())
        print("\nTarget (y) Head:")
        print(y.head())
        print(f"\nFinal dimensions: X={X.shape}, y={y.shape}")

else:
    print("\nSkipping Feature Engineering as cleaned data is not available or valid.")
    X, y = pd.DataFrame(), pd.Series() # Ensure X, y exist but are empty


# %% [markdown]
# ## 4. Data Splitting (Time Series Aware)
#
# Split data into training and testing sets. (No shuffling.)

# %%
if not X.empty and not y.empty:
    # Define split point (e.g., 80% train, 20% test)
    test_size = 0.2
    # Ensures we have enough data to split
    if len(X) < 10: # Arbitrary small number check
         print("Error: Not enough data points to perform train/test split.")
         X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    else:
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTrain set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        print(f"Train set time range: {X_train.index.min()} to {X_train.index.max()}")
        print(f"Test set time range: {X_test.index.min()} to {X_test.index.max()}")
else:
    print("\nSkipping Data Splitting as features/target are empty.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()


# %% [markdown]
# ## 5. Build and Train ML Pipeline
#
# Create pipelines including `StandardScaler` and the regression models.

# %% [markdown]
# ### 5.1 Linear Regression Pipeline

# %%
pipeline_lr = None # Initialize
if not X_train.empty and not y_train.empty:
    # Define the pipeline
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Scale features
        ('regressor', LinearRegression())   # Step 2: Linear Regression model
    ])

    # Train the pipeline
    print("\nTraining Linear Regression Pipeline...")
    try:
        pipeline_lr.fit(X_train, y_train)
        print("Training complete.")
    except Exception as e:
        print(f"Error during Linear Regression training: {e}")
        pipeline_lr = None # Reset pipeline if training failed
else:
    print("\nSkipping Linear Regression Pipeline training due to empty training data.")

# %% [markdown]
# ### 5.2 Decision Tree Pipeline with GridSearchCV

# %%
best_pipeline_dt = None # Initialize
if not X_train.empty and not y_train.empty:
    # Define the pipeline
    pipeline_dt = Pipeline([
        # Optional: Remove scaler if you specifically want to test without scaling
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid_dt = {
        'regressor__max_depth': [5, 10, 15], # Adjusted grid for potentially more data
        'regressor__min_samples_leaf': [5, 10, 20]
    }

    # Set up GridSearchCV
    grid_search_dt = GridSearchCV(
        estimator=pipeline_dt,
        param_grid=param_grid_dt,
        scoring='neg_mean_squared_error',
        cv=5, # Use 5-fold CV. Consider TimeSeriesSplit for rigorous validation
        n_jobs=-1,
        verbose=1
    )

    # Train using GridSearchCV
    print("\nTraining Decision Tree Pipeline with GridSearchCV...")
    try:
        grid_search_dt.fit(X_train, y_train)
        best_pipeline_dt = grid_search_dt.best_estimator_
        print("Training complete.")
        print(f"\nBest Decision Tree parameters: {grid_search_dt.best_params_}")
        print(f"Best CV Score (neg MSE): {grid_search_dt.best_score_:.3f}")
    except Exception as e:
        print(f"Error during Decision Tree GridSearchCV training: {e}")
        best_pipeline_dt = None # Reset if training failed

else:
    print("\nSkipping Decision Tree Pipeline training due to empty training data.")


# %% [markdown]
# ## 6. Model Evaluation on Test Set
#
# Evaluate models on the unseen test data.

# %%
if (pipeline_lr or best_pipeline_dt) and not X_test.empty and not y_test.empty:
    print("\nEvaluating models on the Test Set:")

    def evaluate_model(y_true, y_pred, model_name="Model"):
        if y_true.empty or len(y_pred) != len(y_true):
             print(f"Skipping evaluation for {model_name} due to invalid inputs.")
             return None
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"--- {model_name} Evaluation ---")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R-squared (R²): {r2:.4f}")
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    y_pred_lr, y_pred_dt = None, None # Initialize predictions

    if pipeline_lr:
        try:
            y_pred_lr = pipeline_lr.predict(X_test)
            metrics_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
        except Exception as e:
            print(f"Error during Linear Regression prediction/evaluation: {e}")

    if best_pipeline_dt:
         try:
            y_pred_dt = best_pipeline_dt.predict(X_test)
            metrics_dt = evaluate_model(y_test, y_pred_dt, "Best Decision Tree")
         except Exception as e:
            print(f"Error during Decision Tree prediction/evaluation: {e}")


    # --- Visualize Predictions vs Actuals ---
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Demand', color='blue', alpha=0.6)
    if y_pred_lr is not None:
        plt.plot(y_test.index, y_pred_lr, label='Linear Regression Preds', color='orange', linestyle='--')
    if y_pred_dt is not None:
        plt.plot(y_test.index, y_pred_dt, label='Decision Tree Preds', color='green', linestyle=':')

    plt.title('Actual vs Predicted Demand (Test Set)')
    plt.xlabel('Timestamp')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.show()

    # Scatter plot of Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.suptitle('Model Predictions vs Actual Demand')

    plot_lims = [min(y_test.min(), (y_pred_lr if y_pred_lr is not None else [np.inf]).min(), (y_pred_dt if y_pred_dt is not None else [np.inf]).min()),
                 max(y_test.max(), (y_pred_lr if y_pred_lr is not None else [-np.inf]).max(), (y_pred_dt if y_pred_dt is not None else [-np.inf]).max())]
    plot_lims = [lim * 0.95 if lim > 0 else lim * 1.05 for lim in plot_lims] # Add padding


    if y_pred_lr is not None:
        axes[0].scatter(y_test, y_pred_lr, alpha=0.5)
        axes[0].plot(plot_lims, plot_lims, '--k') # y=x line
        axes[0].set_title('Linear Regression')
        axes[0].set_xlabel('Actual Demand (MW)')
        axes[0].set_ylabel('Predicted Demand (MW)')
        axes[0].grid(True)
        axes[0].set_xlim(plot_lims)
        axes[0].set_ylim(plot_lims)

    if y_pred_dt is not None:
        axes[1].scatter(y_test, y_pred_dt, alpha=0.5)
        axes[1].plot(plot_lims, plot_lims, '--k') # y=x line
        axes[1].set_title('Decision Tree')
        axes[1].set_xlabel('Actual Demand (MW)')
        axes[1].grid(True)
        axes[1].set_xlim(plot_lims)
        axes[1].set_ylim(plot_lims)

    if y_pred_lr is None and y_pred_dt is None:
         axes[0].set_title('No valid predictions to plot')
         axes[1].set_visible(False)
    elif y_pred_lr is None: # Only DT plot shown
         axes[0].set_visible(False)
         axes[1].set_position([0.1, 0.1, 0.8, 0.8]) # Center the plot
    elif y_pred_dt is None: # Only LR plot shown
         axes[1].set_visible(False)
         axes[0].set_position([0.1, 0.1, 0.8, 0.8]) # Center the plot


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

else:
    print("\nSkipping Model Evaluation due to issues in previous steps (data loading, training, or empty test set).")


# %% [markdown]
# ## 7. Feature Importance
#
# Analyze feature influence.

# %%
if not X_test.empty and not y_test.empty: # Check if we have features to analyze
    print("\n--- Feature Importance ---")

    # --- Linear Regression Coefficients ---
    if pipeline_lr:
        try:
            lr_model = pipeline_lr.named_steps['regressor']
            # Get coefficients and feature names (ensure features list matches trained model)
            lr_coeffs = pd.DataFrame({
                'Feature': X_train.columns, # Use columns from training data
                'Coefficient': lr_model.coef_
            }).sort_values(by='Coefficient', key=abs, ascending=False)

            print("\nLinear Regression Coefficients (on scaled data):")
            print(lr_coeffs)
        except Exception as e:
            print(f"Could not retrieve Linear Regression coefficients: {e}")

    # --- Decision Tree Feature Importances ---
    if best_pipeline_dt:
         try:
            dt_model = best_pipeline_dt.named_steps['regressor']
            dt_importances = pd.DataFrame({
                'Feature': X_train.columns, # Use columns from training data
                'Importance': dt_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            print("\nDecision Tree Feature Importances:")
            print(dt_importances)

            # Visualize Decision Tree Importances
            plt.figure(figsize=(10, max(6, len(dt_importances) * 0.3))) # Adjust height based on num features
            sns.barplot(x='Importance', y='Feature', data=dt_importances.head(15)) # Show top 15 or all
            plt.title('Top Feature Importances (Decision Tree)')
            plt.tight_layout()
            plt.show()
         except Exception as e:
            print(f"Could not retrieve Decision Tree importances: {e}")

else:
    print("\nSkipping Feature Importance analysis due to issues in previous steps.")


# %% [markdown]
# ## 8. Mini-Project Conclusion & Potential Next Steps
#
# - Loaded and cleaned raw UK demand data.
# - Built end-to-end pipelines for predicting demand using Linear Regression and a tuned Decision Tree.
# - Evaluated models on unseen test data using RMSE, MAE, R².
# - Analyzed feature importances, identifying key drivers. 


# %%
print("\nPart 2 Mini-Project script (Self-Contained) finished.")