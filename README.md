# UK Grid BESS Analytics

A comprehensive collection of projects exploring electricity demand analytics, economic dispatch, and battery storage optimization in the UK power grid. This repository contains a series of interconnected analyses and optimizations focused on understanding and improving the UK's electricity grid operations.

## 📋 Project Overview

This repository is organized into four main sections, each focusing on a different aspect of grid analytics and optimization:

### 1. UK National Grid Demand Data Analysis
- Analysis of historical electricity demand data
- Visualization of demand patterns and trends
- Statistical analysis of demand distribution
- Key visualizations:
  - Monthly demand patterns
  - Daily demand with 7-day rolling averages
  - Demand distribution analysis
  - Average demand patterns

### 2. Predicting UK Electricity Demand (ML)
- Machine learning models for demand forecasting
- Feature importance analysis
- Model performance evaluation
- Visualizations:
  - Top feature importance
  - Model predictions vs. actual demand
  - Test set performance analysis

### 3. Economic Dispatch Optimization
- Optimization of power generation dispatch
- Battery Energy Storage System (BESS) integration
- Cost minimization strategies
- Two main components:
  - Dispatching optimization
  - BESS arbitrage analysis

### 4. BESS Arbitrage Profit Maximization
- Analysis of battery storage arbitrage opportunities
- Market price analysis
- Profit optimization strategies
- Includes detailed market analysis documentation

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - [Additional packages will be listed in requirements.txt]

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kianjn/uk-grid-bess-analytics.git
   cd uk-grid-bess-analytics
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data

The project uses UK National Grid demand data from 2024, stored in CSV format. The data includes:
- Timestamp
- Electricity demand
- Additional features for ML analysis

## 🛠️ Usage

Each section contains its own Python script and can be run independently:

1. Demand Analysis:
   ```bash
   python "01.UK National Grid Demand Data Analysis/01.py"
   ```

2. Demand Prediction:
   ```bash
   python "02.Predicting UK Electricity Demand.ML/02.py"
   ```

3. Economic Dispatch:
   ```bash
   python "03.Economic Dispatch Optimization/01.Dispatching/dispatch.py"
   ```

4. BESS Optimization:
   ```bash
   python "03.Economic Dispatch Optimization/02.BESS Arbitrage/bess_arbitrage.py"
   ```

## 📈 Results

The repository includes various visualizations and analysis results:
- Demand patterns and trends
- ML model performance metrics
- Optimization results
- Market analysis insights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- UK National Grid Data
- Economic Dispatch Literature
- BESS Optimization Papers
- [Additional references will be added]

## 👥 Authors

- Kianjn
---

For any questions or suggestions, please open an issue in the repository.
