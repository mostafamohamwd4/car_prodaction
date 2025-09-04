#  Car Production Analysis & Prediction App

This is a **Streamlit-based web application** designed for analyzing and predicting car production data.  
It provides an **interactive interface** to explore datasets, visualize patterns, train machine learning models, and generate predictions.

---

## Features

###  Load Data
- Upload CSV files directly.
- Or load data from a local file path.
- Display dataset summary (rows, columns, missing values).

### Explore Data
- View dataset info & descriptive statistics.
- Handle missing values:
  - Delete rows
  - Fill with mean
  - Fill with mode
  - Fill with zero
- Preview dataset samples.

### Statistical Analysis
- Data distribution visualization (histograms).
- Relationship analysis (scatter plots, box plots).
- Correlation matrix heatmaps.

### 🤖 Modeling & Prediction
- Choose target & feature variables.
- Train ML models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Automatic preprocessing:
  - Label Encoding for categorical features
  - Standard Scaling for numerical features
- Data split into training & testing sets.

### Results & Reports
- Model evaluation metrics:
  - **MSE**, **RMSE**, **MAE**, **R² Score**
- Visualize predictions:
  - Actual vs Predicted plots
- Feature importance visualization (tree-based models).
- Predict new values with custom user input.

---

## Technologies Used
- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Regression & Tree-based models
- **Streamlit**: Interactive UI

---

## Project Structure
