# Car Production Analysis & Prediction System

This is a **Streamlit web application** for analyzing, visualizing, and predicting car production data using Machine Learning models.  
The system provides a complete workflow starting from **loading data** to **exploring, analyzing, modeling, and predicting**.

---

##  Features

### 1. Load Data
- Upload a CSV file or provide a file path.
- Display dataset shape (rows, columns).
- Show missing values summary.
- Preview first 10 rows.

### 2. Explore Data
- Dataset information & descriptive statistics.
- Handle missing values (delete rows, fill with mean/mode/zero).
- Update session state with cleaned dataset.

### 3. Statistical Analysis
- Plot data distributions (histograms).
- Explore variable relationships (scatter plots, box plots).
- Correlation matrix (heatmap).
- Analyze categorical variables against numerical variables.

### 4. Modeling & Prediction
- Select **target variable** and **features**.
- Choose ML models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Automatic preprocessing (scaling, label encoding).
- Train/test split with adjustable test size.
- Train model and evaluate.

### 5. Results & Reports
- Show model performance metrics:
  - MSE, RMSE, MAE, R² Score.
- Compare **actual vs predicted values** (scatter plot).
- Feature importance (for tree-based models).
- Predict new values based on user input.

---

## Tech Stack

- **Frontend/UI**: Streamlit  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Machine Learning**: Scikit-learn  

---
