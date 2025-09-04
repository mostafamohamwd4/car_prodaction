import streamlit as st  
import pandas as pd     
import numpy as np    
import matplotlib.pyplot as plt  
import seaborn as sns    
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
import io 
import warnings 
import os 
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Car Production ",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



with st.sidebar:
    st.title(" Car Analysis System")
    st.markdown("---")
    

    page = st.radio(
        "Choose page:",
        ["Load Data", "Explore Data", "Statistical Analysis", "Modeling & Prediction", "Results & Reports"]
    )
    
    st.markdown("---")
    st.info("""
    ### Instructions:
    1. Start by loading data
    2. Explore data distribution
    3. Perform statistical analysis
    4. Create models and predict
    5. View results
    """)


if page == "Load Data":
    st.title("Load Car Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Load Data File")
        data_option = st.radio(
            "Data loading method:",
            ["Upload file", "Use file path"]
        )
        
        if data_option == "Upload file":
            uploaded_file = st.file_uploader("Choose car data file (CSV)", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("Data loaded successfully!")

        else:
            file_path = st.text_input("Enter file path:", "data/car_production.csv")
            if st.button("Load Data"):
                if os.path.exists(file_path):
                    df = load_data(file_path)
                    if df is not None:
                        st.session_state.df = df
                        st.success("Data loaded successfully!")
                else:
                    st.error("File not found at the specified path")
    
    with col2:
        st.header("Data Information")
        if 'df' in st.session_state:
            df = st.session_state.df
            st.metric("Number of Records", df.shape[0])
            st.metric("Number of Variables", df.shape[1])
            st.metric("Missing Values", df.isnull().sum().sum())
        else:
            st.info("Please load data first")
    
   
    if 'df' in st.session_state:
        st.subheader("Data Sample")
        st.dataframe(st.session_state.df.head(10))


elif page == "Explore Data":
    st.title(" Explore Car Data")
    
    if 'df' not in st.session_state:
        st.warning(" Please load data first from 'Load Data' page")
        st.stop()
    
    df = st.session_state.df
    
   
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    with col2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())
    
  
    st.subheader("Handle Missing Values")
    if df.isnull().sum().sum() > 0:
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        for col in missing_cols.index:
            method = st.selectbox(
                f"Method for handling missing values in '{col}'",
                ["Delete rows", "Fill with mean", "Fill with mode", "Fill with zero"],
                key=col
            )
            
            if method == "Delete rows":
                df = df.dropna(subset=[col])
            elif method == "Fill with mean":
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    st.warning(f"Cannot use mean for column '{col}' as it's not numeric")
            elif method == "Fill with mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif method == "Fill with zero":
                df[col] = df[col].fillna(0)
                
        st.success("Missing values handled successfully")
        st.session_state.df = df
    else:
        st.info("No missing values in the data")


elif page == "Statistical Analysis":
    st.title("Statistical Analysis of Car Data")
    
    if 'df' not in st.session_state:
        st.warning(" Please load data first from 'Load Data' page")
        st.stop()
    
    df = st.session_state.df
    
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        
        st.header("Charts and Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Distribution")
            selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist_col")
            fig = px.histogram(df, x=selected_col, nbins=20, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Variable Relationships")
            x_col = st.selectbox("Select X axis", numeric_cols, key="x_col")
            y_col = st.selectbox("Select Y axis", numeric_cols, key="y_col", 
                                index=1 if len(numeric_cols) > 1 else 0)
            color_col = st.selectbox("Select column for coloring", [None] + categorical_cols, key="color_col")
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                            title=f"Relationship between {x_col} and {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader("Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix between Numerical Variables")
        st.plotly_chart(fig, use_container_width=True)
        
        
        if categorical_cols:
            st.subheader("Categorical Variable Analysis")
            cat_col = st.selectbox("Select categorical variable", categorical_cols)
            value_col = st.selectbox("Select numerical variable", numeric_cols)
            
            fig = px.box(df, x=cat_col, y=value_col, title=f"Distribution of {value_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numerical variables in the data for analysis")

elif page == "Modeling & Prediction":
    st.title(" Modeling and Prediction for Car Data")
    
    if 'df' not in st.session_state:
        st.warning("Please load data first from 'Load Data' page")
        st.stop()
    
    df = st.session_state.df
    
    st.header("Select Variables for Model")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        target_options = numeric_cols
        target_var = st.selectbox("Select Target Variable", target_options)
        
        feature_options = [col for col in numeric_cols if col != target_var]
        selected_features = st.multiselect(
            "Select Features",
            feature_options,
            default=feature_options[:min(3, len(feature_options))] if feature_options else []
        )
        
        if not selected_features:
            st.warning("Please select at least one feature")
            st.stop()
        
        st.header("Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_option = st.selectbox(
                "Select Machine Learning Model:",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", 
                 "Decision Tree", "Random Forest", "Gradient Boosting"]
            )
        
        with col2:
            test_size = st.slider("Test Data Percentage (%)", 10, 40, 20)
            random_state = st.slider("Random State", 0, 100, 42)
        
    
        if st.button("Train Model"):
            
            df_encoded = df.copy()
            if categorical_cols:
                le = LabelEncoder()
                for col in categorical_cols:
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            
            X = df_encoded[selected_features]
            y = df_encoded[target_var]
            
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size/100, random_state=random_state
            )
            
            st.success(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Ridge Regression":
                model = Ridge()
            elif model_option == "Lasso Regression":
                model = Lasso()
            elif model_option == "Decision Tree":
                model = DecisionTreeRegressor(random_state=random_state)
            elif model_option == "Random Forest":
                model = RandomForestRegressor(random_state=random_state)
            elif model_option == "Gradient Boosting":
                model = GradientBoostingRegressor(random_state=random_state)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.scaler = scaler
            st.session_state.selected_features = selected_features
            st.session_state.target_var = target_var
            
            st.success("Model trained successfully!")
    else:
        st.warning("No numerical variables in the data for modeling")

elif page == "Results & Reports":
    st.title("Model Results and Reports")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first from 'Modeling & Prediction' page")
        st.stop()
    
    
    st.header("Model Evaluation Results")
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", f"{mse:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")
    with col4:
        st.metric("R² Score", f"{r2:.4f}")
  
    st.subheader("Prediction Results")
    
    results_df = pd.DataFrame({
        'Actual Value': y_test.values,
        'Predicted Value': y_pred,
        'Difference': abs(y_test.values - y_pred)
    })
    
    st.dataframe(results_df.head(10))
   
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.values, y=y_pred, mode='markers',
        name='Predictions', marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()], 
        y=[y_test.min(), y_test.max()],
        name='Ideal Line', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
 
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                     orientation='h', title='Feature Importance in the Model')
        st.plotly_chart(fig, use_container_width=True)
    
    st.header("Predict New Values")
    
    new_data = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(st.session_state.selected_features):
        if i % 2 == 0:
            with col1:
                new_data[feature] = st.number_input(
                    f"{feature}", 
                    value=float(st.session_state.df[feature].mean()) if st.session_state.df[feature].dtype != 'object' else 0,
                    key=f"new_{feature}"
                )
        else:
            with col2:
                new_data[feature] = st.number_input(
                    f"{feature}", 
                    value=float(st.session_state.df[feature].mean()) if st.session_state.df[feature].dtype != 'object' else 0,
                    key=f"new_{feature}"
                )
    
    if st.button("Predict"):
    
        new_df = pd.DataFrame([new_data])
        new_scaled = st.session_state.scaler.transform(new_df)
        
        prediction = st.session_state.model.predict(new_scaled)
        
        st.success(f"Predicted value for {st.session_state.target_var}: **{prediction[0]:.2f}**")