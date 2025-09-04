import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


from model import get_models as get_model1
from model2 import get_models as get_model2

st.set_page_config(page_title="Car Production Analysis", layout="wide")
st.title("Car Production Prediction App")

st.subheader(" تحميل البيانات")
uploaded_file = st.file_uploader("ارفع ملف CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("تم تحميل البيانات بنجاح")
    st.dataframe(df.head())

    #
    target = st.selectbox(" اختر العمود الهدف للتنبؤ", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    
    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {}
    models.update(get_model1())
    models.update(get_model2())

    model_name = st.selectbox("اختر الموديل", list(models.keys()))
    model = models[model_name]


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  
    st.subheader(" نتائج التقييم")
    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

    st.subheader(" مقارنة القيم الحقيقية مع المتوقعة")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, label="Predictions")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Line")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    st.pyplot(fig)

    
    st.subheader(" توقع قيمة جديدة")
    new_data = {}
    for feature in X.columns:
        new_data[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))

    if st.button("Predict New Value"):
        new_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_df)
        prediction = model.predict(new_scaled)
        st.success(f"  التنبؤ للقيمة الهدف ({target}): {prediction[0]:.2f}")

else:
    st.warning("من فضلك ارفع ملف CSV علشان يبدأ التحليل")
