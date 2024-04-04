import streamlit as st
from pyforest import *
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pandas as pd
import time
import os
import pickle
from sklearn.svm import SVR
warnings.filterwarnings('ignore')
import re

# ----- Page configs -----
st.set_page_config(
    page_title="Fallou Fall Portfolio",
    page_icon="🧪",
)



# ----- Caching the menu -----

def load_data() :
    data_path = "data/zomatoM.csv"
    df = pd.read_csv( data_path)
    return df

df = load_data()
x = df.drop('rate', axis=1)
y = df['rate']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)





# ----- Left menu -----
with st.sidebar:
    st.image("eae_img.png", width=200)
    st.write("""Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires.""")
    st.write("Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🌶️ Modeling Zomato ")
st.divider()

st.subheader("Number of Rows to Read")

nrows = st.slider("Select the number of rows to read:", min_value=2000, max_value=24000, step=100)
try:
    df = pd.read_csv("data/zomatoM.csv", index_col=False, nrows=nrows)
    x = df.drop('rate' , axis = 1)
    y = df['rate']
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state = 10)
    st.success(f"Successfully read {nrows} rows from the CSV file.")
except FileNotFoundError:
    st.error("Error: Data file not found. Please check the path.")

if 'df' in locals():
    st.subheader("Data Preview")
    st.dataframe(df.head(10))




def train_and_evaluate_model(model_name, model_class, model_params, x_train, y_train, x_test, y_test):
    st.subheader(model_name)
    with st.empty():
        progress_bar = st.progress(0)
        for i in range(100):  # Adjust for desired progress bar duration
            progress_bar.progress(i + 1)
            time.sleep(0.01)  # Simulate training time
            progress_bar.empty()
    with st.expander(model_name, expanded=True):
        start = time.time()
        model = model_class(**model_params)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        end = time.time()

        if model_name == "Extra Trees Regressor":
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=1000, step=10,
                                     value=model_params.get('n_estimators', 120))
            model_params['n_estimators'] = n_estimators
        elif model_name == "Random Forest Regressor":
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=1000, step=10,
                                     value=model_params.get('n_estimators', 650))
            min_samples_leaf = st.slider("Minimum Samples per Leaf", min_value=0.0001, max_value=0.1, step=0.0001,
                                         value=model_params.get('min_samples_leaf', 0.0001))
            model_params['n_estimators'] = n_estimators
            model_params['min_samples_leaf'] = min_samples_leaf

        results_df = pd.DataFrame({
            'Metric': ['Model', 'Score', 'MSE', 'Time'],
            'Value': [model_name, r2_score(y_test, prediction), mean_squared_error(y_test, prediction),
                      end - start]
        })

        st.dataframe(results_df)


linear_reg_params = {}
extra_tree_params = {'n_estimators': 120}
#xgb_params = {'objective': 'reg:squarederror', 'random_state': 10}
xgb_params = {'objective': 'reg:squarederror', 'random_state': 10, 'n_estimators': 50, 'learning_rate': 0.3, 'max_depth': 2}
svm_params = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
random_forest_params = {'n_estimators': 300, 'random_state': 245, 'min_samples_leaf': 0.0001}

# Train and evaluate models
train_and_evaluate_model("Linear Regression", LinearRegression, linear_reg_params, x_train, y_train, x_test, y_test)
train_and_evaluate_model("Extra Trees Regressor", ExtraTreesRegressor, extra_tree_params, x_train, y_train, x_test,
                         y_test)
train_and_evaluate_model("Random Forest Regressor", RandomForestRegressor, random_forest_params, x_train, y_train,
                         x_test, y_test)
train_and_evaluate_model("Support Vector Machine", SVR, svm_params, x_train, y_train, x_test, y_test)
train_and_evaluate_model("Gradient Boost Regressor", XGBRegressor, xgb_params, x_train, y_train, x_test, y_test)

