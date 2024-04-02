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
    st.write("""Restaurants from all over the world can be found here in Bengaluru. The basic idea of analyzing the Zomato dataset is to get a fair idea about the factors affecting the establishment
    of different types of restaurant at different places in Bengaluru, aggregate rating of each restaurant.""")
    st.write("Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🌶️ Modeling Zomato ")
st.divider()

st.subheader("Number of Rows to Read")

nrows = st.slider("Select the number of rows to read:", min_value=100, max_value=24000, step=100)
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
xgb_params = {'objective': 'reg:squarederror', 'random_state': 10}
random_forest_params = {'n_estimators': 650, 'random_state': 245, 'min_samples_leaf': 0.0001}

# Train and evaluate models
train_and_evaluate_model("Linear Regression", LinearRegression, linear_reg_params, x_train, y_train, x_test, y_test)
train_and_evaluate_model("Extra Trees Regressor", ExtraTreesRegressor, extra_tree_params, x_train, y_train, x_test,
                         y_test)
train_and_evaluate_model("Gradient Boost Regressor", XGBRegressor, xgb_params, x_train, y_train, x_test, y_test)
#train_and_evaluate_model("Random Forest Regressor", RandomForestRegressor, random_forest_params, x_train, y_train,
                         #x_test, y_test)