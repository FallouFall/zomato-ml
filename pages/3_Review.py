import streamlit as st
from pyforest import *
import warnings
import pickle  # Added pickle import
model = pickle.load(open('data/model.pkl', 'rb'))
warnings.filterwarnings('ignore')


# ----- Page configs -----
st.set_page_config(
    page_title="Fallou Fall Portfolio",
    page_icon="🧪",
)

# ----- Left menu -----
with st.sidebar:
    st.image("eae_img.png", width=200)
    st.write("""Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires.""")
    st.write(
        "Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🏗️!!!WORK IN PROGRESS !!!🏗️")
st.divider()
st.subheader("⚠️ONLY  INTEGER VALUE TO MAKE THE MODEL WORK⚠️")

online_order = st.number_input('Online Order', min_value=0, max_value=1, step=1, value=0)
book_table = st.number_input('Book Table', min_value=0, max_value=1, step=1, value=0)
votes = st.number_input('Votes', step=1, min_value=0, value=0)
location = st.number_input('Location', step=1, value=0, min_value=0, )
restaurant_type = st.number_input('Restaurant Type', step=1, value=0, min_value=0, )
cuisines = st.number_input('Cuisines', step=1, value=0, min_value=0, )
cost = st.number_input('Cost', step=1, value=0, min_value=0, )
menu_item = st.number_input('Menu Item', step=1, value=0, min_value=0, )

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: red;
        color:#ffffff
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button('Predict'):
    fields = [online_order, book_table, votes, location, restaurant_type, cuisines, cost, menu_item]
    final_features = [np.array(fields)]
    prediction = model.predict(final_features)
    score = round(prediction[0], 1)
    st.markdown(
        f'<div style="background-color: lightgreen; padding: 10px; font-size: 27px;display: flex; align-items: center; justify-content: center;">Review Score: {score}</div>',
        unsafe_allow_html=True)
