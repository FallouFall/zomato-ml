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

def load_data() :
    data_path = "data/zomatoClean.csv"
    df = pd.read_csv( data_path)
    return df



def deleteDuplicate ( anyListeOfDuplicate ) :
    return [ *{ *anyListeOfDuplicate } ]


def getIndex (value_to_find, anyListe ) :
    for key , value in anyListe.items() :
        if value == value_to_find :
           return  key



df = load_data()
types_list  = df["type"]
types_list = deleteDuplicate ( types_list )
types_dict = {i+1: types_list[i] for i in range(len(types_list))}

location_list = df["location"]
location_list = deleteDuplicate ( location_list )
locations_dict = {i+1: location_list[i] for i in range(len(location_list))}

cost_list = df["cost"]
cost_list = deleteDuplicate (cost_list)
cost_dict = {i+1: cost_list[i] for i in range(len(cost_list))}


votes_list = df["votes"]
votes_list = deleteDuplicate (votes_list)
votes_dict = {i+1: votes_list[i] for i in range(len(votes_list))}

cuisine_list = df["cuisines"]
cuisine_list = deleteDuplicate (cuisine_list )
cuisine_dict = {i+1: cuisine_list[i] for i in range(len(cuisine_list))}

menu_list = df["menu_item"]
menu_list = deleteDuplicate (menu_list )
menu_dict = {i+1: menu_list[i] for i in range(len(menu_list))}


# ----- Left menu -----
with st.sidebar:
    st.image("eae_img.png", width=200)
    st.write("""Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires.""")
    st.write(
        "Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🍝!!!Best Restaurant  !!!🍝")
st.divider()
st.subheader("I can help you find the best Restaurant")

online_order =st.selectbox('Online order', ['Yes', 'No'])
book_table = st.selectbox('Book Table', ['Yes', 'No'])
votes =st.selectbox('Votes', list(votes_dict.values()))
location =st.selectbox('Location', list(locations_dict.values()))
restaurant_type =st.selectbox('Type Restaurant', list(types_dict.values()))
cuisines =st.selectbox('Cuisines', list(cuisine_dict.values()))
cost =st.selectbox('Cost', list(cost_dict.values()))
menu_item =st.selectbox('Menu', list(menu_dict.values()))


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
    book_table = 1 if book_table == 'Yes' else 0
    online_order = 1 if book_table == 'Yes' else 0
    votes=getIndex(votes,votes_dict)
    location = getIndex(location , locations_dict)
    restaurant_type = getIndex(restaurant_type , types_dict)
    cuisines = getIndex(cuisines , cuisine_dict)
    cost = getIndex(cost , cost_dict)
    menu_item = getIndex(menu_item , menu_dict)
    fields = [online_order, book_table, votes, location, restaurant_type, cuisines, cost, menu_item]
    final_features = [np.array(fields)]
    prediction = model.predict(final_features)
    score = round(prediction[0], 1)

    st.markdown(
        f'<div style="background-color: lightgreen; padding: 10px; font-size: 27px;display: flex; align-items: center; justify-content: center;">Review Score: {score}</div>',
        unsafe_allow_html=True)
