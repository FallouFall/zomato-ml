import streamlit as st
import base64
# Set page configuration
st.set_page_config(
    page_title="Fallou Fall Portfolio",
    page_icon="🧪",
)

# ----- Left menu -----
with st.sidebar:
    st.image("eae_img.png", width=200)
    st.header("Data Science 🧬 ")
    st.write("###")
    st.write(" Project Machine Learning - April 2024")
    st.write("**Author:**[Fallou Fall  ](https://github.com/FallouFall)")



# ----- Top title -----
st.write(
        f"""<div style="text-align: center;"><h2 style="text-align: center;">🧑🏽‍🔬 Hi! My name is Fallou Fall</h2></div>""" ,
        unsafe_allow_html = True)


# ----- Profile image file -----
profile_image_file_path = "./data/pp.jpg"

with open(profile_image_file_path, "rb") as img_file:
    img = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()


# ----- Your Profile Image -----
st.write(f"""
<div style="display: flex; justify-content: center;">
    <img src="{img}" alt="Your Name" width="180" height="180" style="border-radius: 50%; object-fit: cover; margin-top: 40px; margin-bottom: 40px;">
</div>
""", unsafe_allow_html=True)


current_role = "Master in Big & Data Analytics at EAE Business School Barcelona"

st.write(f"""<div style="text-align: center;"><h4><i>{current_role}</i></h4></div>""", unsafe_allow_html=True)

st.write("##")

# Sidebar content
with st.sidebar:
    st.write(
        """Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires."""
    )
    st.write("Data extracted from: [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)")

# Title of the page
st.title("🌶️ Modeling Zomato")
st.divider()

# Contact information
st.title('Contact Page')
st.markdown(
    '''
    ## 🧑‍🎓 Data Science Web Application
    - **Tech Stack:** Python
    - **App URL:** [Data MLL App](https://zomato-ml.streamlit.app/)
    ## Projects
    - **H**
        - **GitHub:** [Machine-Learing](https://github.com/FallouFall/Machine-Learing)
        - **GitHub:** [eae_ipld_project](https://github.com/FallouFall/eae_ipld_project)
        - **Tech Stack:** JavaFX/Springboot/Angular https://www.gnudem.com/
        - **GitHub:** [Hospital Management GitHub Repo](https://github.com/FallouFall/Hoggy_Web)
        - **App URL:** [Data Visualisation App](https://datavisualisation.streamlit.app/)
       
        
    - **t**
        - **Tech Stack:** JEE/Springboot/Continuous Integration
        - **GitHub:** [School Management GitHub Repo](https://github.com/FallouFall/GestionScolaire)
        - **GitHub (RMI Version):** [School Management with RMI GitHub Repo](https://github.com/FallouFall/ServeurGestionEtudiant)
        - **GitHub (Client Version):** [Client for School Management with RMI GitHub Repo](https://github.com/FallouFall/ClientGestionEtudiants)
        - **GitHub:** [Zomato Machine Learning](https://github.com/FallouFall/zomato-ml)
    ---

    ## 📫 How to Reach Me:
    - **Email:** falloufalllive@gmail.com
    - **LinkedIn:** [Fallou Fall](https://www.linkedin.com/in/fallou-fall-047675173/)
    - **Tel:** Es (+34) 602552848) ,Us (+1) 769 274-0198, Sn (+221) 776880723
    ## 🏠 Location:
    - Barcelona: Lloret del Mar Barcelona
    '''
)
