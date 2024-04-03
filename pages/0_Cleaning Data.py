import  streamlit as st
from pyforest  import*
import warnings
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import  pandas as pd
import time
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
import base64
# ----- Page configs -----
st.set_page_config(
        page_title = "Fallou Fall Portfolio" ,
        page_icon = "🧪" ,
        )

# ----- Left menu -----
with st.sidebar :
    st.image("eae_img.png" , width = 200)
    st.write(
          """Utilizing a combination of customer votes, location data, food type, delivery options, 
          and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint
           the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that
            align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant 
            discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments 
            that cater to their specific needs and desires.""" )
    st.write(
            "Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----

st.title("🌶️ Zomato Bangalore Restaurants")
st.divider()


nrows = st.slider("Select the number of rows to read:", min_value=10, max_value=15, step=10)
try:
  df = pd.read_csv("data/zomato.csv", index_col=False, nrows=nrows)
  st.success(f"Successfully read {nrows} Samples ")
except FileNotFoundError:
  st.error("Error: Data file not found. Please check the path.")


if 'df' in locals():
  st.subheader("Data Preview")
  st.dataframe(df)

st.divider()
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return encoded_string

images = [
    {"path": "data/1.png", "caption": "Caption Text"},
    {"path": "data/2.png", "caption": "Caption Two"},
    {"path": "data/3.png", "caption": "Caption Three"},
    {"path": "data/4.png", "caption": "Caption Four"},
    {"path": "data/5.png", "caption": "Caption Five"},
    {"path": "data/6.png", "caption": "Caption Six"},
    {"path": "data/7.png", "caption": "Caption Seven"},
    {"path": "data/8.png", "caption": "Caption Eight"}
]

# Replace image src with base64 encoded images
html_content = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {box-sizing: border-box;}
body {font-family: Verdana, sans-serif;}
.mySlides {display: none;}
img {vertical-align: middle;}

/* Slideshow container */
.slideshow-container {
  max-width: 1200px;
  position: relative;
  margin: auto;
}

/* Caption text */
.text {
  color: #f2f2f2;
  font-size: 15px;
  padding: 8px 12px;
  position: absolute;
  bottom: 8px;
  width: 100%;
  text-align: center;
}

/* Number text (1/3 etc) */
.numbertext {
  color: #f2f2f2;
  font-size: 12px;
  padding: 8px 12px;
  position: absolute;
  top: 0;
}

/* Fading animation */
.fade {
  animation-name: fade;
  animation-duration: 2.0s;
}

@keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

/* On smaller screens, decrease text size */
@media only screen and (max-width: 300px) {
  .text {font-size: 11px}
}
</style>
</head>
<body>

<div class="slideshow-container">
"""

for i, image in enumerate(images):
    image_base64 = encode_image_to_base64(image["path"])
    html_content += f"""
    <div class="mySlides fade">
      <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: 300px;">
      <div class="text">{image["caption"]}</div>
    </div>
    """

html_content += """
</div>

<script>
let slideIndex = 0;
showSlides();

function showSlides() {
  let i;
  let slides = document.getElementsByClassName("mySlides");
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";  
  }
  slideIndex++;
  if (slideIndex > slides.length) {slideIndex = 1}    
  slides[slideIndex-1].style.display = "block";  
  setTimeout(showSlides, 5000); 
}
</script>

</body>
</html>
"""

# Render HTML using components.html
components.html(html_content, height=300)



st.divider()

def data_cleaning_section():
    st.header("Data Cleaning")
    st.markdown("During data cleaning, one of the tasks is to ensure that each column has the correct data type.")
    st.markdown("For example, numeric columns should have numeric data types (int, float), dates should have a datetime data type, "
                "and categorical variables should be represented using appropriate data types such as strings or categorical types.")
    st.markdown("Incorrect data types can lead to errors during analysis or modeling.")

def handling_null_duplicates_section():
    st.header("Handling Null Values and Duplicates")
    st.markdown("Handling null values and duplicates is essential in data cleaning for several reasons:")
    st.markdown("- **Data Integrity**: Null values and duplicates can compromise the integrity of the dataset. "
                "Null values represent missing or unknown information, which can distort statistical analysis and machine learning models if not handled properly. "
                "Duplicates can skew summary statistics and lead to overrepresentation of certain data points.")
    st.markdown("- **Accuracy of Analysis**: Null values and duplicates can affect the accuracy of data analysis and modeling. "
                "Performing statistical analysis or building machine learning models on datasets with null values or duplicates can result in biased results and incorrect conclusions.")
    st.markdown("- **Data Quality**: Cleaning null values and duplicates improves the overall quality of the dataset. "
                "Removing or imputing null values ensures that the dataset contains complete and reliable information, "
                "which is crucial for making informed decisions based on the data. "
                "Overall, handling null values and duplicates is a fundamental step in the data cleaning process, "
                "ensuring that the dataset is accurate, reliable, and suitable for analysis and modeling purposes.")

def text_preprocessing_section():
    st.header("Text Preprocessing")
    st.markdown("Text preprocessing involves removing punctuation from text data to simplify it for analysis, especially in natural language processing tasks, "
                "where punctuation typically doesn't carry significant meaning.")
    st.markdown("- **Data Consistency**: Converting data types ensures consistency within the dataset. "
                "Numeric data should be stored as numerical types (e.g., integer or float) for mathematical operations and analysis.")
    st.markdown("- **Data Quality**: Eliminating unnecessary characters like punctuation improves data quality by reducing noise and ensuring cleaner, more reliable data for analysis.")
    st.markdown("- **Compatibility with Analysis Tools**: Many analysis tools expect specific data formats or types. "
                "Converting data to the appropriate format ensures compatibility with these tools, enhancing the efficiency of analysis.")
    st.markdown("- **Efficiency**: Operations on numeric data are generally faster and more memory-efficient than on string data. "
                "Removing punctuation reduces the size of text data, making operations faster and more efficient.")

def label_encoding_section():
    st.header("Label Encoding")
    st.markdown("Label encoding is used in machine learning and data preprocessing to convert categorical variables into numerical format for algorithm compatibility, "
                "handling ordinal data, simplicity, efficiency, reduced memory usage, and improved interpretability.")
    st.markdown("It assigns a unique integer to each category, making the data suitable for analysis by numerical-based algorithms.")
    st.markdown("However, it's important to consider the inherent order or lack thereof in categorical variables when applying label encoding, "
                "as it may lead to misinterpretation in certain scenarios.")

# Main Streamlit app
st.title("Data Science Basics: Data Cleaning, Profiling, Analysis, and Label Encoding")
data_cleaning_section()
handling_null_duplicates_section()
text_preprocessing_section()
label_encoding_section()