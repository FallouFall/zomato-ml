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


# ----- Page configs -----
st.set_page_config(
        page_title = "Fallou Fall Portfolio" ,
        page_icon = "🧪" ,
        )

# ----- Left menu -----
with st.sidebar :
    st.image("eae_img.png" , width = 200)
    st.write(
          """Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires."" )
    st.write(
            "Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🏗️!!!WORK IN PROGRESS !!!🏗️")


image_files = ["data/1.png" , "data/2.png" , "data/3.png" , "data/4.png" , "data/5.png" , "data/6.png"]
desc = [
    "During data cleaning, one of the tasks is to ensure that each column has the correct data type. "
    "For example, numeric columns should have numeric data types (int, float), dates should have a datetime data type, "
    "and categorical variables should be represented using appropriate data types such as strings or categorical types. "
    "Incorrect data types can lead to errors during analysis or modeling." ,

    "Handling null values and duplicates is essential in data cleaning for several reasons: "
    "Data Integrity: Null values and duplicates can compromise the integrity of the dataset. "
    "Null values represent missing or unknown information, which can distort statistical analysis and machine learning models if not handled properly. "
    "Duplicates can skew summary statistics and lead to overrepresentation of certain data points. "
    "Accuracy of Analysis: Null values and duplicates can affect the accuracy of data analysis and modeling. "
    "Performing statistical analysis or building machine learning models on datasets with null values or duplicates can result in biased results and incorrect conclusions. "
    "Data Quality: Cleaning null values and duplicates improves the overall quality of the dataset. "
    "Removing or imputing null values ensures that the dataset contains complete and reliable information, "
    "which is crucial for making informed decisions based on the data. "
    "Overall, handling null values and duplicates is a fundamental step in the data cleaning process, "
    "ensuring that the dataset is accurate, reliable, and suitable for analysis and modeling purposes." ,

    "Text Preprocessing: Removing punctuation from text data simplifies it for analysis, especially in natural language processing tasks, "
    "where punctuation typically doesn't carry significant meaning. Data Consistency: Converting data types ensures consistency within the dataset. "
    "Numeric data should be stored as numerical types (e.g., integer or float) for mathematical operations and analysis. "
    "Data Quality: Eliminating unnecessary characters like punctuation improves data quality by reducing noise and ensuring cleaner, more reliable data for analysis. "
    "Compatibility with Analysis Tools: Many analysis tools expect specific data formats or types. Converting data to the appropriate format ensures compatibility with these tools, "
    "enhancing the efficiency of analysis. Efficiency: Operations on numeric data are generally faster and more memory-efficient than on string data. "
    "Removing punctuation reduces the size of text data, making operations faster and more efficient." ,

    "Label encoding is used in machine learning and data preprocessing to convert categorical variables into numerical format for algorithm compatibility, "
    "handling ordinal data, simplicity, efficiency, reduced memory usage, and improved interpretability. It assigns a unique integer to each category, "
    "making the data suitable for analysis by numerical-based algorithms. However, it's important to consider the inherent order or lack thereof in categorical "
    "variables when applying label encoding, as it may lead to misinterpretation in certain scenarios."
    ]


image_files = ["data/1.png", "data/2.png", "data/3.png", "data/4.png", "data/5.png", "data/6.png"]
desc = [
    "During data cleaning, one of the tasks is to ensure that each column has the correct data type. "
    "For example, numeric columns should have numeric data types (int, float), dates should have a datetime data type, "
    "and categorical variables should be represented using appropriate data types such as strings or categorical types. "
    "Incorrect data types can lead to errors during analysis or modeling.",

    "Handling null values and duplicates is essential in data cleaning for several reasons: "
    "Data Integrity: Null values and duplicates can compromise the integrity of the dataset. "
    "Null values represent missing or unknown information, which can distort statistical analysis and machine learning models if not handled properly. "
    "Duplicates can skew summary statistics and lead to overrepresentation of certain data points. "
    "Accuracy of Analysis: Null values and duplicates can affect the accuracy of data analysis and modeling. "
    "Performing statistical analysis or building machine learning models on datasets with null values or duplicates can result in biased results and incorrect conclusions. "
    "Data Quality: Cleaning null values and duplicates improves the overall quality of the dataset. "
    "Removing or imputing null values ensures that the dataset contains complete and reliable information, "
    "which is crucial for making informed decisions based on the data. "
    "Overall, handling null values and duplicates is a fundamental step in the data cleaning process, "
    "ensuring that the dataset is accurate, reliable, and suitable for analysis and modeling purposes.",

    "Text Preprocessing: Removing punctuation from text data simplifies it for analysis, especially in natural language processing tasks, "
    "where punctuation typically doesn't carry significant meaning. Data Consistency: Converting data types ensures consistency within the dataset. "
    "Numeric data should be stored as numerical types (e.g., integer or float) for mathematical operations and analysis. "
    "Data Quality: Eliminating unnecessary characters like punctuation improves data quality by reducing noise and ensuring cleaner, more reliable data for analysis. "
    "Compatibility with Analysis Tools: Many analysis tools expect specific data formats or types. Converting data to the appropriate format ensures compatibility with these tools, "
    "enhancing the efficiency of analysis. Efficiency: Operations on numeric data are generally faster and more memory-efficient than on string data. "
    "Removing punctuation reduces the size of text data, making operations faster and more efficient.",

    "Label encoding is used in machine learning and data preprocessing to convert categorical variables into numerical format for algorithm compatibility, "
    "handling ordinal data, simplicity, efficiency, reduced memory usage, and improved interpretability. It assigns a unique integer to each category, "
    "making the data suitable for analysis by numerical-based algorithms. However, it's important to consider the inherent order or lack thereof in categorical "
    "variables when applying label encoding, as it may lead to misinterpretation in certain scenarios."
]

# Initialize the index to track the current image
current_index = 0

# Display the initial image and description
image_placeholder = st.empty()
desc_placeholder = st.empty()
image_placeholder.image(image_files[current_index], use_column_width=True)
desc_placeholder.write(desc[current_index])

# Button to go to the previous image
if st.button('Previous') and current_index > 0:
    current_index -= 1
    image_placeholder.image(image_files[current_index], use_column_width=True)
    desc_placeholder.write(desc[current_index])

# Button to go to the next image
if st.button('Next') and current_index < len(image_files) - 1:
    current_index += 1
    image_placeholder.image(image_files[current_index], use_column_width=True)
    desc_placeholder.write(desc[current_index])