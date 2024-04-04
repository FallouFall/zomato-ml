import  streamlit as st
from pyforest  import*
import time
import re
import plotly.express as px
import plotly.graph_objects as go

# ----- Page configs -----
st.set_page_config(
        page_title = "Fallou Fall Portfolio" ,
        page_icon = "🧪" ,
        )

# ----- Left menu -----
with st.sidebar :
    st.image("eae_img.png" , width = 200)
    st.write(
          """Utilizing a combination of customer votes, location data, food type, delivery options, and cuisine preferences, our cutting-edge model generates predictive scores to help users pinpoint the perfect restaurant. By analyzing these multifaceted factors, we offer tailored recommendations that align with individual tastes and convenience preferences. This innovative approach not only facilitates restaurant discovery but also elevates overall dining experiences, ensuring users consistently find top-quality establishments that cater to their specific needs and desires.""" )
    st.write(
            "Data extracted from: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants")

# ----- Title of the page -----
st.title("🌶️ Zomato Bangalore Restaurants")
st.divider()


nrows = st.slider("Select the number of rows to read:", min_value=2000, max_value=24000, step=500)
try:
  df = pd.read_csv("data/zomatoClean.csv", index_col=False, nrows=nrows)
  st.success(f"Successfully read {nrows} Samples ")
except FileNotFoundError:
  st.error("Error: Data file not found. Please check the path.")


if 'df' in locals():
  st.subheader("Data Preview")
  df = pd.DataFrame(df)
  st.write(df)

st.divider()

chains = df['name'].value_counts().nlargest(20).sort_values(ascending=True)  # Sort in descending order
fig = go.Figure(go.Bar(x=chains.values, y=chains.index, orientation='h', marker_color='deepskyblue'))
fig.update_layout(title="Top Restaurant", xaxis_title="Number", yaxis_title="Restaurant Name")
st.plotly_chart(fig)



fig = px.box(
    df,
    y="rate",
    title="Box plot of approx Rating Restaurant",
    width=600,
    height=500,
    color_discrete_sequence=['deepskyblue']
)

fig.update_layout(yaxis_title="Rating")
st.plotly_chart(fig)




st.divider()

with st.container() :
    fig = px.histogram(df , x = 'rate' , title = "Review Rates" , nbins = 20 ,
            color_discrete_sequence = ['deepskyblue'])

st.plotly_chart(fig)





likes = []
for item in df['dish_liked'].dropna():
    array_split = re.split(',', item)
    for item in array_split:
        likes.append(item.strip())


favorite = pd.Series(likes).value_counts().nlargest(20)
favorite = favorite.sort_values(ascending=True)
fig = px.bar(x=favorite.values, y=favorite.index, orientation='h',
             labels={'x': 'Count', 'y': 'Food'},
             title='Top 20 Liked Foods',
             color=favorite.index, color_continuous_scale=['deepskyblue']*len(favorite.index))
st.plotly_chart(fig)


# top favorite restaurant
word_counts = df['type'].value_counts()
fig = px.bar(x=word_counts.index, y=word_counts.values, color=word_counts.index,
             labels={'x': 'Type of Restaurant', 'y': 'Number'},
             title='Number of Occurrences of Each Type of Restaurant')
fig.update_traces(marker_color='deepskyblue')
st.plotly_chart(fig)






st.divider()

top_rest_types = df['rest_type'].value_counts().nlargest(20)
fig = px.bar(top_rest_types, y=top_rest_types.index, x=top_rest_types.values, orientation='h',
             title="Type of Restaurant", labels={'x': 'Number', 'y': 'Type of Restaurant'},
             color=top_rest_types.index, color_discrete_sequence=['deepskyblue']*len(top_rest_types))









st.divider()
# Calculate the number of restaurants in each rating category
x1 = ((df['rate'] >= 1) & (df['rate'] < 2)).sum()
x2 = ((df['rate'] >= 2) & (df['rate'] < 3)).sum()
x3 = ((df['rate'] >= 3) & (df['rate'] < 4)).sum()
x4 = ((df['rate'] >= 4) & (df['rate'] <= 5)).sum()

# Create a DataFrame for the pie chart
pie_data = pd.DataFrame({'Rating Category': ['1<rating<2', '2<rating<3', '3<rating<4', '4<rating<5'],
                         'Count': [x1, x2, x3, x4]})

# Create the pie chart using Plotly with deepskyblue color
fig = px.pie(pie_data, values='Count', names='Rating Category',
             title='Restaurant Reviews %', hole=0.3,
             color_discrete_sequence=['deepskyblue'] * len(pie_data))

# Show the plot
st.plotly_chart(fig)








st.divider()
deep_sky_blue_colorscale = ['#ffffff', '#00BFFF']
fig = px.density_heatmap(df, x='rate', y='cost', nbinsx=25, nbinsy=25,
                         color_continuous_scale=deep_sky_blue_colorscale)
fig.update_layout(
    title="Heatmap of Vote Location by Cost",
    width=800,
    height=600,
    plot_bgcolor='white'
)
st.plotly_chart(fig)



st.divider()


def scatter_plot( data ) :
    fig = px.scatter(data , x = 'rate' , y = 'cost' , title = 'Scatter Plot of Cost vs Rate' ,
            labels = { 'Rate' : 'Rates' , 'Cost' : 'Cost' } ,
            opacity = 0.6 , trendline = 'ols')

    fig.update_layout(
            title = "Scatter Plot of Rate vs Cost" ,
            xaxis_title = "Votes" ,
            yaxis_title = "Rate" ,
            showlegend = False ,
            width = 600 ,
            height = 600 ,
            plot_bgcolor = 'white'
            )
    return fig


st.plotly_chart(scatter_plot(df))