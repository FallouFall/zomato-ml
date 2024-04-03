import  streamlit as st
from pyforest  import*
import time
import re
import plotly.express as px


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


nrows = st.slider("Select the number of rows to read:", min_value=500, max_value=24000, step=500)
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
st.subheader("Top Restaurant")
c = st.container(border = True)
fig = plt.figure(figsize = (10 , 5))
chains = df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("top Restaurant")
plt.xlabel("number")
plt.show()
c.pyplot(fig)





fig = px.box(
    df,
    y="rate",
    title="Box plot of approx Rating Restaurant",
    width=800,
    height=600
)

fig.update_layout(yaxis_title="Rating")
st.plotly_chart(fig)




st.divider()
st.subheader("review rates")
with st.container() :
    fig , ax = plt.subplots(figsize = (10 , 5))
    sns.histplot(df['rate'] , bins = 20 , kde = True)
    st.pyplot(fig)

st.subheader("Type Of Restaurant")
# Count occurrences of each type of restaurant
word_counts = { }
for word in df['type'] :
    if word in word_counts :
        word_counts[word] += 1
    else :
        word_counts[word] = 1

my_df = pd.DataFrame(word_counts.items())
restptype = sns.barplot(x = 0 , y = 1 , data = my_df)
restptype.set(xlabel = 'Type of Restaurant' , ylabel = 'Number' ,
title = 'Number of Occurrences of Each Type of Restaurant')
st.pyplot(plt.gcf())

# top favorite food
df.index = range(df.shape[0])
likes = []
for i in range(df.shape[0]) :
    array_split = re.split(',' , df['dish_liked'][i])
    for item in array_split :
        likes.append(item)

favorite = pd.Series(likes).value_counts()



st.divider()
st.subheader("Type Of Restaurant")
fig=plt.figure(figsize=(18,10))
rest= df['rest_type'].value_counts()[:20]
sns.barplot(x=rest, y=rest.index, alpha=0.9)
plt.title("Type of Restaurant")
plt.xlabel("number")
st.pyplot(fig)


st.divider()
st.subheader("Rating")
x1=((df['rate']>=1) & (df['rate']<2)).sum()
x2=((df['rate']>=2) & (df['rate']<3)).sum()
x3=((df['rate']>=3) & (df['rate']<4)).sum()
x4=((df['rate']>=4) & (df['rate']<=5)).sum()
slices = [x1,x2,x3,x4]
label=['1<rating<2','2<rating<3','3<rating<4','4<rating<5']
plt.figure(figsize=(6, 6))
plt.pie(slices, labels=label, autopct='%1.1f%%', pctdistance=.2)
plt.title('Restaurant Reviews %')
plt.axis('equal')
plt.legend(loc="upper right")
st.pyplot(plt.gcf())


#TOP 20 FOOD
st.divider()
st.subheader("Top Food")
likes = []
for i in range(df.shape[0]):
    array_split = re.split(',', df['dish_liked'][i])
    for item in array_split:
        likes.append(item.strip())


favorite = pd.Series(likes).value_counts()
st.title("Top 20 Liked Foods")
fig, ax = plt.subplots(figsize=(10, 8))
favorite.nlargest(n=20, keep='first').plot(kind='bar', ax=ax)
ax.set_xlabel("Food")
ax.set_ylabel("Count")
ax.set_title("Top 20 Food Likes")
for i in ax.patches:
    ax.annotate(str(i.get_height()), (i.get_x() * 1.005, i.get_height() * 1.005))

st.pyplot(fig)



st.divider()
st.subheader("Heatmap of Vote Location by Cost")
red_colorscale=  ['#ffffff','#8E0142','#C2185B','#E83A53','#FF6260','#FF8A80','#FFB3B1','#FFDAD7','#FFF7F3']
fig = px.density_heatmap(df, x='rate', y='cost', nbinsx=25, nbinsy=25,
                         color_continuous_scale=red_colorscale[::-1],  # Reverse for ascending red intensity
                        )
fig.update_layout(
    width=800,  # Adjust width as needed
    height=600,  # Adjust height as needed

    plot_bgcolor = 'white'
)
st.plotly_chart(fig)


def scatter_plot(data):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data['votes'], data['rate'], color='red', alpha=0.5)
    ax.set_title('Scatter Plot of Votes vs Rate')
    ax.set_xlabel('Votes')
    ax.set_ylabel('Rate')
    ax.grid(False)
    return fig
st.title('Scatter Plot of Votes vs Rate')
st.pyplot(scatter_plot(df))