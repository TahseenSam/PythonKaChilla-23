import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px

st.write("""
# Random Forest Classifier
## By Chacha Japani
""")

st.sidebar.header("Change Iris Parameter")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length",4.3, 7.9, 5.4)#Min, Max, Default Value
    sepal_width = st.sidebar.slider("Sepal Width",2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}

    features =pd.DataFrame(data, index=[0])
    return features


#iris = pd.read_csv("https://raw.githubusercontent.com/streamlit/demo-data/master/iris.csv")
# above link not wprking 😂
iris = sns.load_dataset("iris")
st.subheader("iris Dataset")
st.write(iris) 

st.header("Data Insights - Plotly")
fig = px.scatter(iris, x="sepal_length", y="sepal_width", color="species")
st.plotly_chart(fig)

fig = px.bar(iris, x="species", y="petal_length", color="species")
st.plotly_chart(fig)



fig = px.scatter_3d(iris, x="petal_width", y="petal_length", z="sepal_length", color="species")
st.plotly_chart(fig)


st.header("Data Insights - Seaborn")
fig = plt.figure()
sns.boxplot(iris["sepal_width"])
st.pyplot(fig)

st.header("Plotly GDP Plot")
df = px.data.gapminder()
st.write(df.head())
fig2 = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
st.plotly_chart(fig2)

fig4 = px.bar(df, x="continent", y="pop", color="continent",
  animation_frame="year", animation_group="country", range_y=[0,4000000000])
st.plotly_chart(fig4)


st.header("Covid Generated")
df = pd.read_csv("animatedPlotlyCovid.csv")
st.write(df.head())
fig5 = px.bar(df, x="Country", y="CovidCases",color="Country", animation_frame="Month", range_y=[0,20000])
st.plotly_chart(fig5)


X = iris.drop("species", axis =1)
y = iris['species'] 

model = RandomForestClassifier()
model.fit(X,y)

st.sidebar.header("Change Iris Parameter")

df = user_input_features()
st.subheader("Parameters Selected")
st.write(df)

prediction = model.predict(df)
probab = model.predict_proba(df)

st.subheader("Predicted Class")
st.write(prediction[0]) #Only class
#st.write(iris["species"].unique())

st.subheader("Predicted Probability")
st.write(probab)
