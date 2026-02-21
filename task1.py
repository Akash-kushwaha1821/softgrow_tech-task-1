import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

st.title("Iris Dataset EDA App")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

df.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']

df['species'] = df['species'].map({
    0:'Setosa',
    1:'Versicolor',
    2:'Virginica'
})

option = st.sidebar.radio(
    "Select Section",
    ["Preview","Statistics","Visuals","Pairplot","Conclusion"]
)

if option == "Preview":
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write(df['species'].value_counts())

elif option == "Statistics":
    st.write(df.describe())
    st.write(df.isnull().sum())

elif option == "Visuals":
    fig1, ax1 = plt.subplots()
    df.hist(ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df.drop('species',axis=1), ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.drop('species',axis=1).corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

elif option == "Pairplot":
    fig4 = sns.pairplot(df, hue="species")
    st.pyplot(fig4)

else:
    st.write("Petal features are highly correlated")
    st.write("Setosa is clearly separable")
    st.write("No missing values in dataset")
    st.write("Virginica has largest petals")