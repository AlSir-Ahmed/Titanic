import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.image("./Titanic.jpg")
st.write("""
# Titanic Survival Prediction App

This app predicts the survival from Titanic!
""")
st.write("---")

st.sidebar.header("Specify Input Parameters")
st.sidebar.write("---")
def user_input_features():
    Pclass = st.sidebar.selectbox("Ticket class", [1, 2, 3])
    Sex = st.sidebar.radio("Gender", ["Male", "Female"])
    sex = {"Male": 1, "Female": 0}
    Sex = sex[Sex]
    Age = st.sidebar.slider("Age", 1, 80)
    SibSp = st.sidebar.selectbox("Number of siblings / spouses aboard the Titanic", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    Parch = st.sidebar.selectbox("Number of parents / children aboard the Titanic", [0, 1, 2, 3, 4, 5, 6])
    Fare = st.sidebar.slider("Fare", 0.0, 512.3292)
    Embarked = st.sidebar.selectbox("Port of embarkation", ["Cherbourg", "Queenstown", "Southampton"])
    embarked = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    Embarked = embarked[Embarked]
    data = {"Pclass": Pclass,
            "Sex": Sex,
            "Age": Age,
            "SibSp": SibSp,
            "Parch": Parch,
            "Fare": Fare,
            "Embarked": Embarked}
    features = pd.DataFrame(data, index = [0])
    return features
df = user_input_features()
st.header("Specified Input Parameters")
st.write(df)
st.write("---")

titanic = pd.read_csv("titanic1.csv")
X = titanic.drop("Survived", axis = 1)
Y = titanic["Survived"]
model = LogisticRegression(max_iter = 5000)
model.fit(X, Y)
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.header("Class labels and their corresponding index number")
st.write(Y.unique())
st.header("Prediction")
st.write(prediction)
st.header("Prediction Probability")
st.write(prediction_proba)
