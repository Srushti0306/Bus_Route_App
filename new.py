import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
file = st.file_uploader("Upload CSV", type="csv")
df = pd.read_csv(file)

# Data preprocessing
df.drop(['Unnamed: 5'], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)
df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
df['Kilometer'] = pd.to_numeric(df['Kilometer'], errors='coerce')
df.fillna(df.mean(), inplace=True)

# Train the linear regression model
x = df['Kilometer'].values.reshape(-1, 1)
y = df['Frequency'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)

# Create a Streamlit app
st.title("Bus Frequency Prediction")
st.markdown("Enter the kilometer value to predict the frequency:")

# Input form
kilometer_input = st.number_input("Kilometer", min_value=0.0)
prediction_button = st.button("Predict")

# Perform prediction when the button is clicked
if prediction_button:
    predicted_frequency = lr.predict([[kilometer_input]])
    st.write("Predicted Frequency:", predicted_frequency[0][0])

# Display scatter plot and regression line
st.markdown("## Scatter Plot")
sns.scatterplot(data=df, x='Frequency', y='Kilometer')
st.pyplot()

st.markdown("## Regression Plot")
sns.regplot(x='Frequency', y='Kilometer', data=df)
st.pyplot()

# Display pair plot and distribution plot
st.markdown("## Pair Plot")
sns.pairplot(data=df)
st.pyplot()

st.markdown("## Distribution Plot")
sns.displot(data=df, x="Frequency", y="Kilometer")
st.pyplot()
