import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the dataset
file = st.file_uploader("Upload CSV", type="csv")
df = pd.read_csv(file)

# Remove unnecessary column
df2 = df.drop(['Unnamed: 5'], axis=1)

# Drop rows with missing values
df3 = df2.dropna(axis=0)

# Drop duplicate rows based on 'Route Description'
df4 = df3.drop_duplicates('Route Description', keep='first')

# Convert 'Frequency' and 'Kilometer' columns to numeric
df4['Frequency'] = pd.to_numeric(df4['Frequency'], errors='coerce')
df4['Kilometer'] = pd.to_numeric(df4['Kilometer'], errors='coerce')

st.markdown(
    """
    <style>
    .header {
        color: #336699;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        color: #993366;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    body {
        background-color: #B0E0E6;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title('Route Frequency and Distance Analysis')

# Create a search box
st.markdown('<div class="header">Enter the bus name:</div>', unsafe_allow_html=True)
search_bus = st.text_input("")

# Filter the dataframe based on the search input
filtered_df = df4[df4['Route Description'].str.contains(search_bus, case=False)]

# Display the filtered dataframe
# st.markdown('<div class="header">Filtered Dataframe:</div>', unsafe_allow_html=True)
# st.dataframe(filtered_df)

# Display the analysis and graphs
if not filtered_df.empty:
    st.markdown('<div class="header">Analysis and Graphs:</div>', unsafe_allow_html=True)

    # Histogram
    st.markdown('<div class="subheader">Frequency of buses wrt source</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    filtered_df.hist(ax=ax)
    st.pyplot(fig)

    # Barplot
    st.subheader("Frequency of buses wrt kilometer")
    sns.barplot(x="Frequency", y="Kilometer", data=filtered_df,
                label="Total")
    plt.xlim(0, 15)
    st.pyplot(plt)

    # Distribution plot
    st.subheader("Frequency of buses distribution")
    sns.distplot(filtered_df['Kilometer'], hist=True, kde=True,
                 bins=int(180/5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    st.pyplot(plt)

    # Line plot
    st.markdown('<div class="subheader">Frequency of each bus</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(data=filtered_df, palette="tab10", linewidth=2.5, ax=ax)
    st.pyplot(fig)

    # Linear regression
    st.title("Prediction of Frequency wrt Kilometer")
    filtered_df.dropna(subset=['Frequency'], inplace=True)  # Drop rows with missing values in 'Frequency'
    x = filtered_df['Kilometer'].values.reshape(-1, 1)
    y = filtered_df['Frequency'].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Display the result
    st.subheader("Result")
    st.write("Intercept:", lr.intercept_[0])
    st.write("Coefficient:", lr.coef_[0][0])

    # Scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Kilometer', y='Frequency', data=filtered_df, ax=ax)
    sns.lineplot(x=x_test.flatten(), y=lr.predict(x_test).flatten(), color='red', ax=ax)
    plt.xlabel('Kilometer')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Linear regression
st.title("Prediction of Frequency based on Kilometer")
filtered_df.dropna(subset=['Frequency'], inplace=True)  # Drop rows with missing values in 'Frequency'
x = filtered_df['Kilometer'].values.reshape(-1, 1)
y = filtered_df['Frequency'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)

# Display the result
st.subheader("Result")
st.write("Intercept:", lr.intercept_[0])
st.write("Coefficient:", lr.coef_[0][0])

# Take input from the user
st.markdown('<div class="header">Enter kilometers:</div>', unsafe_allow_html=True)
kilometers = st.number_input("", value=0.0)

# Predict the frequency using the linear regression model
predicted_frequency = lr.predict([[kilometers]])

# Display the predicted frequency
st.markdown('<div class="subheader">Predicted Frequency:</div>', unsafe_allow_html=True)
st.write(predicted_frequency[0][0])
