import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st



    # Load the dataset
df = pd.read_csv("dataset.csv")
df = df.drop(['Unnamed: 5'], axis=1)
df = df.dropna(axis=0)
df = df.drop_duplicates('Route Description', keep='first')
df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
df['Kilometer'] = pd.to_numeric(df['Kilometer'], errors='coerce')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create the Streamlit app
st.title('Route Frequency and Distance Analysis')
st.sidebar.title('Filters')

# Create filters
frequency_range = st.sidebar.slider('Frequency range', 1, 15, (1, 15))
kilometer_range = st.sidebar.slider('Kilometer range', 0, 100, (0, 100))

# Apply filters
filtered_df = df[(df['Frequency'] >= frequency_range[0]) & (df['Frequency'] <= frequency_range[1]) &
                 (df['Kilometer'] >= kilometer_range[0]) & (df['Kilometer'] <= kilometer_range[1])]

# Display data
st.write('Filtered data:')
st.write(filtered_df)

# Display plots
st.write('Frequency distribution plot:')
sns.histplot(filtered_df['Frequency'])
st.pyplot()

st.write('Kilometer distribution plot:')
sns.histplot(filtered_df['Kilometer'])
st.pyplot()

st.write('Frequency vs Kilometer heatmap:')
sns.heatmap(filtered_df[['Frequency', 'Kilometer']
                        ].corr(), annot=True, cmap='coolwarm')
st.pyplot()

# Run the app
if __name__ == '__main__':
    main()
