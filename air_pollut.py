import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Comprehensive Air Quality Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = r'E:\sem 5\AI LABS\lab 13\updated_pollution_dataset.csv'  
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Main dashboard
st.title("Comprehensive Air Quality Dashboard")

# Navigation
st.sidebar.header("Navigation")
options = [
    "Dataset Overview",
    "Summary Statistics",
    "Visualizations",
    "Correlation Analysis",
    "Health Risk Assessment",
    "Seasonal Impact",
    "Model Training and Evaluation",
]
selected_option = st.sidebar.radio("Select an option:", options)

if selected_option == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("### First 5 rows of the dataset:")
    st.dataframe(data.head())
    st.write("### Dataset Information:")
    st.write(data.info())
    st.write("### Missing Values:")
    st.write(data.isnull().sum())

elif selected_option == "Summary Statistics":
    st.subheader("Summary Statistics")
    st.write("### Descriptive Statistics:")
    st.write(data.describe())
    st.write("### Mode of Each Column:")
    st.write(data.mode().iloc[0])

elif selected_option == "Visualizations":
    st.subheader("Data Visualizations")

    st.write("### Distribution of Pollutants")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']], ax=ax, palette="viridis")
    st.pyplot(fig)

    st.write("### Temperature vs PM2.5")
    fig_temp = px.scatter(data, x='Temperature', y='PM2.5', color='Air Quality', title="Temperature vs PM2.5")
    st.plotly_chart(fig_temp)

    st.write("### Population Density vs PM2.5")
    fig_pop = px.scatter(data, x='Population_Density', y='PM2.5', color='Air Quality', title="Population Density vs PM2.5")
    st.plotly_chart(fig_pop)

elif selected_option == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    corr_matrix = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif selected_option == "Health Risk Assessment":
    st.subheader("Health Risk Assessment")
    data['Health_Risk'] = data['PM2.5'].apply(lambda x: 0 if x <= 50 else 1)
    st.write(data['Health_Risk'].value_counts())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=data['Health_Risk'], palette="pastel")
    ax.set_xticklabels(["Low", "High"])
    st.pyplot(fig)

elif selected_option == "Seasonal Impact":
    st.subheader("Seasonal Impact on Pollution")
    seasons = ['Spring', 'Summer', 'Monsoon', 'Autumn', 'Winter']
    data['Season'] = [seasons[i % len(seasons)] for i in range(len(data))]

    fig = px.box(data, x='Season', y='PM2.5', color='Season', title="Seasonal Impact on PM2.5")
    st.plotly_chart(fig)

elif selected_option == "Model Training and Evaluation":
    st.subheader("Model Training and Evaluation")

    # Fill missing values
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical variable
    label_encoder = LabelEncoder()
    data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])

    # Scale numerical data
    scaler = StandardScaler()
    numerical_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Temperature', 'Humidity']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Split data
    X = data.drop('Air Quality', axis=1)
    y = data['Air Quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display evaluation
    st.write("### Model Accuracy:")
    st.write(f"{accuracy_score(y_test, y_pred):.2f}")
    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)
