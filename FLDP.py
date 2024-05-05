import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\rajan\OneDrive\Desktop\Final Project\loan_default_prediction_project.csv")  # Replace with your dataset filename

data = load_data()

# Split features and target variable
X = data.drop(columns=["Loan_Status"])  # Features
y = data["Loan_Status"]  # Target variable

# Define preprocessing steps for numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Data preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Preprocess the data and convert back to DataFrame
X_encoded = preprocessing_pipeline.fit_transform(X)
columns = numeric_features.tolist() + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
X_encoded_df = pd.DataFrame(X_encoded, columns=columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

# Define the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results using Streamlit
st.title("Loan Default Risk Prediction")

st.write("Accuracy:", accuracy)
st.write("Classification Report:\n", report)

# Display overall report
st.title("Overall Loan Default Risk Prediction")

# Predictions on the overall data
y_pred_overall = model.predict(X_encoded_df)

# Model evaluation for overall data
accuracy_overall = accuracy_score(y, y_pred_overall)

# Display overall accuracy in Streamlit
st.write("Overall Accuracy:", accuracy_overall)

# Sidebar widget to select Employment_Status for filtering
st.sidebar.title("Filter by Employment Status")
selected_status = st.sidebar.selectbox("Select Employment Status:", ["All"] + list(map(str, data["Employment_Status"].unique())))

# Filter data based on selected Employment_Status (if not "All")
if selected_status != "All":
    filtered_data = data[data["Employment_Status"] == selected_status]
    st.title(f"Loan Default Risk Prediction for Employment Status: {selected_status}")

    if not filtered_data.empty:
        # Split features and target variable for filtered data
        X_filtered = filtered_data.drop(columns=["Loan_Status"])  # Features
        y_filtered = filtered_data["Loan_Status"]  # Target variable

        # Preprocess the filtered data and convert back to DataFrame
        X_filtered_encoded = preprocessing_pipeline.transform(X_filtered)
        X_filtered_encoded_df = pd.DataFrame(X_filtered_encoded, columns=columns)

        # Predictions on the filtered data
        y_pred_filtered = model.predict(X_filtered_encoded_df)

        # Model evaluation for filtered data
        accuracy_filtered = accuracy_score(y_filtered, y_pred_filtered)

        # Display accuracy for filtered data in Streamlit
        st.write("Accuracy for Employment Status:", accuracy_filtered)
    else:
        st.write("No data available for the selected Employment Status.")

else:
    st.title("Loan Default Risk Prediction")
    st.write("Select an employment status from the sidebar to see the accuracy for that group.")