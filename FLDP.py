import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Loan Default Risk Prediction App",
    page_icon="📊",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': 'https://www.extremelycoolapp.com/bug',
        'About': '# Welcome to the Loan Default Risk Prediction App! This app predicts the risk of loan default based on various features.'
    }
)

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\rajan\OneDrive\Desktop\Final Project\loan_default_prediction_project.csv")

data = load_data()

# Replace missing values in 'Gender' column with the mode
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

# Replace missing values in 'Employment_Status' column with the mode
data['Employment_Status'].fillna(data['Employment_Status'].mode()[0], inplace=True)

# Split features and target variable
X = data.drop(columns=["Loan_Status"])  # Features
y = data["Loan_Status"]  # Target variable

# Define preprocessing steps for numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Add "Location" to categorical features
categorical_features = categorical_features.append(pd.Index(["Location"]))

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

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

# Sidebar widget to select Employment_Status and Location for filtering
st.sidebar.title("Filter by Employment Status and Location")
selected_status = st.sidebar.selectbox("Select Employment Status:", ["All"] + list(map(str, data["Employment_Status"].unique())))
selected_location = st.sidebar.selectbox("Select Location:", ["All"] + list(map(str, data["Location"].unique())))

# Filter data based on selected Employment_Status and Location (if not "All")
if selected_status != "All":
    filtered_data = data[data["Employment_Status"] == selected_status]
    if selected_location != "All":
        filtered_data = filtered_data[filtered_data["Location"] == selected_location]

    st.title(f"Loan Default Risk Prediction for Employment Status: {selected_status}, Location: {selected_location if selected_location != 'All' else 'All'}")

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
        st.write("Accuracy for Employment Status and Location:", accuracy_filtered)
    else:
        st.write("No data available for the selected Employment Status and Location.")
else:
    st.title("Loan Default Risk Prediction")
    st.write("Select an employment status and location from the sidebar to see the accuracy for that group.", accuracy_overall)

col1, col2 = st.columns(2)
with col1:
    # Plot accuracy for different Employment Status and Location
    accuracy_dict = {}
    for status in data["Employment_Status"].unique():
        for location in data["Location"].unique():
            filtered_data = data[(data["Employment_Status"] == status) & (data["Location"] == location)]
            if not filtered_data.empty:
                X_filtered = filtered_data.drop(columns=["Loan_Status"])  # Features
                y_filtered = filtered_data["Loan_Status"]  # Target variable
                X_filtered_encoded = preprocessing_pipeline.transform(X_filtered)
                y_pred_filtered = model.predict(X_filtered_encoded)
                accuracy_filtered = accuracy_score(y_filtered, y_pred_filtered)
                accuracy_dict[(status, location)] = accuracy_filtered

    # Create DataFrame from accuracy dictionary
    accuracy_df = pd.DataFrame(list(accuracy_dict.items()), columns=['Status_Location', 'Accuracy'])
    accuracy_df[['Employment_Status', 'Location']] = pd.DataFrame(accuracy_df['Status_Location'].tolist(), index=accuracy_df.index)
    accuracy_df.drop(columns=['Status_Location'], inplace=True)

    # Plot accuracy for different Employment Status and Location using Plotly
    st.write("Accuracy for Different Employment Statuses and Locations:")
    fig = px.pie(accuracy_df, values='Accuracy', names=accuracy_df['Employment_Status'] + ' - ' + accuracy_df['Location'])
    st.plotly_chart(fig)

with col2:

    # Display a bar chart to describe loan status for each employment status
    st.write("Loan Status for Each Employment Status:")
    employment_status_counts = data.groupby('Employment_Status')['Loan_Status'].value_counts().unstack().fillna(0)
    fig3 = px.bar(employment_status_counts, x=employment_status_counts.index, y=employment_status_counts.columns, labels={'x':'Employment Status', 'y':'Count'}, barmode='group')
    st.plotly_chart(fig3)
