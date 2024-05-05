
# Final-Capstone-Project-5
# Loan Default Risk Prediction Streamlit App
This Streamlit web application predicts loan default risk based on a dataset provided by the user. It includes the following features:

- Loading and preprocessing of data
- Training a logistic regression model
- Evaluating the model's accuracy and generating a classification report
- Filtering data by employment status and location, and displaying accuracy for each group

## Requirements
To run this application, you need the following dependencies:

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Input Data
The input data should be in CSV format and include the following columns:

- Age: Age of the borrower
- Gender: Gender of the borrower
- Income: Income of the borrower
- Employment_Status: Employment status of the borrower
- Location: Location of the borrower
- Credit_Score: Credit score of the borrower
- Debt_to_Income_Ratio: Debt to income ratio of the borrower
- Existing_Loan_Balance: Existing loan balance of the borrower
- Loan_Status: Whether the loan was defaulted or not
- Loan_Amount: Amount of the loan
- Interest_Rate: Interest rate of the loan
- Loan_Duration_Months: Duration of the loan in months

## Output
The app displays the following information:

- Overall Accuracy: Accuracy of the model on the entire dataset
- Classification Report: Detailed classification report including precision, recall, and F1-score
- Filtered Accuracy: Accuracy of the model for a specific employment status and location, selectable via the sidebar
