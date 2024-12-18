# Rotten_Tomatoes_ML_Pipeline

Predicting Audience Ratings Using a Machine Learning Pipeline.
This project demonstrates a machine learning pipeline built with scikit-learn to predict audience ratings (audience_rating) for movies based on various movie-related features. The dataset used for this project comes from an Excel file containing information on movies and ratings.

**Table of Contents**
Dataset Overview
Dependencies
Pipeline Steps
How to Use the Code
Model Evaluation
Improvement Ideas
**Dataset Overview**
The dataset contains the following key columns used in this project:

**Features:**
rating: Movie rating (e.g., PG, R, G).
genre: Movie genre (e.g., Comedy, Drama, Action).
runtime_in_minutes: Duration of the movie in minutes.
tomatometer_rating: Critics' rating from Rotten Tomatoes.
**Target:**
audience_rating: Audience rating percentage (our prediction target).
**Dependencies**
This project uses the following Python libraries:

pandas: For data manipulation.
scikit-learn: For building the machine learning pipeline and model.
**Install Dependencies**
bash
Copy code
pip install pandas scikit-learn openpyxl
Pipeline Steps
The pipeline consists of the following steps:

**Data Preprocessing:**
Numerical Features (runtime_in_minutes, tomatometer_rating):
Impute missing values with the mean.
Standardize the data using StandardScaler.
Categorical Features (rating, genre):
Impute missing values with the most frequent value.
One-hot encode the categorical features.
**Model:**
RandomForestRegressor: A regression model that uses multiple decision trees to improve predictive accuracy.
**Evaluation:**
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
R² Score: Indicates how well the model explains the variance in the target.

**How to Use the Code**
Load the Data:
Ensure the Excel file Rotten_Tomatoes_Movies3.xlsx is in the specified path. Load the data and explore the sheet names:
file_path = "/mnt/data/Rotten_Tomatoes_Movies3.xlsx"
xls = pd.ExcelFile(file_path)
print(xls.sheet_names)

Select Features and Target:

features = ['rating', 'genre', 'runtime_in_minutes', 'tomatometer_rating']
target = 'audience_rating'

Train-Test Split:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Build and Train the Pipeline:

pipeline.fit(X_train, y_train)


Predict and Evaluate:

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

**Model Evaluation**
After running the pipeline, the following evaluation metrics are obtained:

Mean Squared Error (MSE): 230.22
R² Score: 0.44
The model explains approximately 44% of the variance in the audience ratings.

