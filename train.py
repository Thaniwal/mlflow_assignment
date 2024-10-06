#!/usr/bin/env python
# coding: utf-8

# Importing Necessary Packages
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('hour.csv')

# Data Preprocessing
df['dteday'] = pd.to_datetime(df['dteday'])
df['season'] = df['season'].astype('category')
df['yr'] = df['yr'].astype('category')
df['mnth'] = df['mnth'].astype('category')
df['hr'] = df['hr'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weathersit'] = df['weathersit'].astype('category')

# Feature Engineering
df["temp_humidity"] = df["temp"] * df["hum"]
df["atemp_windspeed"] = df["atemp"] * df["windspeed"]

# Splitting data
X = df.drop(['instant', 'dteday', 'cnt'], axis=1)
y = df['cnt']

# Find categorical and numerical columns
categorical_features = X.select_dtypes(include=['category']).columns
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

# Define pipelines
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment tracking
mlflow.set_experiment("Bike_Sharing_Experiment")

def log_model_metrics(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} - MSE: {mse}, R^2: {r2}")
    
# Train and log Linear Regression model
linear_regression_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', LinearRegression())])
log_model_metrics(linear_regression_model, X_train, X_test, y_train, y_test, "Linear_Regression")

# Train and log Random Forest model
random_forest_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor(random_state=42))])
log_model_metrics(random_forest_model, X_train, X_test, y_train, y_test, "Random_Forest")
