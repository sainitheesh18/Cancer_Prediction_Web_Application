# Cancer_Web-Application
 
## Overview
This repository contains the code for a full-stack web application that predicts the likelihood of cancer based on user-reported symptoms. The application leverages machine learning algorithms to analyze input data and provide predictions. The backend is built using Python, and the frontend provides an interface for users to enter their symptoms and receive predictions.

## Features
Symptom Input: Users can input their symptoms through a web interface.
Prediction: The application uses machine learning models to predict the likelihood of cancer based on the provided symptoms.
Model Training: The backend includes code for training and evaluating the machine learning models.

## Machine Learning Model
The machine learning model is built using the following:
Libraries: numpy, pandas, matplotlib, pickle, sklearn
Algorithms: Logistic Regression
Data: The model uses a dataset (Cancer.csv) that contains various features related to cancer symptoms.

## Data Preprocessing
The data preprocessing steps include:
Dropping non-numeric columns like 'Patient Id', 'Gender', and 'Age'.
Encoding categorical data into numeric format.
Normalizing the features.

## Model Training
The model is trained using the Logistic Regression algorithm and evaluated using accuracy score. The trained model is then saved to disk for future use.

The model is a Logistic Regression classifier trained on features derived from user symptoms. The model is saved as model.pkl, which can be loaded and used for prediction.
