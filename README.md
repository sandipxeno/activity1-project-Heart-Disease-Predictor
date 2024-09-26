# activity1-project-Heart-Disease-Predictor
# Heart Disease Prediction Project

This project aims to build a machine learning model to predict the likelihood of heart disease in patients based on various health metrics. The process includes data cleaning, model training, and model evaluation. This repository contains all the necessary scripts, data, and documentation to reproduce the analysis.

## Project Overview

Heart disease is a leading cause of death worldwide, and predicting its occurrence can help in early intervention and treatment. The dataset used in this project includes patient information such as age, gender, cholesterol levels, and more. The goal is to classify whether a patient has heart disease based on these features.

### Key Features:
- Data cleaning and preprocessing with data_cleaning.py
- Model training using a Random Forest classifier with train_model.py
- Model evaluation using accuracy and classification reports with evaluate_model.py
- Exploratory Data Analysis (EDA) in EDA_Heart_Disease.ipynb

## Project Structure
## Dataset

The dataset used in this project is located in the data/heart_disease_dataset.csv file. The dataset contains the following features:
- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- Oldpeak (ST depression induced by exercise relative to rest)
- Slope (the slope of the peak exercise ST segment)
- Target (the presence or absence of heart disease)

The cleaned version of the dataset is saved as cleaned_heart.csv after running the data cleaning script.

## Prerequisites

Before running any scripts, ensure you have the required Python packages installed. You can install them using pip:

```bash
pip install -r requirements.txt