# Bike Sharing Demand Prediction

## Project Overview

This project uses machine learning to predict hourly bike-sharing demand using historical rental, weather, and time-based data. The objective was to build and evaluate a predictive model while identifying the factors that most influence demand.

The analysis follows an end-to-end machine learning workflow, from exploratory data analysis and feature engineering to model development, validation, hyperparameter tuning, and final evaluation.

## Dataset

The dataset contains **10,886 hourly observations** of bike-sharing activity, including:

- Temperature and feels-like temperature
- Humidity and windspeed
- Weather conditions
- Season
- Holidays and working days
- Date and time
- Casual and registered rentals
- Total bike rental demand

The target variable is `count`, representing total hourly bike rentals.

## Analytical Approach

### 1. Data Exploration

I explored distributions, correlations, and relationships between bike demand and the available weather, calendar, and time-related variables.

An important issue identified during this stage was **target leakage**. The variables `casual` and `registered` are components of the target:

`count = casual + registered`

Although both variables were highly correlated with total demand, including them would give the model information that would not realistically be available when predicting demand. They were therefore excluded from model training.

### 2. Feature Engineering

I extracted additional information from the original datetime variable to better represent patterns in bike usage:

- Hour
- Day of week
- Month
- Year

Numerical variables were scaled and categorical variables were one-hot encoded using a preprocessing pipeline.

### 3. Model Development

I first trained a **Linear Regression** model to establish a baseline.

I then developed a **Random Forest Regressor** to capture nonlinear relationships between bike demand and factors such as weather, temperature, working days, and time of day.

Because the observations occur over time, I used **time-series cross-validation** rather than randomly shuffling observations during validation.

### 4. Model Optimization

I evaluated the Random Forest using:

- Time-series cross-validation
- Grid Search
- Randomized Search
- Training vs. validation error
- Feature importance analysis

The tuning process showed that the original Random Forest configuration performed best among the tested alternatives. The difference between training and validation performance also indicated that the model was overfitting, highlighting an opportunity for further regularization and model improvement.

## Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 131.27 | 98.38 | 0.636 |
| Random Forest | 94.44 | 58.45 | 0.812 |

The Random Forest substantially outperformed the Linear Regression baseline, reducing prediction error and explaining approximately **81% of the variation in bike-sharing demand** on the test data.

The final Random Forest achieved a **test RMSE of 94.44**, which was close to its time-series cross-validation RMSE of **93.22**, suggesting that the validation process provided a reasonable estimate of performance on unseen data.

## Key Insights

Feature importance analysis showed that several of the strongest predictors of bike demand were:

- Feels-like temperature (`atemp`)
- Time of day, particularly 5 PM, 6 PM, and 8 AM
- Humidity
- Year
- Whether the day was a working day

These results suggest that demand is strongly influenced by both **weather conditions and commuting patterns**, with several of the most important time periods aligning with typical morning and evening commute hours.

## Business Relevance

Accurately forecasting bike demand can help bike-sharing operators make better operational decisions, such as anticipating periods of high demand and planning bike availability accordingly.

Beyond predictive accuracy, this project demonstrates the importance of identifying data leakage, selecting validation methods appropriate for time-based data, comparing alternative models, and translating model outputs into interpretable insights.

## Tools & Technologies

**Python** · **Pandas** · **NumPy** · **Scikit-learn** · **Matplotlib** · **Jupyter Notebook**

### Techniques

Feature Engineering · Data Preprocessing · Linear Regression · Random Forest · Time-Series Cross-Validation · Grid Search · Randomized Search · Feature Importance

## Project File

[`bike_sharing_demand_prediction.ipynb`](bike_sharing_demand_prediction.ipynb) — Full exploratory analysis, preprocessing, model development, validation, tuning, and evaluation.
