# Simple Linear Regression Project

This project demonstrates how to predict a student's test score based on the number of study hours using a simple linear regression model in R. The project includes data analysis, model fitting, and predictions.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Analysis](#analysis)
- [Assumptions](#assumptions)
- [Influential Point Detection](#influential-point-detection)
- [Predictions](#predictions)
- [Conclusion](#conclusion)
- [Contributing](#contributing)

## Introduction

The goal of this project is to predict the percentage of marks a student is expected to score based on the number of hours they studied. This is a simple linear regression task that involves just two variables: study hours and marks scored.

## Data

The dataset used in this project contains information about students' study hours and their corresponding scores. The data is loaded from a CSV file hosted at a URL.

## Analysis

The analysis includes the following steps:

- Loading and exploring the dataset.
- Checking assumptions of linear regression (linearity, normality, homoscedasticity, and autocorrelation).
- Fitting a simple linear regression model.
- Making predictions using the fitted model.
- Evaluating model performance using R-squared, RMSE, and MAE.

## Assumptions

The project verifies the following assumptions of simple linear regression:

- **Linearity**: There should be a linear relationship between study hours and scores.
- **Normality**: Residuals should be normally distributed.
- **Homoscedasticity**: Residuals should have constant variance.
- **Autocorrelation**: Residuals should be uncorrelated.
- **Mean of errors**: Errors should have a mean close to zero.

The analysis includes statistical tests and visualizations to check these assumptions.

## Influential Point Detection

The project checks for influential points using Cook's distance. Influential points are data points that significantly impact the model when removed. The analysis includes visualizations and statistical measures to identify influential points.

## Predictions

The project makes predictions using the fitted model:

- Predicting scores for test data.
- Evaluating model performance using R-squared, RMSE, and MAE.
- Predicting the score for a student studying 9.25 hours.

## Conclusion

The simple linear regression model performs well and explains a large proportion of the variance in scores. Model performance metrics (R-squared, RMSE, MAE) indicate a good fit.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes with a descriptive message.
- Push your branch and submit a pull request.
