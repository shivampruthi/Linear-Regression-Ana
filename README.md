# Linear Regression Analysis

This repository contains Python code for implementing simple linear regression using the least squares method and comparing it with the built-in linear regression model from scikit-learn.

## Overview

Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features. This repository demonstrates the implementation of linear regression from scratch using the ordinary least squares method.

## Contents

- `linear_regression.ipynb`: Jupyter Notebook containing the code for implementing linear regression and evaluating the model's performance.
- `data.csv`: Dataset used for the analysis, containing two columns representing the input feature and target variable.

## Usage

1. Ensure you have Python and the necessary libraries installed (NumPy, scikit-learn).
2. Clone this repository to your local machine.
3. Open the Jupyter Notebook `linear_regression.ipynb` and run the cells to see the implementation and evaluation of the linear regression model.

## Implementation Details

- `fit(x_train, y_train)`: Custom function to fit a linear regression model to the training data.
- `predict(x, m, c)`: Function to make predictions using the model coefficients.
- `score(y_truth, y_pred)`: Function to calculate the R-squared score of the model.
- `cost(y, x, m, c)`: Function to compute the cost of the model on the training data.

## Results

The implemented model's performance is compared with scikit-learn's built-in linear regression model. Here are the results:

- Training Score (Custom Model): 0.5795
- Testing Score (Custom Model): 0.6690
- Training Score (Scikit-Learn Model): 0.5795
- Testing Score (Scikit-Learn Model): 0.6690
- Model Coefficients (Custom Model): M = 1.3871, C = 4.8557
- Model Coefficients (Scikit-Learn Model): M = 1.3871, C = 4.8557
- Cost on Training Data (Custom Model): 132.63

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to contribute or report issues in this repository.
