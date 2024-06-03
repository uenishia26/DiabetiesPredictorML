# DiabetiesPredictorML

This is a very simple Diabetes predictor ML model implemented using the K-Nearest Neighbors (KNN) algorithm.

## Overview
The goal of this project was to create a basic machine learning model to predict whether a patient has diabetes or not based on certain features. This project serves as a learning exercise to understand the basics of machine learning model development using Python and scikit-learn.

## Implementation Details
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Choice of K**: K was calculated using the square root of the number of samples in the dataset, aiming for an odd number of neighbors.
- **Performance Metrics**:
  - Accuracy: 79%
  - F1 Score: 70%
  
## Usage
1. **Requirements**: Make sure you have Python and necessary libraries (such as scikit-learn) installed.
2. **Dataset**: You'll need a dataset containing features and corresponding labels indicating whether a patient has diabetes or not. Ensure that the dataset is preprocessed and formatted appropriately for input to the model.
3. **Training the Model**: Run the provided Python script to train the model on your dataset.
4. **Evaluation**: Once trained, the model's performance can be evaluated using accuracy and F1 score metrics. 
5. **Testing**: You can use the trained model to make predictions on new, unseen data to predict whether a patient has diabetes.

## Notes
- This model is a basic implementation and may not achieve the highest accuracy or F1 score. It serves as a starting point for further exploration and experimentation with more advanced machine learning techniques.
- Feel free to modify and extend the code to improve the model's performance or explore different algorithms and techniques.


