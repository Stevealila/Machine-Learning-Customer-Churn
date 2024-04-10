# Telco Customer Churn Prediction using Artificial Neural Networks

This repository contains code for predicting customer churn in a telecom company using machine learning techniques.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It consists of various features related to telecom services and customer attributes.

## Technologies Used
- pandas
- numpy
- matplotlib
- tensorflow
- scikit-learn

## Preprocessing
- Read the dataset using pandas.
- Removed the 'customerID' column.
- Encoded categorical variables using one-hot encoding.
- Converted binary categorical variables to integers.
- Converted gender feature to binary integer representation.
- Handled missing values in 'TotalCharges' column.
- Scaled numerical features using MinMaxScaler.

## Model Training
- Split the data into training and testing sets.
- Built a simple neural network model using TensorFlow's Keras API.
- Compiled the model with binary crossentropy loss and Adam optimizer.
- Trained the model on the training data.
- Evaluated the model's performance on the testing data.

## Results
The model achieved an accuracy of 73.6% on the testing data.

For further details, refer to the code in the repository.
