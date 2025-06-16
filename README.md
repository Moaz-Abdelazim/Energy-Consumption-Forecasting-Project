Energy Consumption Forecasting Project
Overview
This project focuses on forecasting energy consumption using weather data as features. The goal is to predict electricity usage based on environmental factors such as air temperature, dew temperature, wind speed, and others. The project leverages machine learning techniques, specifically a Random Forest Regressor, to model the relationship between weather conditions and energy consumption.
Dataset
The dataset consists of two main files:

weather.csv: Contains weather-related features including:
timestamp: Date and time of the observation
site_id: Identifier for the location
airTemperature, dewTemperature, windSpeed, windDirection, seaLvlPressure, precipDepth1HR, precipDepth6HR, cloudCoverage: Weather variables


electricity.csv: Contains energy consumption data (targets) for multiple sites, with 1579 columns representing different meters or locations.

Data Characteristics

Weather Data: 331,166 rows, 10 columns, with some missing values in features like cloudCoverage (170,987 missing) and precipDepth6HR (313,004 missing).
Electricity Data: 17,544 rows, 1,579 columns, representing energy consumption across various sites.

Project Structure
The project is implemented in a Jupyter Notebook (Energy_Consumption_Forecasting_Project.ipynb) with the following sections:

Import Libraries: Imports necessary Python libraries (pandas, numpy, matplotlib, seaborn, scikit-learn).
Data Upload: Loads the weather and electricity datasets.
Data Preparation: Examines the dataset shapes and handles missing values.
Exploratory Data Analysis (EDA): Analyzes the features, including data types and missing values.
Model Training: Uses a Random Forest Regressor with 800 estimators to predict energy consumption.
Model Evaluation: Evaluates the model using Mean Squared Error (MSE), R² Score, and Mean Absolute Error (MAE).

Key Results

MSE: 1.531
R² Score: 0.843
Mean Absolute Error: 0.886

Prerequisites
To run this project, ensure you have the following installed:

Python 3.6+
Jupyter Notebook
Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Install dependencies using:
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Clone the repository:git clone <repository-url>


Place the weather.csv and electricity.csv files in the project directory.
Open the Jupyter Notebook:jupyter notebook Energy_Consumption_Forecasting_Project.ipynb


Run the cells sequentially to preprocess the data, train the model, and evaluate the results.

Key Features

Data Preprocessing: Handles missing values and aligns weather and energy consumption data by timestamp and site.
Feature Engineering: Utilizes weather variables as predictors for energy consumption.
Machine Learning: Implements a Random Forest Regressor for robust multi-output regression.
Evaluation: Provides comprehensive metrics (MSE, R², MAE) to assess model performance.

Future Improvements

Address missing values in cloudCoverage and precipDepth6HR using advanced imputation techniques.
Explore additional features, such as time-based features (e.g., hour of day, day of week).
Test other models (e.g., Gradient Boosting, Neural Networks) for improved accuracy.
Incorporate cross-validation to ensure model robustness.

License
This project is licensed under the MIT License.
Contact
For questions or feedback, feel free to reach out via [Your Email] or [Your LinkedIn Profile].
