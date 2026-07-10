# House-Prices-Prediction
An end-to-end machine learning project to predict house prices using Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, and ensemble learning on the Ames Housing dataset.

## Overview
This project predicts residential house prices using the Ames Housing dataset from Kaggle.
The objective is to build and compare multiple machine learning models, perform feature engineering, tune hyperparameters, and improve predictive performance using ensemble learning.

Dataset:
House Prices: Advanced Regression Techniques (Kaggle)


## Problem Statement
Predict the final sale price of residential homes in Ames, Iowa using various property features.


## Dataset
- Training samples: 1460
- Test samples: 1459
- Features: 79 (before preprocessing)

Target Variable:
- SalePrice


## Exploratory Data Analysis
The following analyses were performed:

- Distribution of SalePrice
- Correlation Heatmap
- Missing Value Analysis
- Outlier Detection
- Feature Relationships


## Data Preprocessing

### Missing Values
Missing values were handled using appropriate strategies:
- Numerical features → Median / Zero
- Categorical features → Mode / "None"

### Outlier Removal
Two extreme outliers were removed:
- Row 523
- Row 1298

### Feature Engineering
The following new features were created:
- HouseAge
- YearsSinceRemodel
- TotalSF
- TotalBathrooms

### Encoding
- Ordinal Encoding
- One-Hot Encoding

### Target Transformation
The target variable was transformed using:
```python
np.log1p(SalePrice)
```
to reduce skewness and improve model performance.


## Models Used
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Ensemble Learning


## Hyperparameter Tuning
RandomizedSearchCV with 5-Fold Cross Validation was used for tuning:

- Random Forest
- XGBoost
- LightGBM
- CatBoost


## Model Performance
| Model | Kaggle Score |
|---------|-------------|
| XGBoost | 0.13290 |
| LightGBM | 0.13144 |
| CatBoost | 0.13159 |
| **Ensemble (XGBoost + LightGBM + CatBoost)** | **0.12781** |


## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost


## Future Improvements
- Weighted ensemble learning
- Stacking
- Bayesian Hyperparameter Optimization (Optuna)
- Additional Feature Engineering
- Streamlit Deployment


## Kaggle Competition
House Prices: Advanced Regression Techniques
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


## Author

Anshika
