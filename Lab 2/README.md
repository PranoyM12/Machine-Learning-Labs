# Machine Learning Lab 2: Linear Regression

**Student Name:** Pranoy Mridha  
**Roll No. :** 36  
**Section:** B
**Batch:** B3  
**Course:** Machine Learning Lab  

---

## Overview
This directory contains the implementation of a Linear Regression model trained to predict housing prices using the `USA_Housing.csv` dataset.

## Dataset Info
* **File:** `USA_Housing.csv`
* **Features:** Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population
* **Target Variable:** Price

## Workflow Summary
1. **Data Preprocessing & EDA:** Visualized feature correlations and handled data distribution plots.
2. **Train/Test Split:** Split the dataset (e.g., 80% train / 20% test) using `scikit-learn`.
3. **Model Training:** Fitted a `LinearRegression` model on the training features.
4. **Evaluation:** Measured performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and $R^2$ Score.

## Results
* **$R^2$ Score:** [Insert your value, e.g., 0.91]
* **Key Takeaway:** Area Income and Average House Age showed the strongest positive correlation with housing prices.

## How to Run
1. Navigate to this directory:
   ```bash
   cd ML-Lab/Lab-2
