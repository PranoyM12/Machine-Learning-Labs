
# Lab 5: K-Nearest Neighbours (KNN) for Diabetes Classification

## Student Details

- **Name:** Pranoy Mridha
- **Section:** B
- **Lab Batch:** B3
- **Roll Number:** 36
- **Subject:** Machine Learning

---

## Aim

To perform Exploratory Data Analysis (EDA) and implement the K-Nearest Neighbours (KNN) classification algorithm to predict whether a person has diabetes based on clinical measurements. The objective is to identify the optimal value of K and evaluate the model's performance.

## Dataset Description

The project uses the **Pima Indians Diabetes Dataset** (`diabetes.csv`).

- Total records: 768
- Total columns: 9
- Input features: 8
- Target variable: Outcome

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | 0 = No Diabetes, 1 = Diabetes |

## Technologies and Libraries Used

- Python 3
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Load the diabetes dataset using Pandas.
2. Perform initial data inspection and understand the dataset.
3. Handle invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI by replacing them with missing values.
4. Perform median imputation to handle missing values.
5. Conduct Exploratory Data Analysis (EDA).
6. Generate histograms, boxplots, scatter plots, pair plots, and a correlation heatmap.
7. Separate independent features and the target variable.
8. Split the dataset into training and testing sets using an 80:20 stratified split.
9. Apply StandardScaler to standardize the features.
10. Train a baseline KNN classifier with K = 5.
11. Evaluate the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
12. Experiment with different values of K.
13. Perform cross-validation and hyperparameter tuning using GridSearchCV.
14. Evaluate the tuned KNN model.
15. Predict diabetes for a new patient using the trained model.

## Model Implementation

### 1. Baseline KNN Model

The initial KNN classifier was trained using K = 5.

**Evaluation Results:**

| Metric | Score |
|---|---:|
| Accuracy | 75.32% |
| Precision | 0.6600 |
| Recall | 0.6111 |
| F1-Score | 0.6346 |
| ROC-AUC | 0.7886 |

Confusion Matrix:

```text
[[83 17]
 [21 33]]
```

### 2. Finding the Optimal K

Different values of K, ranging from 1 to 30, were tested.

- Best test accuracy: 76.62%
- Best K based on test accuracy: 8
- Best mean cross-validation accuracy: 77.70%
- Best K based on 10-fold cross-validation: 24

### 3. Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation to find the best combination of:

- Number of neighbours (`n_neighbors`)
- Distance weighting (`weights`)
- Distance metric (`metric`)

**Best Parameters:**

| Parameter | Best Value |
|---|---|
| n_neighbors | 19 |
| weights | uniform |
| metric | euclidean |
| Best CV Accuracy | 77.69% |

### 4. Tuned KNN Model Results

| Metric | Score |
|---|---:|
| Accuracy | 74.68% |
| Precision | 0.6667 |
| Recall | 0.5556 |
| F1-Score | 0.6061 |
| ROC-AUC | 0.8077 |

Confusion Matrix:

```text
[[85 15]
 [24 30]]
```

## New Patient Prediction

The tuned KNN model was tested using the following sample patient data:

| Feature | Value |
|---|---:|
| Pregnancies | 2 |
| Glucose | 150 |
| BloodPressure | 80 |
| SkinThickness | 30 |
| Insulin | 120 |
| BMI | 33.5 |
| DiabetesPedigreeFunction | 0.6 |
| Age | 45 |

**Prediction:** Diabetes

**Predicted Probability:** 63%

*Note: This is an educational machine learning prediction and not a medical diagnosis.*

## Key Learnings

- Understanding the K-Nearest Neighbours classification algorithm.
- Performing data cleaning and preprocessing.
- Handling missing and invalid values using median imputation.
- Performing Exploratory Data Analysis.
- Understanding the importance of feature scaling in distance-based algorithms.
- Implementing stratified train-test splitting.
- Evaluating classification models using different performance metrics.
- Selecting the optimal K value using cross-validation.
- Performing hyperparameter tuning using GridSearchCV.
- Making predictions using a trained classification model.

## Conclusion

In this experiment, the K-Nearest Neighbours algorithm was implemented to predict diabetes using clinical measurements from the Pima Indians Diabetes Dataset.

Data preprocessing, exploratory analysis, feature scaling, and hyperparameter tuning were performed to improve the classification process. Different values of K were evaluated, and GridSearchCV identified the best hyperparameter combination.

The experiment demonstrated how KNN can be applied to a healthcare-related classification problem and highlighted the importance of preprocessing, feature scaling, and model evaluation.

## Repository Structure

```text
ML-Lab-Submissions/
│
└── Lab-5-KNN-Diabetes/
    │
    ├── README.md
    ├── Lab5_KNN_Diabetes_Complete.ipynb
    └── diabetes.csv
```

## How to Run the Project

1. Clone or download the repository.
2. Open the `Lab-5-KNN-Diabetes` folder.
3. Ensure that Python and the required libraries are installed.
4. Open the Jupyter Notebook.
5. Run the notebook cells sequentially.

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

**Developed by:** Pranoy Mridha  
**Section:** B  
**Lab Batch:** B3  
**Roll Number:** 36
