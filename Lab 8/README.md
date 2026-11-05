
# Lab 8: Support Vector Machine (SVM) for Machine Failure Prediction

## Student Details

- **Name:** Pranoy Mridha
- **Roll Number:** 36
- **Section:** B
- **Lab Batch:** B3
- **Subject:** Machine Learning
- **Lab Number:** 8

---

## Aim

To perform Exploratory Data Analysis (EDA) and build Support Vector Machine (SVM) classifiers with a Linear kernel, an RBF kernel, and tuned hyperparameters to predict machine failure using sensor readings and product type from the AI4I 2020 Predictive Maintenance Dataset.

## Objectives

- Load and inspect the dataset and check for missing values and duplicate rows.
- Perform univariate, bivariate, and multivariate Exploratory Data Analysis.
- Analyse the class imbalance of the target variable, Machine failure.
- Detect potential outliers in numerical sensor features using the Interquartile Range (IQR) method.
- Perform a stratified 80:20 train-test split.
- Apply StandardScaler and OneHotEncoder using ColumnTransformer and Pipeline.
- Train SVM models using Linear and RBF kernels.
- Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
- Tune SVM hyperparameters using GridSearchCV.
- Compare the performance of all three models.

## Dataset Description

The experiment uses the **AI4I 2020 Predictive Maintenance Dataset** (`ai4i2020.csv`).

- Total records: 10,000
- Total columns: 14
- Target variable: Machine failure
- Dataset type: Predictive Maintenance

### Dataset Features

| Feature | Description |
|---|---|
| UDI | Unique identifier |
| Product ID | Product identification number |
| Type | Product quality type (L, M, H) |
| Air temperature [K] | Air temperature in Kelvin |
| Process temperature [K] | Process temperature in Kelvin |
| Rotational speed [rpm] | Rotational speed in RPM |
| Torque [Nm] | Torque in Newton-metres |
| Tool wear [min] | Tool wear in minutes |
| Machine failure | 0 = No Failure, 1 = Failure |
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

**Note:** Identifier columns and failure-mode flags are not used as model inputs. The failure-mode flags directly describe failure information.

## Technologies and Libraries Used

- Python 3
- Jupyter Notebook / Google Colab
- VS Code (Optional)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Data Loading and Inspection

- Import the required Python libraries.
- Load `ai4i2020.csv` using Pandas.
- Inspect the dataset using `head()`, `tail()`, `shape`, `columns`, `dtypes`, `info()`, and `describe()`.
- Check for missing values and duplicate records.

### 2. Exploratory Data Analysis (EDA)

**Univariate Analysis:**
- Analyse the distribution of the target variable.
- Examine product Type counts.
- Generate histograms with KDE curves.
- Generate boxplots for numerical features.

**Outlier Detection:**
- Calculate Q1, Q3, and IQR.
- Identify potential outliers using the IQR rule.
- Examine outlier counts and percentages for numerical features.

**Bivariate Analysis:**
- Study numerical features against Machine failure.
- Analyse failure rates by product Type.
- Generate scatter plots.

**Multivariate Analysis:**
- Generate a correlation heatmap.
- Generate a pair plot to examine relationships between features.

### 3. Data Preprocessing

- Separate independent features (X) and target variable (y).
- Perform an 80:20 stratified train-test split using `random_state=42`.
- Apply StandardScaler to numerical features.
- Apply OneHotEncoder to the categorical Type feature.
- Use ColumnTransformer and Pipeline to combine preprocessing and model training.

### 4. Linear SVM Model

- Create an SVM classifier using `SVC(kernel="linear")`.
- Train the model using the training dataset.
- Predict machine failures on the testing dataset.
- Evaluate the model using classification metrics and a confusion matrix.

### 5. RBF SVM Model

- Create an SVM classifier using `SVC(kernel="rbf")`.
- Train and evaluate the model.
- Compare its performance with the Linear SVM model.

### 6. Hyperparameter Tuning

Use GridSearchCV with the following parameter grid:

- `C`: [1, 10]
- `kernel`: ["linear", "rbf"]
- Cross-validation: 3-fold
- Scoring metric: F1-score

The best-performing model is selected based on the mean cross-validation F1-score.

## Important Concepts

### Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm used for classification. It finds an optimal hyperplane that separates different classes while maximising the margin between them.

### Support Vectors

Support vectors are the nearest data points to the decision boundary. They determine the position of the separating hyperplane.

### Linear Kernel

The Linear kernel creates a straight decision boundary between classes.

### RBF Kernel

The Radial Basis Function (RBF) kernel helps SVM identify non-linear decision boundaries by measuring similarity between data points.

### Regularisation Parameter (C)

The parameter C controls the trade-off between maximising the margin and minimising classification errors.

- Small C: Wider margin with more permitted classification errors.
- Large C: Narrower margin with fewer training errors, but a greater risk of overfitting.

### Feature Scaling

StandardScaler is used because SVM is sensitive to differences in feature scales.

### Class Imbalance

The dataset contains significantly more records of No Failure than Failure. Therefore, accuracy alone is not sufficient to evaluate the model.

## Experimental Results

### Dataset Inspection

| Parameter | Result |
|---|---:|
| Total Records | 10,000 |
| Total Columns | 14 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| No Failure Records | 9,661 |
| Failure Records | 339 |
| Failure Percentage | 3.39% |

### Train-Test Split

| Dataset | Number of Samples |
|---|---:|
| Training Set | 8,000 |
| Testing Set | 2,000 |

The split was performed using stratification to preserve the original class distribution.

### 1. Linear SVM Results

| Metric | Score |
|---|---:|
| Accuracy | 0.966 |
| Precision (Failure) | 0.000 |
| Recall (Failure) | 0.000 |
| F1-Score (Failure) | 0.000 |

The Linear SVM achieved 96.6% accuracy but failed to identify any machine failures. It predicted No Failure for every test sample.

**Confusion Matrix:**

```text
[[1932    0]
 [  68    0]]
```

### 2. RBF SVM Results

| Metric | Score |
|---|---:|
| Accuracy | 0.972 |
| Precision (Failure) | 0.875 |
| Recall (Failure) | 0.206 |
| F1-Score (Failure) | 0.333 |

The RBF kernel detected some machine failures, achieving better precision, recall, and F1-score than the Linear SVM.

**Confusion Matrix:**

```text
[[1930    2]
 [  54   14]]
```

### 3. Hyperparameter Tuning Using GridSearchCV

The following parameters were tested:

| Parameter | Values |
|---|---|
| C | 1, 10 |
| Kernel | Linear, RBF |
| Cross-validation | 3-fold |
| Scoring | F1-score |

**Best Parameters:**

| Parameter | Best Value |
|---|---|
| C | 10 |
| Kernel | RBF |
| Best Cross-validation F1-score | 0.6324 |

### 4. Tuned SVM Results

| Metric | Score |
|---|---:|
| Accuracy | 0.979 |
| Precision (Failure) | 0.7708 |
| Recall (Failure) | 0.5441 |
| F1-Score (Failure) | 0.6379 |

The tuned RBF SVM identified 37 out of 68 actual machine failures.

## Final Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Linear SVM | 0.966 | 0.000 | 0.000 | 0.000 |
| RBF SVM | 0.972 | 0.875 | 0.206 | 0.333 |
| Tuned SVM | 0.979 | 0.7708 | 0.5441 | 0.6379 |

**Best Model:** Tuned RBF SVM (C = 10)

The tuned RBF SVM achieved the highest F1-score and identified more machine failures than the other two models.

## Key Learnings

- Understanding the working principle of Support Vector Machines.
- Understanding Linear and RBF kernels.
- Learning the importance of support vectors and margin maximisation.
- Performing Exploratory Data Analysis on a predictive maintenance dataset.
- Detecting outliers using the IQR method.
- Handling categorical and numerical features using preprocessing techniques.
- Understanding the importance of StandardScaler for SVM.
- Implementing ColumnTransformer and Pipeline.
- Evaluating classification models using accuracy, precision, recall, F1-score, and confusion matrices.
- Understanding class imbalance and the accuracy paradox.
- Performing hyperparameter tuning using GridSearchCV.

## Conclusion

In this practical, Support Vector Machine classifiers were implemented to predict machine failures using the AI4I 2020 Predictive Maintenance Dataset.

Exploratory Data Analysis revealed that the dataset was highly imbalanced, with only 3.39% of records representing machine failures. The Linear SVM achieved 96.6% accuracy but failed to identify any failures. The RBF SVM performed better by detecting some failure cases.

GridSearchCV selected the RBF kernel with C = 10 as the best configuration. The tuned model achieved 97.9% accuracy, 0.5441 recall, and 0.6379 F1-score for the Failure class.

The experiment demonstrated that accuracy alone can be misleading when working with imbalanced datasets. Precision, recall, F1-score, and the confusion matrix provide a more meaningful evaluation of machine failure prediction.

## Viva Questions and Answers

**Q1. What is a Support Vector Machine (SVM)?**

Ans. SVM is a supervised machine learning algorithm that separates different classes using an optimal hyperplane with the maximum possible margin.

**Q2. What are support vectors?**

Ans. Support vectors are the nearest data points to the decision boundary. They determine the position of the separating hyperplane.

**Q3. What is the difference between Linear and RBF kernels?**

Ans. The Linear kernel creates a straight decision boundary, whereas the RBF kernel can create non-linear decision boundaries.

**Q4. Why is StandardScaler used in SVM?**

Ans. SVM depends on distances and margins. StandardScaler brings numerical features to a comparable scale so that features with larger numerical ranges do not dominate.

**Q5. Why is F1-score used instead of accuracy for this dataset?**

Ans. The dataset is highly imbalanced, with only 3.39% machine failures. A model can achieve high accuracy by predicting No Failure for every sample. F1-score considers both precision and recall and provides a more meaningful evaluation of failure detection.

---

## Repository Structure

```text
ML-Lab-Submissions/
│
└── Lab-8-SVM-Machine-Failure/
    │
    ├── README.md
    ├── ML-PL8.ipynb
    └── ai4i2020.csv
```

## How to Run the Project

1. Clone or download the repository.
2. Open the `Lab-8-SVM-Machine-Failure` folder.
3. Ensure Python and the required libraries are installed.
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
