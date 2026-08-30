# Lab 4 – Decision Tree Classification

**Course:** Machine Learning

**Name:** Pranoy Mridha

**Roll No.:** B_B3_36

## 📌 Aim

To implement the **Decision Tree Classification** algorithm on a dataset and evaluate its performance using standard classification metrics, followed by hyperparameter tuning and cross-validation.

## 📊 Dataset

**Wine Quality (Red Wine) Dataset** — `winequality-red.csv`

- 11 physicochemical input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- Target variable: `quality` (integer score)

## 🛠️ Tech Stack / Libraries Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (`DecisionTreeClassifier`, `GridSearchCV`, `train_test_split`, `cross_val_score`, `learning_curve`)

## 🔎 Workflow

### 1. Data Understanding & Cleaning
- Loaded the dataset and inspected shape, data types, summary statistics
- Checked for missing values and duplicate rows

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Histograms with KDE and boxplots for every feature to study distribution and outliers
- **Outlier Detection:** IQR method used to compute lower/upper bounds and count outliers per feature
- **Bivariate Analysis:** Boxplots and scatter plots of each feature against `quality` (e.g., alcohol vs quality, volatile acidity vs quality)
- **Multivariate Analysis:** Correlation heatmap, pair plots, and quality-group-wise comparisons (`Low`, `Medium`, `High` quality bins)

### 3. Data Preparation
- Separated features (`X`) and target (`y = quality`)
- Split data into training (80%) and testing (20%) sets using `train_test_split` with stratification on `quality`

### 4. Model Building — Decision Tree Classifier
- Built a baseline `DecisionTreeClassifier` with `criterion='entropy'`
- Trained the model and generated predictions on train/test sets
- Visualized the full decision tree and a depth-limited (top 3 levels) version using `plot_tree`

### 5. Model Evaluation
- Metrics computed: **Accuracy, Precision, Recall, F1-score**
- Generated a full **classification report**
- Plotted a **confusion matrix** heatmap
- Plotted **multiclass ROC curves** with AUC for each quality class (One-vs-Rest)

### 6. Hyperparameter Tuning
- Used `GridSearchCV` (5-fold CV) to tune:
  - `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `ccp_alpha`
- Selected the best estimator based on cross-validated accuracy
- Re-evaluated the tuned model (train/test accuracy, classification report) and visualized the pruned tree

### 7. Cross Validation
- Performed 10-fold cross-validation on the best model to check consistency of accuracy across folds

### 8. Learning Curve
- Plotted training vs validation accuracy against training set size to diagnose bias/variance behavior

### 9. Prediction on New Data
- Predicted the wine quality for a new, custom sample input and displayed prediction probabilities

## 📈 Results

| Stage | Metric |
|---|---|
| Baseline Decision Tree | Train/Test Accuracy, Precision, Recall, F1 (see notebook output) |
| Tuned Decision Tree (GridSearchCV) | Improved generalization with reduced overfitting |
| 10-Fold Cross Validation | Mean accuracy and standard deviation reported |

*(Exact numeric values are available in the notebook outputs.)*

## 📁 Repository Structure

```
├── ML-Lab4.ipynb          # Main Jupyter Notebook
├── winequality-red.csv    # Dataset
└── README.md              # Project documentation
```

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
3. Open and run the notebook
   ```bash
   jupyter notebook ML-Lab4.ipynb
   ```

## 📝 Key Learnings

- Decision trees split data based on information gain (entropy criterion)
- Unpruned trees tend to overfit the training data
- Hyperparameter tuning (max depth, min samples per split/leaf, pruning via `ccp_alpha`) helps control model complexity
- Cross-validation and learning curves are essential to validate model generalization
