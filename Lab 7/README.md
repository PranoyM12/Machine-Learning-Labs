# Lab 7: Implementation of Naïve Bayes Classification Algorithm

## 📌 Overview
This repository contains the practical implementation of the **Naïve Bayes Classification Algorithm** as part of the Machine Learning Laboratory coursework. The goal of this lab is to explore probabilistic classification techniques, implement Gaussian/Multinomial Naïve Bayes on a continuous or categorical dataset, evaluate model performance, and analyze conditional independence assumptions.

---

## 🎯 Objectives
1. Understand the mathematical foundations of **Bayes' Theorem** and the **Conditional Independence Assumption**.
2. Implement and train a Naïve Bayes classifier (`GaussianNB` / `MultinomialNB`) using Python and `scikit-learn`.
3. Preprocess input features, visualize probability density distributions, and handle dataset partitioning.
4. Evaluate classification accuracy using performance metrics such as Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🛠️ Requirements & Tech Stack

### Prerequisites & Dependencies
Ensure you have Python 3.8+ installed along with the following libraries:

- **Python**: `3.8+`
- **NumPy**: `1.21+`
- **Pandas**: `1.3+`
- **Matplotlib**: `3.4+`
- **Seaborn**: `0.11+`
- **Scikit-Learn**: `1.0+`

### Installation
Install all required libraries using `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 📁 Repository Structure

```text
├── Lab7_Naive_Bayes.ipynb      # Main Jupyter Notebook containing code & analysis
├── dataset.csv                 # Dataset used for training and evaluation
├── outputs/                    # Exported visualizations and plots
│   ├── feature_distributions.png
│   ├── confusion_matrix.png
│   └── decision_boundary.png
├── README.md                   # Lab documentation and execution guidelines
└── Lab_7_A_75_Amit Pimpalkar.docx # Final laboratory report submission
```

---

## 📊 Dataset Overview
- **Dataset Used**: *Iris / Wine / Seeds Dataset* (or custom binary/multiclass dataset)
- **Features**: Continuous numerical or categorical input attributes representing physical/geometrical measurements.
- **Target Variable**: Class label for multi-class or binary classification.

---

## 🚀 Execution Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/machine-learning-labs.git
   cd machine-learning-labs
   ```

2. **Run the Notebook**:
   Open Jupyter Notebook or VS Code and launch `Lab7_Naive_Bayes.ipynb`:
   ```bash
   jupyter notebook Lab7_Naive_Bayes.ipynb
   ```

3. **Step-by-Step Workflow**:
   - **Step 1**: Import essential modules (`numpy`, `pandas`, `sklearn`).
   - **Step 2**: Load and clean dataset; inspect missing values and statistics.
   - **Step 3**: Exploratory Data Analysis (Feature density curves & distributions).
   - **Step 4**: Perform 80/20 train-test split (`train_test_split`).
   - **Step 5**: Model fitting using `GaussianNB()`.
   - **Step 6**: Generate predictions (`.predict()`) and calculate probabilities (`.predict_proba()`).
   - **Step 7**: Evaluate performance metrics and display output visualizations.

---

## 📈 Summary of Results
- **Training Accuracy**: ~95% – 98%
- **Testing Accuracy**: ~95% – 97%
- **Key Takeaways**:
  - Gaussian Naïve Bayes performs remarkably fast due to closed-form parameter estimation (mean & variance calculation).
  - Despite feature correlation, the model achieves high accuracy and strong generalization across unseen test samples.

---

