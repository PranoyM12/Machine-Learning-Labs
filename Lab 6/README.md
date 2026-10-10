# ML Lab 6 – Seed Classification Using Artificial Neural Network

## 📌 Lab Details

* **Lab No.:** 6
* **Aim:** To implement the Perceptron Learning Algorithm.
* **Implementation:** Artificial Neural Network (ANN) for multiclass seed classification
* **Student Name:** Pranoy Mridha
* **Class / Section:** B-B3
* **Roll No.:** 36
* **Dataset:** `seeds_new.csv`

## 🎯 Objective

To analyze seed measurements and build a neural-network-based classification model that predicts the seed class from its numerical features. The practical includes data exploration, visualization, preprocessing, model training, prediction, and performance evaluation.

## 📂 Repository Contents

```text
Lab_6/
├── ML-Lab6.ipynb
├── seeds_new.csv
└── README.md
```

* **`ML-Lab6.ipynb`** – Jupyter Notebook containing the implementation, visualizations, model training, predictions, and evaluation.
* **`seeds_new.csv`** – Dataset containing seed measurements and their corresponding class labels.
* **`README.md`** – Documentation describing the experiment and its results.

## 📊 Dataset Description

The Seeds dataset contains **210 records** representing three seed classes, with 70 samples in each class.

The dataset contains seven numerical input features:

| Feature                 | Description             |
| ----------------------- | ----------------------- |
| Area                    | Area of the seed        |
| Perimeter               | Perimeter of the seed   |
| Compactness             | Compactness measurement |
| Length of kernel        | Kernel length           |
| Width of kernel         | Kernel width            |
| Asymmetry coefficient   | Asymmetry measurement   |
| Length of kernel groove | Kernel groove length    |
| `Class (1, 2, 3)`       | Target class label      |

The target variable contains three classes: 1, 2, and 3.

## 🛠️ Technologies and Libraries

* **Python** – Programming language
* **Jupyter Notebook** – Interactive development environment
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical computation
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical plots and heatmaps
* **Scikit-learn** – Data splitting, feature scaling, and evaluation metrics
* **TensorFlow / Keras** – Neural network construction and training

## ⚙️ Methodology

### 1. Data Loading and Inspection

Load the CSV dataset using Pandas and inspect the first few records, missing values, duplicate records, and class distribution.

### 2. Exploratory Data Analysis

Visualize the dataset using:

* Class distribution pie chart
* Histograms of numerical features
* Boxplots for feature distributions
* Feature correlation heatmap
* Area-versus-perimeter scatter plot
* Pair plot for comparing features across classes

### 3. Data Preprocessing

* Separate the input features and target labels.
* Encode the original class labels from 1–3 into 0–2 for model training.
* Split the dataset into 80% training data and 20% testing data using stratification.
* Standardize the features using `StandardScaler`.

### 4. Model Architecture

The notebook implements a feed-forward Artificial Neural Network with the following architecture:

```text
7 Input Features
       ↓
Dense Layer: 16 Neurons (ReLU)
       ↓
Dense Layer: 8 Neurons (ReLU)
       ↓
Output Layer: 3 Neurons (Softmax)
       ↓
Predicted Seed Class
```

The model is compiled using the Adam optimizer with a learning rate of 0.001 and sparse categorical cross-entropy as its loss function. Early stopping monitors validation loss to help prevent unnecessary training.

### 5. Model Training

Train the network for up to 100 epochs with a batch size of 16 and a validation split of 20% of the training data. Early stopping is configured with a patience of 10 epochs and restores the best model weights.

### 6. Prediction and Evaluation

Evaluate the trained model on the held-out test set. Generate predicted class labels and compare them with the actual labels using accuracy, weighted precision, weighted recall, weighted F1-score, a classification report, and a confusion matrix.

## 📈 Results

The following results were recorded in the notebook's evaluation output:

| Evaluation Metric  | Result |
| ------------------ | -----: |
| Test Accuracy      | 88.10% |
| Weighted Precision | 89.95% |
| Weighted Recall    | 88.10% |
| Weighted F1-Score  | 87.31% |
| Test Loss          | 0.2973 |

The model correctly classified approximately 88% of the test samples. The confusion matrix and classification report provide additional information about its performance across the three seed classes.

*Note: Neural-network training can produce slightly different results across runs unless random seeds and the execution environment are controlled.*

## ▶️ How to Run

### Prerequisites

Install Python and Jupyter Notebook, or use an environment that supports `.ipynb` files.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Lab 6 folder

```bash
cd ML-Lab-Submissions/Lab_6
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Run the notebook

```bash
jupyter notebook ML-Lab6.ipynb
```

Ensure that `seeds_new.csv` is in the same directory as the notebook. Execute the cells in order to reproduce the data analysis, training, predictions, and evaluation.

## 🧠 Key Learning Outcomes

* Loading and inspecting a structured dataset using Pandas.
* Performing exploratory data analysis and visualizing numerical features.
* Splitting data into training and testing sets.
* Standardizing input features.
* Building and training a multiclass neural-network classifier.
* Applying early stopping during model training.
* Evaluating a classification model using multiple performance metrics.

## 📝 Conclusion

This practical demonstrates a neural-network-based approach to classifying seeds using seven numerical measurements. After preprocessing and training, the model achieved approximately 88.10% test accuracy on the recorded run. The experiment illustrates the importance of data exploration, feature scaling, model training, and systematic evaluation in machine learning.

**Implementation note:** Although the prescribed lab aim refers to the Perceptron Learning Algorithm, the current notebook uses a multilayer ANN implemented with TensorFlow/Keras. A traditional single-layer perceptron would require a different implementation.

---

**GitHub Repository:** [ML Lab Submissions](YOUR_GITHUB_REPOSITORY_URL)

**Notebook:** [ML-Lab6.ipynb](./ML-Lab6.ipynb)

**Dataset:** [seeds_new.csv](./seeds_new.csv)
