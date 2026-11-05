# Lab 9: K-Means and Hierarchical Clustering for Customer Segmentation

## 1. Student Details

- **Name:** Pranoy Mridha
- **University:** Ramdeobaba University, Nagpur
- **Course:** B.Tech – Artificial Intelligence and Machine Learning
- **Subject:** Machine Learning Laboratory
- **Section:** B
- **Lab Batch:** B3
- **Roll Number:** 36
- **Lab Number:** 9
- **Date of Performance:** 30/10/2026

---

## 2. Lab Aim and Objectives

### Aim
To segment wholesale customers into distinct groups based on their annual spending using K-Means Clustering and Hierarchical (Agglomerative) Clustering, determine an optimal number of clusters, visualize the resulting customer segments, compare the two algorithms, and extract actionable business insights from the clusters.

### Objectives
1. Load and inspect the UCI Wholesale Customers dataset, verifying missing values, duplicate rows, and feature data types.
2. Perform univariate, bivariate, and multivariate exploratory data analysis (EDA) of customer spending distributions.
3. Detect potential outliers in spending columns using the Interquartile Range (IQR) method.
4. Scale spending columns using `StandardScaler` to ensure distance metric consistency.
5. Determine the optimal number of clusters using the Elbow Method and Silhouette Score analysis.
6. Implement the K-Means algorithm ($n_{\text{init}} = 10$) and evaluate the size and distribution of each cluster.
7. Implement Hierarchical (Agglomerative) Clustering using Ward linkage and construct both full and truncated dendrograms.
8. Visualize customer segments in two dimensions using Principal Component Analysis (PCA).
9. Compare K-Means and Hierarchical Clustering using Silhouette Scores and cluster size distributions.
10. Profile the clusters using the mean spending of each group and formulate targeted business strategies.

---

## 3. Dataset Information

- **Dataset Name:** UCI Wholesale Customers Dataset
- **Filename:** `Wholesale customers data.csv`
- **Total Records:** 440 customers
- **Total Columns:** 8 (2 categorical codes, 6 numerical continuous attributes)
- **Missing Values:** 0
- **Duplicate Rows:** 0

### Feature Breakdown

| Column Name | Data Type | Operational Role | Description |
|---|---|---|---|
| `Channel` | Integer | Excluded from Matrix $X$ | Category code ($1 = \text{Horeca}$, $2 = \text{Retail}$) |
| `Region` | Integer | Excluded from Matrix $X$ | Category code ($1 = \text{Lisbon}$, $2 = \text{Oporto}$, $3 = \text{Other}$) |
| `Fresh` | Integer | Continuous Input | Annual spending on fresh products |
| `Milk` | Integer | Continuous Input | Annual spending on milk products |
| `Grocery` | Integer | Continuous Input | Annual spending on grocery products |
| `Frozen` | Integer | Continuous Input | Annual spending on frozen products |
| `Detergents_Paper` | Integer | Continuous Input | Annual spending on detergents and paper products |
| `Delicassen` | Integer | Continuous Input | Annual spending on delicatessen products |

> **Feature Selection Note:** `Channel` and `Region` represent non-metric category codes rather than continuous financial measurements. Including them in Euclidean distance calculations would introduce artificial geometric bias. Therefore, they are excluded from the clustering feature matrix $X$ and retained solely for comparative EDA.

---

## 4. Exploratory Data Analysis (EDA)

### Categorical Feature Distribution
- **Channel Breakdown:**
  - `Channel 1` (Horeca - Hotel/Restaurant/Café): 298 customers (67.73%)
  - `Channel 2` (Retail): 142 customers (32.27%)
- **Region Breakdown:**
  - `Region 3` (Other): 316 customers (71.82%)
  - `Region 1` (Lisbon): 77 customers (17.50%)
  - `Region 2` (Oporto): 47 customers (10.68%)

### Univariate & Outlier Analysis
Visual inspection via histograms with KDE curves and boxplots confirms that all six spending features are heavily right-skewed. Most customers make modest annual purchases, while a few high-volume buyers form long distribution tails.

#### IQR Outlier Detection Analysis
Using the standard Interquartile Range bounds:
$$\text{IQR} = Q3 - Q1$$
$$\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q3 + 1.5 \times \text{IQR}$$

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % |
|---|---|---|---|---|---|---|---|
| `Fresh` | 3,127.75 | 16,933.75 | 13,806.00 | -17,581.25 | 37,642.75 | 20 | 4.55% |
| `Milk` | 1,533.00 | 7,190.25 | 5,657.25 | -6,952.88 | 15,676.13 | 28 | 6.36% |
| `Grocery` | 2,153.00 | 10,655.75 | 8,502.75 | -10,601.13 | 23,409.88 | 24 | 5.46% |
| `Frozen` | 742.25 | 3,554.25 | 2,812.00 | -3,475.75 | 7,772.25 | 43 | 9.77% |
| `Detergents_Paper` | 256.75 | 3,922.00 | 3,665.25 | -5,241.13 | 9,419.88 | 30 | 6.82% |
| `Delicassen` | 408.25 | 1,820.25 | 1,412.00 | -1,709.75 | 3,938.25 | 27 | 6.14% |

*Handling Decision:* All detected outliers exist above the upper bound and represent real, high-volume wholesale commercial clients. They were retained in the dataset to preserve authentic commercial behavior patterns.

### Key Bivariate & Multivariate Insights
1. **Channel Spending Differentiation:**
   - Retail buyers (`Channel 2`) spend substantially more on `Grocery` (mean: 16,322.9 vs 3,962.1), `Milk` (mean: 10,716.5 vs 3,451.7), and `Detergents_Paper` (mean: 7,269.5 vs 790.6).
   - Horeca buyers (`Channel 1`) spend more on `Fresh` (mean: 13,475.6 vs 8,904.3) and `Frozen` (mean: 3,748.3 vs 1,652.6).
2. **Regional Invariance:** Means across `Region 1`, `Region 2`, and `Region 3` differ minimally across features (e.g., mean `Fresh` ranges between 9,887.7 and 12,533.5), confirming `Region` provides little segment separation.
3. **Correlation Analysis:**
   - Very strong positive linear relationship between `Grocery` and `Detergents_Paper` ($r = 0.92$).
   - Strong positive relationship between `Milk` and `Grocery` ($r = 0.73$), and `Milk` and `Detergents_Paper` ($r = 0.66$).
   - Near-zero correlation between `Region` and all numerical spending columns ($r \in [-0.02, 0.06]$).

---

## 5. Important Machine Learning Concepts

1. **Unsupervised Learning:** Machine learning tasks where models analyze unlabeled data matrices to discover structural patterns, density variations, or clusters without explicit ground-truth labels.
2. **K-Means Clustering:** A centroid-based partitioning algorithm that divides $n$ observations into $K$ distinct clusters by iteratively minimizing the Within-Cluster Sum of Squares (WCSS / Inertia):
   $$\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$
3. **Elbow Method:** A technique plotting WCSS against values of $K$. The optimal cluster count is identified at the "elbow" curve bend, beyond which additional clusters produce marginal reductions in inertia.
4. **Silhouette Score:** Evaluates cluster cohesion and separation quality for sample $i$:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. The overall metric ranges from $-1$ to $+1$.
5. **Hierarchical Agglomerative Clustering:** A bottom-up clustering technique where every observation starts as an individual cluster and pairs are iteratively merged based on distance metrics.
6. **Ward Linkage Method:** An agglomerative merging metric that minimizes the increase in total within-cluster variance at each iteration step:
   $$\Delta \text{Var} = \frac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$
7. **Dendrogram:** A tree diagram illustrating the sequential hierarchy of cluster merges, where horizontal join heights represent Euclidean merge distances.
8. **Feature Scaling (`StandardScaler`):** Standardizes features to zero mean ($\mu = 0$) and unit standard deviation ($\sigma = 1$), ensuring attributes with larger scales (e.g., `Fresh` up to 112,151) do not dominate Euclidean metric spaces.
9. **Principal Component Analysis (PCA):** A dimensionality reduction technique that orthogonally transforms correlated variables into uncorrelated principal components while maximizing variance retention.

---

## 6. Preprocessing & Feature Scaling Execution

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Exclude non-spending categorical variables
X = df.drop(columns=['Channel', 'Region'])

# Standardize numerical spending attributes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Verification of transformation
print("Scaled Means:\n", X_scaled_df.mean().round(3))  # Yields 0.0 across all columns
print("Scaled Std Devs:\n", X_scaled_df.std(ddof=0).round(3))  # Yields 1.0 across all columns
```

---

## 7. Model Implementation & Clustering Analysis

### Cluster Selection Dynamics ($K$)

#### 1. Elbow Method Curve (WCSS / Inertia)
- $K=1$: $\text{WCSS} = 2,640.00$
- $K=2$: $\text{WCSS} = 1,960.00$
- $K=3$: $\text{WCSS} = 1,610.00$
- $K=4$: $\text{WCSS} = 1,320.00$
- $K=5$: $\text{WCSS} = 1,060.00$
- *Observation:* The curve exhibits a gradual inflection between $K=3$ and $K=6$ without a single sharp elbow, necessitating Silhouette evaluation.

#### 2. Silhouette Score Evaluation ($K=2$ to $10$)

| Number of Clusters ($K$) | Silhouette Score |
|---|---|
| 2 | 0.5472 |
| **3** | **0.5483** (Highest / Optimal) |
| 4 | 0.3485 |
| 5 | 0.3690 |
| 6 | 0.3782 |
| 7 | 0.3343 |
| 8 | 0.3201 |
| 9 | 0.3090 |
| 10 | 0.3112 |

**Selected Model Parameter:** $K = 3$ yields the highest Silhouette Score ($0.5483$).

---

### K-Means Execution ($K=3, n_{\text{init}}=10$)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled_df)
```

- **Silhouette Score:** `0.5483`
- **Cluster Distributions:**
  - Cluster 0: **45 customers** (10.23%)
  - Cluster 1: **393 customers** (89.32%)
  - Cluster 2: **2 customers** (0.45%)

---

### Hierarchical Clustering Execution ($K=3$, Ward Linkage)

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# Construct Ward linkage matrix
Z = linkage(X_scaled_df, method="ward", metric="euclidean")

# Agglomerative Model
hierarchical_model = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
hierarchical_labels = hierarchical_model.fit_predict(X_scaled_df)
```

- **Silhouette Score:** `0.2646`
- **Cluster Distributions:**
  - Cluster 0: **6 customers** (1.36%)
  - Cluster 1: **281 customers** (63.86%)
  - Cluster 2: **153 customers** (34.77%)

---

### PCA 2D Visualization Analysis
- **Components Selected:** 2
- **Cumulative Variance Explained:** **72.46%**

PCA mapping reveals that K-Means Cluster 1 forms a dense mass near the origin, Cluster 0 stretches along PC1 (high grocery/milk spending), and Cluster 2 consists of two extreme high-variance points at the top of the plot space.

---

## 8. Final Model Comparison

| Evaluation Metric | K-Means Clustering ($K=3$) | Hierarchical Agglomerative ($K=3$) |
|---|---|---|
| **Linkage / Algorithm Type** | Centroid-based iterative assignment | Bottom-up Ward variance minimization |
| **Silhouette Score** | **0.5483** | 0.2646 |
| **Cluster 0 Size** | 45 customers | 6 customers |
| **Cluster 1 Size** | 393 customers | 281 customers |
| **Cluster 2 Size** | 2 customers | 153 customers |
| **Population Balance** | Severely imbalanced (89.3% in Cluster 1) | Moderately balanced (63.9% / 34.8% / 1.4%) |

### Technical Comparative Summary
1. **Silhouette Performance:** K-Means achieved a substantially higher Silhouette Score ($0.5483$ vs $0.2646$). This is driven by K-Means pulling extreme outliers into tiny isolated clusters (Cluster 2 with 2 points), leaving the majority of points in a tight core cluster.
2. **Operational Utility:** Hierarchical Clustering produced a lower Silhouette Score but delivered a more balanced operational split ($281$ mainstream spenders vs $153$ intermediate spenders), making its groupings more practical for targeted marketing divisions.

---

## 9. Customer Profiling & Business Insights

Mean feature spending per K-Means cluster (original monetary values):

| Spending Category | Cluster 0 ($N=45$) | Cluster 1 ($N=393$) | Cluster 2 ($N=2$) | Overall Dataset Mean |
|---|---|---|---|---|
| `Fresh` | 10,440.93 | 12,062.91 | **34,782.00** | 12,000.30 |
| `Milk` | **19,386.42** | 4,115.10 | **30,367.00** | 5,796.27 |
| `Grocery` | **28,656.09** | 5,534.97 | 16,898.00 | 7,951.28 |
| `Frozen` | 2,190.24 | 2,940.68 | **48,701.50** | 3,071.93 |
| `Detergents_Paper` | **13,327.80** | 1,696.17 | 755.50 | 2,881.49 |
| `Delicassen` | 2,374.20 | 1,299.11 | **26,776.00** | 1,524.87 |

### Strategic Business Recommendations

1. **Cluster 0: Retail Supermarket / Grocer Bulk Buyers ($N=45$)**
   - *Profile:* Extremely high spending on `Milk` (19,386.4), `Grocery` (28,656.1), and `Detergents_Paper` (13,327.8). Aligns strongly with Channel 2 Retail profiles.
   - *Strategy:* Provide bulk volume discounting, tiered loyalty rebate structures, and bundled procurement promotions on non-perishable goods and cleaning supplies.

2. **Cluster 1: Standard Wholesale / Small Business Segment ($N=393$)**
   - *Profile:* Low-to-moderate baseline spending across all product categories. Comprises ~89% of the overall client base.
   - *Strategy:* Deploy automated reordering systems, standardized catalog promotions, and cross-selling campaigns to incrementally raise basket sizes.

3. **Cluster 2: Key Account VIP Outliers ($N=2$)**
   - *Profile:* Exceptional, massive expenditure in `Frozen` (48,701.5), `Fresh` (34,782.0), `Milk` (30,367.0), and `Delicassen` (26,776.0).
   - *Strategy:* Provide white-glove executive account management, specialized cold-chain logistics, and bespoke contract pricing agreements.

---

## 10. Key Learnings

1. Applied unsupervised clustering algorithms to unlabeled continuous commercial datasets.
2. Verified the mathematical necessity of `StandardScaler` feature standardization for distance-based estimators.
3. Used the Elbow Method (WCSS) and Silhouette Score to objectively select cluster counts ($K=3$).
4. Analyzed how extreme right-skewed outliers skew centroid placement in K-Means, causing unbalanced cluster sizes.
5. Interpreted dendrogram trees and Ward linkage distances to determine natural hierarchical split boundaries.
6. Reduced 6-dimensional feature spaces to 2D using PCA (explaining 72.46% variance) for intuitive visual segment evaluation.

---

## 11. Viva Questions & Answers

**Q1. What is unsupervised learning, and how does K-Means operate?**
*Ans:* Unsupervised learning finds intrinsic structural patterns in unlabeled data. K-Means initializes $K$ centroids, assigns each sample to its nearest centroid using Euclidean distance, recalculates centroids as the mean of assigned cluster samples, and iterates until convergence, minimizing WCSS.

**Q2. How do the Elbow Method and Silhouette Score differ in selecting $K$?**
*Ans:* The Elbow Method measures total internal cluster variance (WCSS) drop and searches for a visual curve inflection. The Silhouette Score quantifies both intra-cluster cohesion ($a$) and inter-cluster separation ($b$) on a scale of $-1$ to $+1$. Silhouette analysis provides an objective metric when WCSS curves bend gradually.

**Q3. What is a dendrogram, Ward linkage, and how are clusters extracted?**
*Ans:* A dendrogram is a tree visual representing sequential hierarchical merges. Ward linkage merges cluster pairs that minimize total within-cluster variance increase. Cutting the dendrogram horizontally at a chosen distance threshold determines the final cluster count.

**Q4. Why is feature scaling mandatory prior to distance-based clustering?**
*Ans:* Clustering relies on distance calculations:
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
Unscaled features with huge ranges (e.g., `Fresh` up to 112,151) will completely dominate distance metrics compared to attributes with smaller ranges (e.g., `Delicassen` averaging 1,525). `StandardScaler` normalizes all features to equal variance ($\sigma = 1$).

**Q5. K-Means achieved a higher Silhouette Score (0.5483) than Hierarchical Clustering (0.2646), but its clusters are 45, 393, and 2. Which is better?**
*Ans:* K-Means achieved a higher score by isolating 2 extreme outliers into Cluster 2 and forming a compact group of 45 bulk buyers, leaving 89% of data in one massive group. While mathematically cohesive, this provides limited operational value. Hierarchical Clustering (sizes 6, 281, 153) yields a lower score but provides a more balanced segmentation for targeted marketing campaigns.

---

## 12. Required Libraries & Dependencies

### Python Environment
- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `jupyter`

### Installation Command
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

---

## 13. Repository Structure

```text
ML-Lab-Submissions/
│
└── Lab-9-Clustering-Customer-Segmentation/
    ├── README.md
    ├── ML-PL9.ipynb
    └── Wholesale customers data.csv
```

---

## 14. Instructions to Run the Notebook

1. **Clone the Repository:**
   ```bash
   git clone [Add your GitHub repository URL here]
   cd ML-Lab-Submissions/Lab-9-Clustering-Customer-Segmentation
   ```

2. **Verify Data File:**
   Ensure `Wholesale customers data.csv` is present in the directory alongside `ML-PL9.ipynb`.

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook ML-PL9.ipynb
   ```

4. **Execute Pipeline:**
   Select **Kernel** $\rightarrow$ **Restart & Run All** to run all data processing, EDA plots, clustering algorithms, dendrogram visualizations, and comparative tables.

---

## 15. Conclusion

This laboratory practical successfully demonstrated customer segmentation on the UCI Wholesale Customers dataset using K-Means and Hierarchical Agglomerative Clustering. Feature standardization via `StandardScaler` was validated as a critical step. Silhouette Score analysis identified $K=3$ as the optimal cluster count. K-Means created highly cohesive but unbalanced clusters due to outlier isolation, whereas Hierarchical Clustering delivered a more operationally balanced customer split. Profiling cluster means enabled targeted business strategies across retail grocers, standard wholesale clients, and high-volume VIP accounts.
