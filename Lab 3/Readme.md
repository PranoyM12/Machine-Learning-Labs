# Lab 3 — Find-S and Candidate Elimination Algorithm

## Title
Implementation of Find-S and Candidate Elimination Algorithms for Concept Learning

## Aim
To implement the Find-S and Candidate Elimination algorithms on the EnjoySport dataset and study how each algorithm derives a hypothesis consistent with the given training examples.

## Objective
To understand the concept of hypothesis space search in machine learning by implementing:
- The **Find-S algorithm**, which finds the most specific hypothesis consistent with the positive training examples.
- The **Candidate Elimination algorithm**, which maintains a Specific Boundary (S) and a General Boundary (G) consistent with all training examples (both positive and negative).

## Dataset
**EnjoySport Dataset** (`enjoy.csv`)

## Dataset Description
The dataset contains 10 training examples describing weather conditions under which a person does or does not enjoy a sport. It has 6 attributes:

| Attribute | Example Values |
|---|---|
| Sky | Sunny, Rainy, Cloudy |
| AirTemp | Warm, Cold |
| Humidity | Normal, High |
| Wind | Strong, Weak |
| Water | Warm, Cool |
| Forecast | Same, Change |

The target attribute is **EnjoySport**, with values `Yes` / `No`.

## Technologies / Libraries Used
- Python 3
- Pandas
- NumPy

## Techniques Implemented
- **Find-S Algorithm** — starts with the most specific hypothesis and generalizes it using only positive (`Yes`) training examples; negative examples are ignored.
- **Candidate Elimination Algorithm** — maintains both a Specific Boundary (S) and a General Boundary (G); S is generalized on positive examples, and G is specialized on negative examples.

## Practical Workflow
1. Import required libraries (Pandas, NumPy).
2. Load the EnjoySport dataset using `pd.read_csv()`.
3. Inspect the dataset using `.head()` and `.tail()`.
4. Separate the dataset into input features `X` and target labels `y`.
5. Implement and run the Find-S algorithm to obtain the final specific hypothesis.
6. Implement and run the Candidate Elimination algorithm to obtain the final Specific Boundary (S) and General Boundary (G).
7. Compare the hypotheses produced by both algorithms.

## Files in this Folder
| File | Description |
|---|---|
| `B_B3_36_Lab3.ipynb` | Jupyter notebook containing the Find-S and Candidate Elimination implementation |
| `enjoy.csv` | EnjoySport dataset used for training |
| `README.md` | Documentation for this lab |

## Student Details
- **Name:** Pranoy Mridha
- **Roll No.:** 36
- **Section:** B
- **Batch:** B3
- **Course:** B.Tech Artificial Intelligence & Machine Learning
- **University:** Ramdeobaba University, Nagpur

## Result
- The **Find-S algorithm** converged to the final specific hypothesis: `['Sunny', 'Warm', '?', '?', '?', '?']`.
- The **Candidate Elimination algorithm** converged to the same final Specific Boundary: `['Sunny', 'Warm', '?', '?', '?', '?']`, along with a General Boundary consisting of hypotheses consistent with the negative examples.

## Conclusion
Both the Find-S and Candidate Elimination algorithms were successfully implemented on the EnjoySport dataset. Find-S produced a single most-specific hypothesis using only the positive examples, while Candidate Elimination used both positive and negative examples to maintain a version space bounded by the Specific and General boundaries. The final specific hypothesis from both algorithms matched, confirming consistent concept learning behaviour on this dataset.
