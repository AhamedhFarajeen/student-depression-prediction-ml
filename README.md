# Student Depression Prediction — ML Classification

A machine learning project that predicts student depression risk from survey data, developed as a group project for an AI/ML module (Y2S1). The pipeline addresses class imbalance, trains and compares five classification models, and includes an interactive prediction interface.

> **Disclaimer:** This is an academic project built on a public dataset for learning purposes. It is **not** a clinical or diagnostic tool and should not be used to make real mental health assessments.

## Overview

- **Dataset:** [Student Depression Dataset](https://www.kaggle.com/datasets) (Kaggle) — survey-based features (age, CGPA, study hours, lifestyle factors, etc.) with a binary depression label
- **Problem type:** Binary classification
- **Class imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, SVM, MLP (Neural Network)
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, k-fold cross-validation

## My Contribution

This was a 5-person team project. My individual contribution was the **MLP (Multi-Layer Perceptron) neural network model** — architecture design, feature scaling, training, and evaluation (`Models/IT24102168_MLP.ipynb`).

| Component | Contributor |
|---|---|
| MLP (Neural Network) | IT24102168 (me) |
| Logistic Regression | IT24102091 |
| Decision Tree | IT24102172 |
| SVM | IT24102082 |
| Random Forest | IT24102042 |

## Project Structure

```
student-depression-prediction-ml/
│
├── Models/
│   ├── Balanced_Dataset.ipynb          # SMOTE class-balancing
│   ├── IT24102168_MLP.ipynb            # MLP model (my contribution)
│   ├── IT24102091_Logistic_Regression.ipynb
│   ├── IT24102172_Decision_Tree.ipynb
│   ├── IT24102082_SVM.ipynb
│   ├── IT24102042_Random_Forest.ipynb
│   └── Comparison.ipynb                # Final model comparison
│
├── Data/
│   ├── student_depression_dataset.csv
│   └── student_depression_balanced.csv
│
├── UI.ipynb                            # Interactive prediction widget
└── README.md
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- ipywidgets (prediction UI)

## Running the Notebooks

These notebooks were built in Google Colab and mount Google Drive for data access. To run locally:

1. Clone the repo
2. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn ipywidgets`
3. Update the data paths in each notebook (they currently point to a Google Drive path)
4. Run `Models/Balanced_Dataset.ipynb` first to generate the balanced dataset
5. Run individual model notebooks, then `Models/Comparison.ipynb` for the final comparison

## Results

See `Models/Comparison.ipynb` for the full metric breakdown and ROC curve comparison across all five models.

## License

This project was developed for educational purposes as part of an academic module.
