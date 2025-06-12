# CodeSoft Machine Learning Tasks

## About

This repository contains machine learning projects implemented in **Python** using the popular library **scikit-learn**.  
Scikit-learn is a powerful and easy-to-use Python library for data mining and data analysis, providing simple and efficient tools for predictive data modeling.

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```
or individually:
```bash
pip install pandas scikit-learn matplotlib
```

## What is Machine Learning?

Machine learning is a field of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed.  
It is widely used for tasks such as classification, regression, clustering, and more.

## What is scikit-learn?

[scikit-learn](https://scikit-learn.org/) is an open-source Python library that provides simple and efficient tools for data analysis and modeling.  
It supports various supervised and unsupervised learning algorithms, including classification, regression, clustering, and dimensionality reduction.

---

## Task 3: Iris KNN Classification

This task implements a K-Nearest Neighbors (KNN) classifier on the Iris dataset.  
The script performs cross-validation to select the best value of K, trains the final model, evaluates its performance, and plots accuracy vs K.

**How to run:**
```bash
python Task-3/Iris.py
```

**Output:**
- Prints best K, accuracy, classification report, and confusion matrix.
- Saves a plot: `accuracy_vs_k.png`

---

## Author

[Your Name]
