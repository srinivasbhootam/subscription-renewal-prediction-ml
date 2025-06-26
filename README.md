# Subscription Renewal Prediction

Predicting which users will renew their subscription using machine learning.

##  Overview

This project analyzes a subscription service dataset of 1,000 users to predict renewal behavior. We implement three classifiers (Decision Tree, K-Nearest Neighbors, Random Forest), evaluate them on accuracy, precision, recall, and F1-score, and apply PCA to study the impact of dimensionality reduction on model performance.

##  Dataset

- **Source:** Kaggle – Subscription Renewal Prediction  
- **Records:** 1,000 users  
- **Features:**  
  - `usage_days` (0–29 days/month)  
  - `last_login` (days since last login)  
  - `monthly_fee` (\$10 / \$15 / \$20 tiers)  
- **Target:**  
  - `renewed` (0 = did not renew, 1 = renewed)

##  Preprocessing

1. Checked for and found no missing values.  
2. Dropped the `user_id` column (no predictive power).  
3. Standardized all numeric features (zero mean, unit variance).

##  Models & Evaluation

| Model             | Test Acc | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Decision Tree     | 0.535    | 0.61      | 0.63   | 0.62     |
| K-Nearest Neighbors | 0.575  | 0.62      | 0.74   | 0.68     |
| Random Forest     | 0.515    | 0.58      | 0.67   | 0.62     |

- **Best before PCA:** K-NN (F1 = 0.68)  
- Evaluated via 5-fold cross-validation and confusion matrices.

##  Dimensionality Reduction

- **Technique:** PCA (95% variance)  
- **Components retained:** 3 → 3 (no reduction at 95% threshold)  
- **Post-PCA performance:**

| Model             | Test Acc | F1-Score |
|-------------------|----------|----------|
| Decision Tree     | 0.505    | 0.59     |
| K-Nearest Neighbors | 0.575  | 0.68     |
| Random Forest     | 0.595    | 0.70     |

- **Insight:** Random Forest gains +8 pp accuracy and +0.07 F1; PCA smooths noise for ensemble methods.

##  Key Findings

- **Renewal drivers:** Higher `usage_days` and recent `last_login` strongly correlate with renewal.  
- **Price tier:** `monthly_fee` alone is not a strong predictor.  
- **Model choice:**  
  - **K-NN** for high recall on renewals  
  - **Random Forest + PCA** for best overall uplift  
  - **Decision Tree** when interpretability and balance are priorities  

##  Next Steps

- Hyperparameter tuning with `GridSearchCV`.  
- Address class imbalance (SMOTE or class weights).  
- Engineer new features (e.g., interaction of usage and recency).


---

**Kaggle Notebook:**  
https://www.kaggle.com/your-username/subscription-renewal-prediction

