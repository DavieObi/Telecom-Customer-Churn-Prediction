## Telecom Customer Churn Prediction ☎️

This project aims to build a machine learning model to predict **customer churn** for a telecom company. By identifying customers likely to churn (stop using the service), the company can proactively implement retention strategies. The model uses a **Random Forest Classifier** to predict the binary outcome (`Yes` or `No` for churn).

-----

## 💾 Dataset Overview

The analysis uses the **Telecom Customer Churn dataset** containing various demographic, service, and billing information for over 7,000 customers.

### Key Columns

The dataset includes 21 features, such as:

  * **Target Variable:** `Churn` (Yes/No)
  * **Customer Information:** `Gender`, `SeniorCitizen`, `Partner`, `Dependents`, `customerID`
  * **Account Information:** `Tenure` (months customer stayed with the company), `Contract`, `PaperlessBilling`, `PaymentMethod`
  * **Service Information:** `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `MonthlyCharges`, `TotalCharges`, etc.

-----

## 💻 Methodology

### 1\. Data Loading and Initial Inspection

  * The data was loaded using pandas from a public GitHub repository.
  * Initial checks (`df.info()`, `df.describe()`, `df.isnull().sum()`) confirmed the data structure and revealed **no missing values** across the columns in the raw input count, though an implicit data type issue for `TotalCharges` (read as `object`) was handled during the Ordinal Encoding step.

### 2\. Handling Class Imbalance

The original dataset was highly imbalanced:

  * `Churn: No` - **5174**
  * `Churn: Yes` - **1869**

To address this, the **Random Over-Sampler** from the `imblearn` library was used to balance the classes in the training data, resulting in:

  * `Churn: No` - **5174**
  * `Churn: Yes` - **5174**

### 3\. Feature Engineering and Preprocessing

  * The unique identifier column (`customerID`) and the target variable (`Churn`) were dropped from the feature matrix $X$.
  * All remaining categorical and numerical features were converted to numerical representations using **Ordinal Encoding** (`sklearn.preprocessing.OrdinalEncoder`).

### 4\. Model Training

  * The preprocessed data was split into training and testing sets.
  * A **Random Forest Classifier** (`RandomForestClassifier`) was selected and trained on the oversampled and encoded training data.

-----

## 📈 Results and Evaluation

The model's performance was evaluated on the unseen test set using accuracy, a confusion matrix, and a classification report.

### Accuracy

The model achieved an overall accuracy of **$\approx 88.21\%$** on the test data.

### Confusion Matrix

| | Predicted No | Predicted Yes |
| :---: | :---: | :---: |
| **Actual No** | 1045 | 245 |
| **Actual Yes** | 60 | 1237 |

### Classification Report

| | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **No** | 0.95 | 0.81 | 0.87 | 1290 |
| **Yes** | 0.83 | 0.95 | 0.89 | 1297 |
| **Accuracy** | | | **0.88** | 2587 |

### Interpretation

  * **High Recall for Churn (`Yes`):** The model correctly identified $\mathbf{95\%}$ of the actual churning customers (1237 out of 1297). This is crucial for churn prediction, as the goal is to *catch* as many potential churners as possible.
  * **Good Precision for Non-Churn (`No`):** When the model predicted a customer would *not* churn, it was correct $95\%$ of the time.
  * The **F1-Score** is balanced for both classes ($\approx 0.87$ and $0.89$), indicating strong overall predictive performance across both the majority and minority classes.

-----

## 🛠️ Requirements

To run the notebook, you need the following Python libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
```

### Installation

You can install the necessary packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn
```

-----

## 🚀 Future Enhancements

  * **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize the Random Forest model for better performance.
  * **Alternative Models:** Test other classification algorithms like Gradient Boosting, Support Vector Machines, or Logistic Regression.
  * **Advanced Encoding:** Implement One-Hot Encoding for specific categorical features instead of applying Ordinal Encoding to all, as the latter might impose an arbitrary order.
  * **Feature Scaling:** Apply standard scaling or normalization to numerical features like `Tenure`, `MonthlyCharges`, and `TotalCharges`.
