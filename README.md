# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project is a **Machine Learning classification system** for predicting whether a patient has heart disease based on clinical and medical features.

The project uses several Machine Learning classification algorithms and compares their performance using different evaluation metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC
* ROC Curve

A **Voting Classifier** is also implemented to combine the predictions of multiple models.

Finally, the trained Voting Classifier is deployed using a **Streamlit web application**.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Preprocess the heart disease dataset.
2. Handle missing values.
3. Convert categorical features into numerical features.
4. Select the most important features using `SelectKBest`.
5. Scale the selected features using `StandardScaler`.
6. Train multiple Machine Learning models.
7. Compare the models using different evaluation metrics.
8. Plot ROC curves and calculate ROC-AUC.
9. Build an interactive prediction interface using Streamlit.

---

## 📊 Dataset

The project uses the **Heart Disease UCI dataset**.

The dataset contains medical information about patients, including features such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression
* And other clinical features

The original target variable is `num`.

It is converted into a binary classification problem:

```python
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
```

Where:

* `0` → No Heart Disease
* `1` → Heart Disease

---

## 🧹 Data Preprocessing

Several preprocessing steps are performed.

### 1. Handling Missing Values

Missing values in selected columns are replaced using the mode:

```python
df[['fbs','exang']] = df[['fbs','exang']].fillna(
    df[['fbs','exang']].mode().iloc[0]
).astype(int)
```

Other missing numerical values are handled using `SimpleImputer` with the mean strategy:

```python
imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

### 2. Encoding Categorical Features

Categorical variables are converted into numerical variables using One-Hot Encoding:

```python
df = pd.get_dummies(
    df,
    columns=['cp','dataset','restecg','thal','slope','sex'],
    drop_first=True,
    dtype=int
)
```

### 3. Feature Selection

The project uses `SelectKBest` with ANOVA F-test:

```python
kBest = SelectKBest(
    score_func=f_classif,
    k=k
)
```

The top 10 features are selected.

### 4. Feature Scaling

The selected features are standardized using:

```python
scaler = StandardScaler()
```

---

## 🤖 Machine Learning Models

The following models are trained:

### 1. Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

Logistic Regression is used as a baseline classification model.

---

### 2. Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Random Forest combines multiple decision trees to improve prediction performance.

---

### 3. Decision Tree

```python
DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)
```

The Decision Tree uses entropy as the splitting criterion.

---

### 4. Voting Classifier

The final model combines:

* Logistic Regression
* Random Forest
* Decision Tree

using a Soft Voting Classifier:

```python
VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('RanForst', RanForst),
        ('tree', tree)
    ],
    voting='soft'
)
```

The Soft Voting Classifier uses the predicted probabilities from the individual models to make the final prediction.

---

## 📈 Model Evaluation

Each model is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many of the patients predicted as having heart disease actually have heart disease.

### Recall

Measures how many of the actual heart disease cases were correctly detected.

### F1-Score

The harmonic mean of Precision and Recall.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

The ROC curve is generated using:

```python
fpr, tpr, _ = roc_curve(
    y_test,
    y_probability
)
```

and the AUC is calculated using:

```python
roc_auc_score(
    y_test,
    y_probability
)
```

---

## 📉 ROC Curves

The project compares the ROC curves of all four models:

* Logistic Regression
* Random Forest
* Decision Tree
* Voting Classifier

The ROC curve plots:

* **False Positive Rate (FPR)** on the X-axis
* **True Positive Rate (TPR)** on the Y-axis

The AUC value is displayed for each model.

---

## 💾 Model Saving

The final Voting Classifier is saved using Joblib:

```python
joblib.dump(vot, 'voting_model.pkl')
```

The saved model is then loaded inside the Streamlit application:

```python
model = joblib.load('voting_model.pkl')
```

---

## 🌐 Streamlit Application

The project includes an interactive web application built with Streamlit.

The user can enter patient information such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression

After clicking **Predict**, the application displays the prediction:

### ❤️ Heart Disease Detected

or

### 💚 No Heart Disease Detected

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Pyngrok

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart_disease_uci.csv
├── heart_disease_prediction.ipynb
├── app.py
├── voting_model.pkl
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
```

Move into the project directory:

```bash
cd Heart-Disease-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open locally in your browser.

Usually it will be available at:

```text
http://localhost:8501
```

---

## 📋 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
pyngrok
```

---

## ⚠️ Important Note

This project is intended for **educational and demonstration purposes only**.

The prediction generated by the model should **not be considered a medical diagnosis** or a replacement for professional medical advice.

---

## 👨‍💻 Author

**Mohamed Srour**

Machine Learning / AI Project

---

## ⭐ Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* More Machine Learning algorithms
* Better handling of missing data
* Feature importance visualization
* Interactive ROC-AUC comparison
* Probability-based prediction display
* Improved Streamlit UI
* Deployment using Streamlit Cloud or another cloud platform
