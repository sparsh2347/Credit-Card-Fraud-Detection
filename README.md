# 💳 Credit Card Fraud Detection App

This is a Streamlit-based web application that detects whether a credit card transaction is **fraudulent** or **legitimate** using a machine learning model trained on transaction data.

> Built by Sparsh Sinha  
> IIIT Lucknow | B.Tech in Computer Science and Business

---

## 🔍 Project Overview

Fraudulent transactions are a critical issue in the financial industry. This project applies a **Random Forest Classifier** to identify fraud based on attributes like transaction category, gender, city, state, amount, and location (latitude & longitude).

The trained model and all encoders are integrated into a Streamlit frontend, allowing users to interactively test predictions with custom inputs.

---

## 🧠 How It Works

- **Input**: Transaction details (category, gender, location, amount, etc.)
- **Preprocessing**: Encoders convert categorical features to numerical format
- **Model**: A trained `RandomForestClassifier` predicts fraud or legit
- **Output**: ✅ Legit transaction or 🚨 Fraudulent transaction

## ⚙️ Workflow Summary

1. **Data Collection**: Using Kaggle API
2. **Preprocessing**:
   - Dropping irrelevant features (e.g., `name`, `cc_num`, `zip`)
   - Encoding categorical features (e.g., `gender`, `category`, `state`, `city`)
3. **Exploratory Data Analysis**:
   - Class distribution plots
   - Correlation heatmaps
4. **Train-Test Split** (e.g., 80/20)
5. **Balancing the Data**:
   - SMOTE for oversampling minority class
   - Random undersampling for majority class
6. **Model Training & Hyperparameter Tuning**:
   - GridSearchCV for finding optimal parameters
   - Cross-validation for reliable metrics
7. **Evaluation**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
8. **Visualization**:
   - Fraud distribution
   - Correlation heatmaps

---

## 🛠️ Technologies Used

- Python 3.8+
- Streamlit
- NumPy
- scikit-learn
- Pickle (for model + encoder serialization)

---

## 📁 Project Structure

```plaintext
fraud-detection-app/
│
├── app.py                   # Main Streamlit app
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
│
├── encoders/                # Stored label encoders
│   ├── gender_encoder.pkl
│   ├── category_encoder.pkl
│   ├── city_encoder.pkl
│   └── state_encoder.pkl
│
├── model/                 # Trained model file
│   └── trained_model_rf.sav
```

---

## 📂 Dataset

- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

---
---

## 📊 Model Performance

The trained `RandomForestClassifier` achieves:

- ✅ **Accuracy**: 97%
- 🚨 **Precision**: 68%

This performance strikes a strong balance:
- ⚖️ Maintains **high overall accuracy**
- 🔍 Reduces **false positives**, which is crucial for minimizing disruptions to legitimate users while still catching fraud effectively.

---


## ⚙️ Setup Instructions

### 🔹 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app
```

### 🔹 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Download the Trained Model

You can download the trained model from the following drive link:

👉 [Download `trained_model_rf.sav`](https://drive.google.com/file/d/1rv1nZt1ecL-n1XWp3_2gRtIyCLkQROIv/view?usp=drive_link)

Place the model inside the fraud-detection-app directory.

### 🔹 4. Run the App

```bash
streamlit run app.py
```
