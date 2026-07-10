# 🎬 IMDb Sentiment Analysis using NLP and Machine Learning

## 📌 Overview

This project performs **Sentiment Analysis** on IMDb movie reviews using **Natural Language Processing (NLP)** and **Machine Learning**. The model predicts whether a movie review is **Positive** or **Negative** based on the review text.

---

## 📂 Dataset

- **Dataset:** IMDb Movie Reviews Dataset
- **Total Reviews:** 50,000
- **Classes:** Positive and Negative (Balanced Dataset)

> **Note:** The dataset is not included in this repository due to its large file size. You can download it from Kaggle.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Label Encoding
- Text Preprocessing
- Lowercasing
- HTML Tag Removal
- Punctuation Removal
- Number Removal
- Emoji Removal
- Stopword Removal
- Bag of Words (BoW)
- TF-IDF Vectorization
- Naive Bayes Classification
- Logistic Regression Classification
- Model Comparison
- User Review Prediction

---

## 🛠 Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn

---

## 📖 Project Workflow

1. Load the Dataset
2. Perform Exploratory Data Analysis (EDA)
3. Encode Sentiment Labels
4. Clean and Preprocess Review Text
5. Split Dataset into Training and Testing Sets
6. Convert Text into Numerical Features using Bag of Words and TF-IDF
7. Train Multiple Machine Learning Models
8. Compare Model Performance
9. Select the Best Model
10. Predict Sentiment for User Input

---

## 🤖 Machine Learning Models

- Multinomial Naive Bayes (Bag of Words)
- Multinomial Naive Bayes (TF-IDF)
- Logistic Regression (Bag of Words)
- Logistic Regression (TF-IDF)

---

## 📊 Best Model

**Logistic Regression with TF-IDF**

**Accuracy:** **89.53%**

---

## 📈 Evaluation Metrics

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## ▶️ How to Run the Project

1. Install the required libraries.

```bash
pip install -r requirements.txt
```

2. Run the Python file.

```bash
python sentiment_analysis.py
```

3. Enter a movie review when prompted.

---

## 🔮 Future Improvements

- Lemmatization
- Hyperparameter Tuning
- Deep Learning (LSTM)
- Transformer Models (BERT)

---

## 👨‍💻 Author

** Nikhil Kumar**

B.Tech Student | Aspiring AI/ML Engineer

GitHub: https://github.com/nimmalapudinikhilkumar07-collab
