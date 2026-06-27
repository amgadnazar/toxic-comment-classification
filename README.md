# 🛡️ Toxic Comment Classification using Machine Learning & Deep Learning

A research project comparing **Logistic Regression**, **LSTM**, and **DistilBERT** for toxic comment detection using an augmented version of the Jigsaw Toxic Comment dataset containing more than **178,000 comments**.

The project evaluates traditional machine learning and transformer-based deep learning approaches, applies class imbalance techniques, and optimizes classification thresholds to maximize predictive performance.

---

## 📌 Project Overview

Online platforms generate millions of user comments every day, making automatic toxic content detection essential for maintaining healthy communities.

This project investigates three different modeling approaches:

- Logistic Regression (Baseline)
- Long Short-Term Memory (LSTM)
- DistilBERT Transformer

Each model is trained and evaluated using the same dataset to compare their effectiveness in identifying toxic comments.

---

## 🎯 Objectives

- Detect toxic comments automatically
- Compare classical ML with deep learning approaches
- Handle class imbalance
- Optimize classification thresholds
- Evaluate model performance using multiple metrics

---

## 📂 Dataset

The project uses an **augmented Jigsaw Toxic Comment dataset** containing over **178,000 labeled comments**.

The dataset includes binary labels indicating whether a comment is toxic or non-toxic.

---

## 🧠 Models Implemented

### 1️⃣ Logistic Regression
- TF-IDF Vectorization
- Baseline machine learning model
- Fast training and inference

### 2️⃣ LSTM
- Tokenization
- Embedding Layer
- Long Short-Term Memory Network
- Binary Classification

### 3️⃣ DistilBERT
- Hugging Face Transformers
- Fine-tuned for toxic comment classification
- State-of-the-art contextual language representation

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Matplotlib
- Seaborn

---

## 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall Curve
- Confusion Matrix

Threshold optimization was also performed to maximize the F1-score.

---

## 🏆 Results

| Model | ROC-AUC | F1 Score |
|--------|---------|----------|
| Logistic Regression | Lower | Lower |
| LSTM | Competitive | Moderate |
| **DistilBERT** | **0.9727** | **0.73** |

### Best Performing Model

✅ **DistilBERT**

It achieved:

- ROC-AUC: **0.9727**
- F1-score: **0.73**

making it the strongest model among all evaluated approaches.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── .gitkeep
│
├── models/
│   ├── LogisticRegression.py
│   ├── LSTM.py
│   └── DistilBERT.py
│
├── paper/
│   ├── Toxic_Comment_Classification_Report.pdf
│
├── final_combined_dataset.zip
├── test_labels.csv
├── README.md
```

---

## 📄 Research Paper

The complete research paper describing the methodology, experiments, and results is available here:

📑 [Toxic Comment Classification Research Paper](paper/Toxic_Comment_Classification_Report.pdf)

---

## 🚀 Future Improvements

- Multi-label toxic classification
- Explainable AI using SHAP or LIME
- Hyperparameter optimization
- Deploy the model as a REST API
- Build a Streamlit or Flask web application
- Real-time moderation system

---

## 👨‍💻 Author

**Amjad Salih**

**Abubakre Osama**

**Yahya Mohamed**

Google Certified Data Analyst

📍 Cairo, Egypt

- 💼 LinkedIn: https://www.linkedin.com/in/amjad-nazar
- 🌐 Portfolio: https://amgadnazar.github.io/
- 📧 Email: amgadnazar11@gmail.com

---

⭐ If you found this project useful, consider giving it a star!
