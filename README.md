# 🛍️ Amazon Product Review Sentiment Analysis

Analyzing customer feedback is crucial for business intelligence and customer satisfaction. This project applies **Natural Language Processing (NLP)** techniques to Amazon product reviews to classify sentiment as either **positive** or **negative**.

## 📌 Overview

This repository presents a complete workflow for sentiment classification of Amazon product reviews. The goal is to detect emotional tone from textual reviews and assign appropriate sentiment labels using robust preprocessing and machine learning models.

> **Author:** Harel Amezcua  
> **Status:** 🎯 In Progress  
> **License:** [MIT License](./LICENSE)

---

## 🗂️ Project Structure

```

├── datasets/                         # Raw and cleaned datasets
├── notebooks/                        # Jupyter Notebooks for EDA and modeling
├── Solución\_Departamento\_de\_Relaciones\_Públicas.ipynb  # Main analysis notebook
├── .gitignore
├── .gitattributes
├── LICENSE
└── README.md                         # You're here!

````

---

## ⚙️ Technologies Used

- **Python 3.x**
- **Pandas** & **NumPy** – Data handling
- **Matplotlib** & **Seaborn** – Data visualization
- **Scikit-learn** – Model building and evaluation
- **NLTK / spaCy** – Text preprocessing

---

## 🔍 Key Features

- Text normalization: lowercasing, tokenization, stopword removal
- Exploratory Data Analysis (EDA) on review distributions
- Feature engineering with bag-of-words and TF-IDF
- Classification using Logistic Regression
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/HarelAmezcua/amazon-sentiment-analysis.git
   cd amazon-sentiment-analysis
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Open and run the notebook**

   ```bash
   jupyter notebook notebooks/Solución_Departamento_de_Relaciones_Públicas.ipynb
   ```

---

## 💡 Sample Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_review'])
y = data['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📈 Results

The model achieved strong performance on the validation set, demonstrating its capability to accurately classify user sentiment. Further improvements can be made with deep learning architectures like LSTM or transformers.

---

## 📬 Contact

For suggestions or collaborations:

**Harel Amezcua**
[LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com/HarelAmezcua)

---

## 📄 License

This project is licensed under the terms of the [MIT license](./LICENSE).
