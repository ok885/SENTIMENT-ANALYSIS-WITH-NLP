
# SENTIMENT-ANALYSIS-NLP


**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: BHARAT BHANDARI  
**INTERN ID**: CT04DF123  
**DOMAIN**: MACHINE LEARNING  
**DURATION**: 4 WEEKS  
**MENTOR**: NEELA SANTOSH

This repository contains a complete sentiment analysis implementation using **TF-IDF vectorization** and **Logistic Regression**, developed in **Python** via **Google Colab (Jupyter Notebook)**. The project focuses on classifying textual product reviews into different sentiment categories using standard NLP and ML techniques.

---

## 🧠 Project Overview

Sentiment Analysis is a Natural Language Processing task that determines the emotional tone of a piece of text. In this project, user-generated product reviews are classified into sentiment categories ranging from negative to positive.

This implementation uses:
- **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.
- **Logistic Regression** as the classification model.
- **Custom NLP preprocessing** using NLTK.

---

## 📂 Dataset Description

The dataset comes from a tab-separated file structure, commonly based on the **Rotten Tomatoes** sentiment analysis dataset.

- `train.tsv`: Contains text phrases with labeled sentiment classes (0 to 4).
- `test.tsv`: Contains phrases with no labels (for prediction).

**Sentiment Classes:**
- 0 → Negative  
- 1 → Somewhat Negative  
- 2 → Neutral  
- 3 → Somewhat Positive  
- 4 → Positive

---

## 🧹 Text Preprocessing

The project includes custom text cleaning steps using `nltk`, including:

- Lowercasing text
- Tokenization (`word_tokenize`)
- Removing punctuation and digits
- Removing stopwords
- Stemming via `SnowballStemmer`

These steps are encapsulated in a custom function and passed to `TfidfVectorizer`.

---

## 🔍 Feature Extraction: TF-IDF

The `TfidfVectorizer` converts the cleaned phrases into numeric feature vectors using:

- `tokenizer=custom_tokenize_function`
- `ngram_range=(1, 2)`
- `max_features=2300` to reduce dimensionality

This allows the model to learn from both individual words and adjacent word pairs (bigrams).

---

## 🧠 Model: Logistic Regression

We used `LogisticRegression` from scikit-learn with:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

The model is trained on the TF-IDF features and evaluated using a validation set or test predictions.

---

## 📊 Evaluation

Evaluation metrics used include:

- **Accuracy**
- **Confusion Matrix**
- *(Optional)* **Classification Report** (Precision, Recall, F1-score)

Visualizations are done using `matplotlib`.

---

## 🛠 Dependencies

Required packages:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`

Install them using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Upload `train.tsv` and `test.tsv` to your Colab workspace.
2. Run the Jupyter notebook step-by-step.
3. View metrics and prediction results.

---

## ✅ Conclusion

This project demonstrates how classic machine learning and basic NLP can be combined for effective sentiment analysis. It serves as a practical example of using TF-IDF and Logistic Regression on real-world text data with custom preprocessing.

---

## 🙋 Author

**BHARAT BHANDARI**  
**Intern ID**: CT04DF123

---
