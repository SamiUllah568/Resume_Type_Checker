# Resume Checker

## Contact

🔗 **GitHub:** [SamiUllah568](https://github.com/SamiUllah568)  
📧 **Email:** [sk2579784@gmail.com](mailto:sk2579784@gmail.com)


## Dataset Source

The dataset used in this project can be accessed from the Kaggle:

[Kaggle](<https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset>)


## Aim

The aim of the project is to develop a machine learning-based resume screening system that can automatically analyze and classify resumes based on their relevance to job postings.

## Installation
```bash
pip install pandas numpy scikit-learn spacy nltk
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

## Key Features
- **Text Preprocessing Pipeline**
  - Encoding fixes
  - URL/hashtag removal
  - Spacy lemmatization
  - Porter stemming
- **Multi-class Classification**
  - 25+ job categories supported
- **Model Performance**
  - Logistic Regression: 99.48% test accuracy
  - Decision Trees: 98.45% test accuracy
  - KNN: 92.23% test accuracy


## Import Necessary Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')
```

## Load Dataset

```python
df = pd.read_csv("/content/UpdatedResumeDataSet.csv")
df.shape
```

## Show Top 5 Rows

```python
df.head(5)
```

## Check Missing Values and Duplicated Values

```python
print("Missing Values --> ", df.isnull().sum())
print("Duplicated Values --> ", df.duplicated().sum())
```

## Category Distribution

```python
df["Category"].value_counts()
sns.countplot(y=df["Category"])
plt.xticks(rotation=45)
plt.title("Distribution Category Counts", fontweight='bold')
plt.show()

plt.figure(figsize=(8,8))
plt.pie(df["Category"].value_counts(), labels=df["Category"].value_counts().index, autopct="%0.02f%%", shadow=True)
plt.title("Pie Chart for Categories", fontweight='bold')
plt.show()
```

## Text Preprocessing for Resume Data

```python
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
nltk.download('stopwords')

def text_cleaning(text):
    text = text.replace("Naïve", "Naive")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stopwords.words("english")]
    tokens = [stemmer.stem(token) for token in tokens]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Resume"] = df["Resume"].apply(text_cleaning)
df = df.sample(962, random_state=42)
```

## Data Splitting for Model Training and Testing

```python
x_train, x_test, y_train, y_test = train_test_split(df["Resume"], df["Category"], test_size=0.2, random_state=42)
```

## Label Encoding for Target Variables

```python
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
y_train = encode.fit_transform(y_train)
y_test = encode.transform(y_test)
```

## Text Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(ngram_range=(1, 1), max_features=5000)
x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)
```

## Model Training and Evaluation

```python
def model_training_evaluation(model):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_score = accuracy_score(y_train, train_pred)
    test_score = accuracy_score(y_test, test_pred)
    train_rep = classification_report(y_train, train_pred)
    test_rep = classification_report(y_test, test_pred)
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    print("\n### Model Performance Evaluation ###\n")
    print("#### Train Data Performance ####")
    print(f"Train Accuracy Score: {train_score:.4f}")
    print(f"Train Classification Report:\n{train_rep}")
    print(f"Train Confusion Matrix:\n{train_cm}")
    print("\n#### Test Data Performance ####")
    print(f"Test Accuracy Score: {test_score:.4f}")
    print(f"Test Classification Report:\n{test_rep}")
    print(f"Test Confusion Matrix:\n{test_cm}")

from sklearn.linear_model import LogisticRegression
log_reg = OneVsRestClassifier(LogisticRegression(random_state=42))
model_training_evaluation(log_reg)
```

# Results
## Model Comparison

| Model                 | Train Accuracy | Test Accuracy | Precision | Recall |
|-----------------------|---------------|--------------|-----------|--------|
| Logistic Regression  | 100%          | 99.48%       | 0.99      | 0.99   |
| Decision Tree        | 100%          | 98.45%       | 0.98      | 0.98   |
| KNN                 | 97.79%        | 92.23%       | 0.93      | 0.91   |

Logisticc Regression perform well


## Save Model

```python
import pickle
pickle.dump(vector, open("ngram.pkl", 'wb'))
pickle.dump(log_reg, open("model.pkl", 'wb'))
pickle.dump(encode, open("encoder.pkl", 'wb'))
```

## Function to Predict Text Category

```python
def pred(text):
    clean_text = text_cleaning(text)
    vectorize_text = vector.transform([clean_text])
    vectorize_text = vectorize_text.toarray()
    predicted = log_reg.predict(vectorize_text)
    predicted_value = encode.inverse_transform(predicted)
    return predicted_value[0]
```

## Test Model with Sample Resumes

```python
myresume = """I am a data scientist specializing in machine learning, deep learning, and computer vision..."""
print(pred(myresume))  # Output: 'Data Science'

myresume = """Jane Smith is a certified personal trainer with over 5 years of experience..."""
print(pred(myresume))  # Output: 'Health and fitness'

myresume = """John Doe is an experienced Network Security Engineer with over 7 years of expertise..."""
print(pred(myresume))  # Output: 'Network Security Engineer'
```
