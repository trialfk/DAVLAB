import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("spam.csv")
# print(data.head(7)) # print first 7 rows of dataset

#  Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
#  Features and target
X = data['message']
y = data['label']
#  Convert text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
#  Stratified split (avoids warning)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
#  Model
model = MultinomialNB()
model.fit(X_train, y_train)
#  Predictions
y_pred = model.predict(X_test)
#  Metrics (safe output)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
