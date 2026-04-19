import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("sentiment.csv")
# print(head(data)) # ro print whole data 
# print(data.head(5))  #to print dataset ( top 5 rows )

#  Convert labels
data['sentiment'] = data['sentiment'].map({
    'negative': 0,
    'neutral': 1,
    'positive': 2
})
#  Features & target
X = data['message']
y = data['sentiment']
#  Text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
#  Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
#  Model
model = MultinomialNB()
model.fit(X_train, y_train)
#  Prediction
y_pred = model.predict(X_test)
#  Metrics
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
