import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("sentiment.csv")
# data = pd.read_csv(r"sentiment.csv")  #use for windows

# data = pd.DataFrame({   # use if importing csv not working else remove this block
#     "message": [
#         "I really love this product",
#         "This is the worst experience ever",
#         "Amazing service and friendly staff",
#         "I am very disappointed with the quality",
#         "It was okay nothing special",
#         "Absolutely fantastic highly recommend",
#         "Terrible I will never buy again",
#         "Pretty good overall",
#         "Customer support was very helpful",
#         "The product broke after one use",
#         "Not bad could be better",
#         "Excellent quality and fast delivery",
#         "Very poor packaging and late delivery",
#         "I am satisfied with the purchase",
#         "Waste of money totally disappointed",
#         "Average experience nothing great"
#     ],
#     "sentiment": [
#         "positive","negative","positive","negative","neutral","positive","negative","positive",
#         "positive","negative","neutral","positive","negative","positive","negative","neutral"
#     ]
# })


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
