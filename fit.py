from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Fit the vectorizer on your training data (X_train)
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

# 2. Fit the classifier on the transformed data and labels (y_train)
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# 3. Save the *fitted* objects
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl') # Must be the fitted vectorizer!