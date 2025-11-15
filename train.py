# train_and_save_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# 2️⃣ Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4️⃣ Create Bag of Words model
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 5️⃣ Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# 6️⃣ Evaluate model
y_pred = model.predict(X_test_cv)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save trained model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n🎯 Model and Vectorizer saved successfully!")
