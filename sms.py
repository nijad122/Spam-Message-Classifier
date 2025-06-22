import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download SMS Spam Collection Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)

# Convert text to features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predict
predictions = model.predict(X_test_counts)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Test on custom message
custom_msg = ["You have won a free prize worth $1000! Click here."]
custom_vec = vectorizer.transform(custom_msg)
print("Spam Prediction:", model.predict(custom_vec))  # 1 = Spam, 0 = Not spam
