import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.model_selection import GridSearchCV

base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "../data", "IMDB_Dataset.csv")
df = pd.read_csv(dataset_dir)

df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=2)), ('model', LogisticRegression(max_iter=1000))])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))

tfidf = pipeline.named_steps['tfidf']
print(len(tfidf.get_feature_names_out()))

feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
coefs = pipeline.named_steps['model'].coef_[0]

top_pos = pd.DataFrame({'word': feature_names, 'coef': coefs}).sort_values('coef', ascending=False)
print(top_pos.head(10))
top_neg = pd.DataFrame({'word': feature_names, 'coef': coefs}).sort_values('coef')
print(top_neg.head(10))

def predict_review(text):
    proba = pipeline.predict_proba([text])[0][1]
    sentiment = ('positive' if proba >= 0.5 else 'negative')
    return sentiment, proba

print(predict_review("awfully good")) # ('positive', 0.9434169568959465)

joblib.dump(pipeline, 'imdb_sentiment.pkl')
model = joblib.load('imdb_sentiment.pkl')

param_grid = {'tfidf__max_features': [5000, 10000, 20000], 'tfidf__min_df': [2, 5, 10], 'model__C': [0.1, 1, 5]}
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

print(classification_report(y_test, pred))