"""Step 2: a simple baseline - count words, then a linear model.

TF-IDF turns each sentence into word scores (rare, telling words count more),
then logistic regression learns how much each word pushes towards each emotion.
It knows words, but not their order or context.

Run with:  .venv/Scripts/python.exe 02_baseline.py
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Always guess the most common emotion ('{train.emotion.mode()[0]}'): "
      f"{(test.emotion == train.emotion.mode()[0]).mean():.1%} accuracy\n")

# ngram_range=(1, 2): use single words AND pairs of words ("not happy" != "happy")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train, X_test = vec.fit_transform(train.text), vec.transform(test.text)

model = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
model.fit(X_train, train.emotion)
pred = model.predict(X_test)

print("TF-IDF + logistic regression")
print(f"  accuracy: {accuracy_score(test.emotion, pred):.1%}")
print(f"  macro F1: {f1_score(test.emotion, pred, average='macro'):.1%}   (treats every emotion equally, even rare ones)")

# What did it learn? The words that push hardest towards each emotion.
print("\nStrongest words per emotion:")
terms = vec.get_feature_names_out()
for i, emo in enumerate(model.classes_):
    top = model.coef_[i].argsort()[-7:][::-1]
    print(f"  {emo:>8}: {', '.join(terms[j] for j in top)}")
