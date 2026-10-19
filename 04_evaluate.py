"""Step 4: the final test, then WHY the model decides.

Run with:  .venv/Scripts/python.exe 04_evaluate.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHARTS = Path("charts")
device = "cuda" if torch.cuda.is_available() else "cpu"
train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
EMOTIONS = sorted(train.emotion.unique())
K = len(EMOTIONS)

# --- 1. Every model on the 2,000 untouched test sentences ---------------------------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
base = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(vec.fit_transform(train.text), train.emotion)
results = {"TF-IDF baseline": base.predict(vec.transform(test.text))}

models = {}
for name in ["mini", "distilbert"]:
    tok = AutoTokenizer.from_pretrained(f"models/{name}")
    m = AutoModelForSequenceClassification.from_pretrained(f"models/{name}").to(device).eval()
    models[name] = (tok, m)


@torch.no_grad()
def probs(name, texts):
    tok, m = models[name]
    out = []
    for i in range(0, len(texts), 256):
        enc = tok(list(texts[i:i + 256]), truncation=True, max_length=64, padding=True, return_tensors="pt").to(device)
        out.append(m(**enc).logits.softmax(1).cpu())
    return torch.cat(out).numpy()


for name, label in [("mini", "BERT-mini (11M)"), ("distilbert", "DistilBERT (67M)")]:
    results[label] = np.array(EMOTIONS)[probs(name, test.text).argmax(1)]

print(f"{'model':<20} {'accuracy':>9} {'macro F1':>9}")
for label, pred in results.items():
    print(f"{label:<20} {accuracy_score(test.emotion, pred):>9.1%} {f1_score(test.emotion, pred, average='macro'):>9.1%}")

# --- 2. Per emotion, for the model that will go in the web app ----------------------
pred = results["BERT-mini (11M)"]
print("\nBERT-mini, F1 per emotion (hardest first):")
per = f1_score(test.emotion, pred, average=None, labels=EMOTIONS)
for e, s in sorted(zip(EMOTIONS, per), key=lambda x: x[1]):
    print(f"  {e:>8}: {s:.1%}")

cm = confusion_matrix(test.emotion, pred, labels=EMOTIONS, normalize="true")
pairs = sorted(((cm[i, j], EMOTIONS[i], EMOTIONS[j]) for i in range(K) for j in range(K) if i != j), reverse=True)
print("\nMost common mix-ups:")
for share, true, guess in pairs[:5]:
    print(f"  {true:>8} read as {guess:<8} {share:.1%}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K), EMOTIONS, rotation=40, ha="right")
ax.set_yticks(range(K), EMOTIONS)
ax.set_xlabel("model said")
ax.set_ylabel("true emotion")
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black", fontsize=9)
ax.set_title("BERT-mini confusion matrix (% of each emotion)")
fig.tight_layout()
fig.savefig(CHARTS / "confusion.png", dpi=120)

# --- 3. Explainability: which words drive the decision? ------------------------------
# "Leave one word out": remove each word in turn and see how much the winning
# emotion's log-probability drops. A big drop = that word mattered. (Log-probability,
# not probability: when the model is ~100% sure, probabilities barely move.)
# Simple, honest, works with any model - the web app does exactly the same thing.
def explain(text, name="mini"):
    words = text.split()
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    p = probs(name, variants)
    top = p[0].argmax()
    impact = np.log(p[0, top]) - np.log(p[1:, top])
    return EMOTIONS[top], p[0, top], list(zip(words, impact))


def show(text):
    emo, conf, impact = explain(text)
    key = sorted(impact, key=lambda x: -x[1])[:2]
    marked = " ".join(f"[{w}]" if (w, s) in key and s > .1 else w for w, s in impact)
    print(f"  {emo:>8} {conf:4.0%} | {marked}")


print("\nWhy? Words in [brackets] pushed hardest towards the answer:")
for t in test.text.sample(4, random_state=3):
    show(t)

# --- 4. Tricky sentences the model never saw -----------------------------------------
print("\nTricky sentences (written by hand, not from the dataset):")
for t in ["I feel so happy today!",
          "I do not feel happy at all.",                      # negation
          "Oh great, another Monday. I just love waking up at six.",  # sarcasm
          "I feel nervous but excited about my new job",      # mixed feelings
          "They cancelled my flight again and nobody even apologised.",
          "I can't believe she remembered my birthday!",
          "The package arrived on Tuesday."]:                 # nothing emotional
    show(t)

# --- 5. Errors: confident and wrong ----------------------------------------------------
p = probs("mini", test.text)
wrong = np.where(np.array(EMOTIONS)[p.argmax(1)] != test.emotion.values)[0]
worst = wrong[np.argsort(-p[wrong].max(1))][:6]
print("\nMost confident mistakes:")
for i in worst:
    print(f"  labelled {test.emotion[i]:>8}, model said {EMOTIONS[p[i].argmax()]:>8} {p[i].max():.0%} | {test.text[i][:90]}")

# --- comparison chart -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.4))
labels = list(results)
f1s = [f1_score(test.emotion, results[l], average="macro") for l in labels]
ax.barh(labels, f1s, color=["#bbb", "#4a7bd0", "#2b4c8c"])
for y, v in enumerate(f1s):
    ax.text(v + .003, y, f"{v:.1%}", va="center")
ax.set_xlim(.7, .97)
ax.set_xlabel("macro F1 on 2,000 test sentences")
ax.set_title("Transformers vs word counting")
fig.tight_layout()
fig.savefig(CHARTS / "comparison.png", dpi=120)
print("\nCharts saved: charts/confusion.png, charts/comparison.png")
