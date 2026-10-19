"""Step 1: get to know the sentences before training anything.

Run with:  .venv/Scripts/python.exe 01_explore.py
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)
train = pd.read_csv("data/train.csv")

print(f"{len(train):,} training sentences\n")
print("A few examples:")
for _, row in train.sample(6, random_state=1).iterrows():
    print(f"  [{row.emotion:>8}] {row.text}")

# Imbalance: some emotions are much rarer. A model can look accurate by
# mostly guessing the common ones, so we'll also report a fair "macro" score.
counts = train.emotion.value_counts()
print("\nSentences per emotion:")
for emo, n in counts.items():
    print(f"  {emo:>8}: {n:5,}  ({n / len(train):5.1%})  {'#' * (n // 150)}")

lengths = train.text.str.split().str.len()
print(f"\nSentence length: median {lengths.median():.0f} words, longest {lengths.max()} words")
print("Note: real Reddit comments - capitals, punctuation, slang and emoji are kept.")

# The most telling words for each emotion: frequent in that emotion, rare elsewhere.
STOP = set("i im a an the and to of in it is was that my me for on with this be so but have as at feel feeling just like am you not about".split())
clean = train.text.str.lower().str.replace(r"[^a-z' ]", " ", regex=True)
words = {e: Counter(w for t in clean[train.emotion == e] for w in t.split() if w not in STOP) for e in counts.index}
total = Counter()
for c in words.values():
    total.update(c)
print("\nMost typical words per emotion:")
for e in counts.index:
    score = {w: n / total[w] * min(n, 60) for w, n in words[e].items() if total[w] >= 15}
    top = sorted(score, key=score.get, reverse=True)[:8]
    print(f"  {e:>8}: {', '.join(top)}")

# --- chart ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
counts.sort_values().plot.barh(ax=ax, color="#888")
ax.set_title("Sentences per emotion (training set)")
ax.set_xlabel("sentences")
fig.tight_layout()
fig.savefig(CHARTS / "emotions.png", dpi=120)
print("\nChart saved to charts/emotions.png")
