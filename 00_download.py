"""Step 0: download GoEmotions and turn it into 7 simple emotions.

GoEmotions (Google, CC BY 4.0): 58,000 real Reddit comments labelled by humans
with 27 fine emotions + neutral. Google's own "Ekman" grouping folds those into
6 basic emotions + neutral, which is what we use.

Comments can carry several labels. We keep the ones whose labels all fall in ONE
group, so every sentence has a single clear answer.

(The first version of this project used dair-ai/emotion - see data/dair-ai/ and
the README for why we switched.)
Run with:  .venv/Scripts/python.exe 00_download.py
"""
from pathlib import Path

from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)

# Google's Ekman mapping (from the GoEmotions repository)
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
fine_to_group = {fine: group for group, fines in EKMAN.items() for fine in fines}

ds = load_dataset("google-research-datasets/go_emotions", "simplified")
names = ds["train"].features["labels"].feature.names

for split in ["train", "validation", "test"]:
    df = ds[split].to_pandas()
    df["groups"] = df["labels"].map(lambda ids: {fine_to_group[names[i]] for i in ids})
    single = df[df["groups"].map(len) == 1].copy()
    single["emotion"] = single["groups"].map(lambda g: next(iter(g)))
    single[["text", "emotion"]].to_csv(OUT / f"{split}.csv", index=False)
    print(f"{split:>10}: kept {len(single):,} of {len(df):,} comments (one clear emotion) -> data/{split}.csv")
