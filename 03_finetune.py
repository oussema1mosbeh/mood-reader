"""Step 3: fine-tune a pretrained transformer to read emotions.

A transformer like DistilBERT has already read a huge amount of English, so it
knows what words mean in context ("not happy" vs "happy"). Fine-tuning adds a
small 6-emotion layer on top and gently adjusts the whole network to our task.

Run with:  .venv/Scripts/python.exe 03_finetune.py distilbert
           .venv/Scripts/python.exe 03_finetune.py mini
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Two sizes of the same idea: a big accurate one, and a tiny one that fits in a web page.
PRESETS = {
    "distilbert": dict(hub="distilbert-base-uncased", epochs=3, lr=4e-5, batch=32),        # 66M parameters
    "mini": dict(hub="google/bert_uncased_L-4_H-256_A-4", epochs=8, lr=1e-4, batch=32),    # 11M parameters
}
name = sys.argv[1] if len(sys.argv) > 1 else "mini"
cfg = PRESETS[name]
MAX_LEN = 64            # sentences here are short; 64 tokens covers almost all of them
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

train, val = pd.read_csv("data/train.csv"), pd.read_csv("data/validation.csv")
EMOTIONS = sorted(train.emotion.unique())
to_id = {e: i for i, e in enumerate(EMOTIONS)}

# The tokenizer splits text into the pieces the model knows ("unbelievable" -> "un", "##believ", "##able")
tok = AutoTokenizer.from_pretrained(cfg["hub"])
model = AutoModelForSequenceClassification.from_pretrained(
    cfg["hub"], num_labels=len(EMOTIONS),
    id2label=dict(enumerate(EMOTIONS)), label2id=to_id).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{name}: {cfg['hub']} | {n_params / 1e6:.1f}M parameters | training on "
      f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")


def encode(df):
    enc = tok(list(df.text), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor([to_id[e] for e in df.emotion])


tr, va = encode(train), encode(val)

# Class weights: fear and disgust are ~1% of the data. Without help the model would
# almost never predict them, so mistakes on rare emotions are made to count more.
# (Square root keeps it gentle, so common emotions don't get neglected.)
counts = torch.bincount(tr[2], minlength=len(EMOTIONS)).float()
class_weight = ((counts.sum() / (len(EMOTIONS) * counts)) ** 0.5).to(device)
print("class weights:", {e: round(w, 2) for e, w in zip(EMOTIONS, class_weight.tolist())})


@torch.no_grad()
def evaluate(ids, mask, y):
    model.eval()
    preds = []
    for i in range(0, len(ids), 256):
        out = model(input_ids=ids[i:i + 256].to(device), attention_mask=mask[i:i + 256].to(device))
        preds.append(out.logits.argmax(1).cpu())
    p = torch.cat(preds)
    return accuracy_score(y, p), f1_score(y, p, average="macro")


opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
steps = cfg["epochs"] * (len(tr[0]) // cfg["batch"])
sched = get_linear_schedule_with_warmup(opt, int(steps * .06), steps)   # warm up, then slowly lower the learning rate
scaler = torch.amp.GradScaler(enabled=device == "cuda")                  # mixed precision: faster on the GPU

out_dir = Path("models") / name
best, history, start = 0, [], time.time()
for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    order = torch.randperm(len(tr[0]))
    total = 0
    for i in range(len(order) // cfg["batch"]):
        idx = order[i * cfg["batch"]:(i + 1) * cfg["batch"]]
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=tr[0][idx].to(device), attention_mask=tr[1][idx].to(device)).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), tr[2][idx].to(device), weight=class_weight)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
    acc, f1 = evaluate(*va)
    history.append({"epoch": epoch, "loss": total / (len(order) // cfg["batch"]), "val_acc": acc, "val_f1": f1})
    saved = ""
    if f1 > best:   # keep the version with the best macro F1 on sentences it didn't train on
        best = f1
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        saved = "  <- best so far, saved"
    print(f"epoch {epoch:2d} | loss {history[-1]['loss']:.3f} | val accuracy {acc:.1%} | val macro F1 {f1:.1%}{saved}")

secs = time.time() - start
(out_dir / "training.json").write_text(json.dumps(
    {"hub": cfg["hub"], "params": n_params, "seconds": round(secs), "history": history}, indent=1))
print(f"\nBest validation macro F1: {best:.1%}  (trained in {secs:.0f}s) -> {out_dir}/")
