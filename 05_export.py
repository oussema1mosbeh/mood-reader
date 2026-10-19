"""Step 5: export both transformers for the browser, shrink them, and measure the trade-off.

Quantization stores each weight as an 8-bit integer instead of a 32-bit float:
about 4x smaller and faster, usually for a tiny loss in accuracy. We measure
size, speed and accuracy for each model so the choice is based on facts.

The output folders follow the layout Transformers.js expects in the browser:
  export/<name>/config.json, tokenizer.json, ...   and   export/<name>/onnx/model_quantized.onnx

Run with:  .venv/Scripts/python.exe 05_export.py
"""
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

test = pd.read_csv("data/test.csv")
EMOTIONS = sorted(pd.read_csv("data/train.csv").emotion.unique())
y = test.emotion.map({e: i for i, e in enumerate(EMOTIONS)}).values
mb = lambda p: p.stat().st_size / 1e6
rows = []

for name in ["mini", "distilbert"]:
    src, out = Path("models") / name, Path("export") / name
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()

    # 1. Export to ONNX. Batch size and sentence length stay flexible (dynamic axes).
    sample = tok(["an example sentence"], return_tensors="pt")
    inputs = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(inputs, args))).logits

    fp32 = out / "onnx" / "model.onnx"
    torch.onnx.export(Wrapper(model), tuple(sample[k] for k in inputs), fp32, input_names=inputs,
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={**{k: {0: "batch", 1: "sequence"} for k in inputs}, "logits": {0: "batch"}})

    # 2. Quantize to 8-bit integers
    int8 = out / "onnx" / "model_quantized.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

    # Config + tokenizer files next to the model, as Transformers.js expects
    model.config.save_pretrained(out)
    tok.save_pretrained(out)

    # 3. Measure the quantized model: accuracy on the full test set, and speed for one sentence
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]   # e.g. DistilBERT ignores token_type_ids, so the file has no such input
    preds = []
    for i in range(0, len(test), 64):
        enc = tok(list(test.text[i:i + 64]), padding=True, truncation=True, max_length=64, return_tensors="np")
        preds.append(sess.run(None, {k: enc[k].astype(np.int64) for k in inputs})[0].argmax(1))
    pred = np.concatenate(preds)

    one = tok(["They cancelled my flight again and nobody even apologised."], return_tensors="np")
    feed = {k: one[k].astype(np.int64) for k in inputs}
    sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, feed)
    ms = (time.perf_counter() - t0) / 50 * 1000

    with torch.no_grad():   # the original (not quantized) model, for comparison
        full = []
        for i in range(0, len(test), 64):
            enc = tok(list(test.text[i:i + 64]), padding=True, truncation=True, max_length=64, return_tensors="pt")
            full.append(model(**enc).logits.argmax(1).numpy())
    rows.append({"model": name, "original MB": round(mb(fp32)), "quantized MB": round(mb(int8)),
                 "F1 original": f1_score(y, np.concatenate(full), average="macro"),
                 "F1 quantized": f1_score(y, pred, average="macro"), "ms / sentence (CPU)": round(ms, 1)})
    fp32.unlink()   # the browser only needs the quantized file
    print(f"exported {out}/")

table = pd.DataFrame(rows).set_index("model")
print("\n" + table.to_string(formatters={"F1 original": "{:.1%}".format, "F1 quantized": "{:.1%}".format}))
