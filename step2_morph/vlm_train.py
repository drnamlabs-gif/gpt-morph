# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""LoRA-tune Qwen2.5-VL-3B to describe a robot's behaviour from the composite image (vlm_data.py).

Split by BODY: every body signature (pairs x segments x joint axes) is in train or test, never both, so the test
asks about bodies the model has never seen. The allowed phrases are listed in the instruction (closed set), the same
instruction for the base model and the tuned one, so both are scored on the same question.

Usage (GPU env): python vlm_train.py            -> vlm/lora/final, vlm/eval_test.json
"""

import hashlib
import json
import os
import random
import sys

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "vlm", "data")
OUT = os.path.join(HERE, "vlm", "lora")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
torch.set_num_threads(8)


def rows():
    return [json.loads(l) for l in open(os.path.join(DATA, "labels.jsonl"))]


def is_test(r):
    return int(hashlib.md5(r["body_sig"].encode()).hexdigest(), 16) % 5 == 0


def instruction(phrases):
    return ("This image summarises 6 seconds of a simulated legged robot. Top left: top view of the torso path "
            "(forward = up, blue arrows = heading, red = the part that counts). Top right: torso height (blue bands = "
            "all feet off the ground) and torso tilt (red dashed line = fall). Bottom: three camera frames. "
            "What does the robot do? Answer with exactly one of: " + "; ".join(phrases) + ".")


class DS(Dataset):
    def __init__(self, rs, proc, phrases):
        self.rs, self.proc, self.ins = rs, proc, instruction(phrases)

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, i):
        r = self.rs[i]
        img = Image.open(os.path.join(DATA, r["file"])).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.ins}]},
                {"role": "assistant", "content": [{"type": "text", "text": r["caption"]}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = self.proc(text=[text], images=[img], return_tensors="pt")
        ids = enc["input_ids"][0]
        labels = ids.clone()
        cap = self.proc.tokenizer(r["caption"], add_special_tokens=False)["input_ids"]
        L = ids.tolist()
        start = next((s for s in range(len(L) - len(cap), -1, -1) if L[s:s + len(cap)] == cap), None)
        if start is None:
            labels[:] = -100
        else:
            labels[:start] = -100
        item = {k: (v[0] if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v) for k, v in enc.items()}
        item["labels"] = labels
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            n = max(v.shape[0] for v in vals)
            fill = -100 if k == "labels" else 0
            out[k] = torch.stack([torch.cat([v, torch.full((n - v.shape[0],), fill, dtype=v.dtype)]) for v in vals])
        else:
            out[k] = torch.cat(vals, 0)
    return out


def answer(model, proc, img_path, ins):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ins}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[Image.open(img_path).convert("RGB")], return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=24, do_sample=False)
    return proc.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().rstrip(".")


def main():
    rs = rows()
    phrases = sorted({r["caption"] for r in rs})
    train = [r for r in rs if not is_test(r)]
    test = [r for r in rs if is_test(r)]
    random.Random(0).shuffle(train)
    print(f"{len(train)} train / {len(test)} test images; {len(phrases)} phrases;"
          f" test bodies {len({r['body_sig'] for r in test})} (none in train)", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=448 * 448)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    if "--eval-only" not in sys.argv:
        args = TrainingArguments(output_dir=OUT, num_train_epochs=2, per_device_train_batch_size=4,
                                 gradient_accumulation_steps=4, learning_rate=1e-4, lr_scheduler_type="cosine",
                                 warmup_ratio=0.03, bf16=True, logging_steps=20, save_strategy="no", report_to=[],
                                 dataloader_num_workers=0, remove_unused_columns=False, seed=0)
        Trainer(model=model, args=args, train_dataset=DS(train, proc, phrases), data_collator=collate).train()
        model.save_pretrained(os.path.join(OUT, "final"))
    else:
        model = PeftModel.from_pretrained(model.base_model.model, os.path.join(OUT, "final"))
    model.eval()
    model.config.use_cache = True
    ins = instruction(phrases)
    res = []
    for r in test:
        p = os.path.join(DATA, r["file"])
        tuned = answer(model, proc, p, ins)
        with model.disable_adapter():
            base = answer(model, proc, p, ins)
        res.append({"id": r["id"], "truth": r["caption"], "lora": tuned, "base": base, "body_sig": r["body_sig"]})
    json.dump({"phrases": phrases, "results": res}, open(os.path.join(HERE, "vlm", "eval_test.json"), "w"), indent=1)
    for who in ("base", "lora"):
        print(f"{who}: {sum(x[who] == x['truth'] for x in res)}/{len(res)} exact on unseen bodies")


if __name__ == "__main__":
    main()
