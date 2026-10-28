# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""LoRA fine-tune a small instruct LLM: sentence -> spec JSON. Then decode the 25 test sentences
with and without the adapter. Runs in the GPU env; grading is done by lora_grade.py.

Usage: python lora_train.py      -> lora/adapter/, lora/predictions.json
"""

import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lora")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED, EPOCHS, BATCH, LR = 0, 3, 8, 2e-4
SYSTEM = ("Translate the request for a legged robot into a spec JSON for the robot-design optimiser. "
          "Behaviour descriptors: dx (forward m), dy (leftward m), yaw (left turn rad), airtime, travel. "
          "Morphology descriptors: n_legs, mean_leg_len, mass. Reply with the JSON only.")
torch.set_num_threads(4)


def chat(tok, text, answer=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt if answer is None else (prompt, answer + tok.eos_token)


def batches(tok, rows, rng):
    rows = rows[:]
    rng.shuffle(rows)
    for i in range(0, len(rows), BATCH):
        ids, labels = [], []
        for r in rows[i:i + BATCH]:
            p, a = chat(tok, r["text"], r["spec"])
            pi, ai = tok(p, add_special_tokens=False).input_ids, tok(a, add_special_tokens=False).input_ids
            ids.append(pi + ai)
            labels.append([-100] * len(pi) + ai)            # loss on the answer only
        n = max(map(len, ids))
        yield (torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x in ids]),
               torch.tensor([x + [-100] * (n - len(x)) for x in labels]),
               torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids]))


def decode(model, tok, texts):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), 8):
        enc = tok([chat(tok, t) for t in texts[i:i + 8]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=160, do_sample=False)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    rows = [json.loads(l) for l in open(os.path.join(OUT, "train.jsonl"), encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot, n = 0.0, 0
        for ids, lab, att in batches(tok, rows, rng):
            loss = model(input_ids=ids.cuda(), labels=lab.cuda(), attention_mask=att.cuda()).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        print(f"epoch {ep}: loss {tot / n:.4f}", flush=True)
    model.save_pretrained(os.path.join(OUT, "adapter"))
    model.eval()

    right = json.load(open(os.path.join(HERE, "paraphrase_right.json"), encoding="utf-8"))
    items = json.load(open(os.path.join(HERE, "paraphrase_body.json"), encoding="utf-8"))["items"]
    texts = [p["text"] for p in right["paraphrases"]] + [it["text"] for it in items]
    tuned = decode(model, tok, texts)
    with model.disable_adapter():
        base = decode(model, tok, texts)
    json.dump({"model": MODEL_ID, "texts": texts, "lora": tuned, "base": base},
              open(os.path.join(OUT, "predictions.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for t, a in zip(texts, tuned):
        print(f"{a.strip()[:110]:110s} | {t}")


if __name__ == "__main__":
    main()
