# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Compile sentences with the LoRA compiler and write specs/<name>.json in the evolve_morph format.

Usage: python lora_compile.py demo_prompts.json      (GPU env)
"""

import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_train import HERE, MODEL_ID, OUT, decode

torch.set_num_threads(4)


def main(path):
    prompts = json.load(open(path, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, os.path.join(OUT, "adapter")).eval()
    names = list(prompts)
    outs = decode(model, tok, [prompts[n] for n in names])
    for n, o in zip(names, outs):
        try:
            spec = json.loads(o.strip())
        except json.JSONDecodeError:
            print(f"{n:16s} INVALID JSON: {o!r}")
            continue
        spec = {"name": n, "prompt": prompts[n], "compiler": MODEL_ID + " + LoRA", **spec}
        json.dump(spec, open(os.path.join(HERE, "specs", n + ".json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        beh = " ".join(f"{t['weight']:+g}*{t['descriptor']}" for t in spec["behaviour"])
        mor = ", ".join(f"{c['descriptor']} {c['op']} {c['value']:g}" for c in spec["morphology"]) or "(none)"
        print(f"{n:16s} {beh:28s} | {mor:28s} | {prompts[n]}")


if __name__ == "__main__":
    main(sys.argv[1])
